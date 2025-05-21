import os
import logging as log
from abc import ABC, abstractmethod
from threading import Lock
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional, List

from dotenv import load_dotenv
from arango.client import ArangoClient
from arango.graph import Graph as ArangoGraph
from neo4j import GraphDatabase, Driver as Neo4jDriver

import constants as c
from utils import read_file, write_file
from tqdm import tqdm


load_dotenv()
log.getLogger().setLevel(log.INFO)
lock = Lock()


def get_arango_graph() -> ArangoGraph:
    """
    Connects to the arango host and database, and retrieves the graph
    :return: The arango graph object representing the wanted graph
    """
    try:
        arango_host = os.environ[c.ARANGO_HOST]
        db_name = os.environ[c.ARANGO_DB_NAME]
        db_username = os.environ[c.ARANGO_USERNAME]
        db_password = os.environ[c.ARANGO_PASSWORD]
        db_graph_name = os.environ[c.ARANGO_GRAPH_NAME]
    except KeyError as e:
        raise EnvironmentError(f"Missing environment variable: {e}")

    arango_client = ArangoClient(hosts=arango_host)
    db = arango_client.db(name=db_name, username=db_username, password=db_password)
    return db.graph(name=db_graph_name)


def get_args_mapping(obj: dict[str, Any]) -> str:
    """
    Generates an args mapping for a given dict, to later use in a neo4j query
    :param obj: the object
    :return: string that represents the object before parameters substitution
    """
    return "{" + ", ".join([f"{k}: ${k}" for k in obj]) + "}"


def make_primitives(obj: dict[str, Any]) -> dict[str, str | int | float | None | list[str]]:
    primitive = str | int | float
    primitives_dict = {}
    for k, v in obj.items():
        if v is None or isinstance(v, primitive):
            primitives_dict[k] = v
        elif isinstance(v, list):
            if all([isinstance(x, primitive) for x in v]):
                primitives_dict[k] = v
            else:
                primitives_dict[k] = [x if isinstance(x, primitive) else json.dumps(x, indent=2) for x in v]
        else:
            primitives_dict[k] = json.dumps(v, indent=2)

    return primitives_dict


def get_neo4j_driver() -> Neo4jDriver:
    try:
        neo4j_uri = os.environ[c.NEO4J_HOST]
        neo4j_auth = (os.environ[c.NEO4J_BRON_USERNAME], os.environ[c.NEO4J_BRON_PASSWORD])
    except KeyError as e:
        raise EnvironmentError(f"Missing environment variable: {e}")

    return GraphDatabase.driver(neo4j_uri, auth=neo4j_auth)


def execute_queries(queries: list[tuple[str, dict]], filename: Optional[str] = None) -> Any:
    if filename:
        pbar = tqdm(total=len(queries), desc=f"Executing instructions from '{filename}'")
    else:
        pbar = tqdm(total=len(queries), desc=f"Executing instructions")

    driver = get_neo4j_driver()

    def run_query(query: str, params: dict):
        with driver.session(database=os.getenv(c.NEO4J_DB_NAME)) as session:
            result = session.run(query, params)
        pbar.update(1)
        return result

    with ThreadPoolExecutor(max_workers=c.MAX_THREADS) as executor:
        futures = [executor.submit(run_query, q, params) for q, params in queries]
        results = [f.result() for f in futures]
    
    pbar.close()
    
    return results


def create_neo4j_node_query(node: dict, node_type: str, merge: bool) -> tuple[str, dict]:
    """
    Creates a query to insert the node to a neo4j database.
    :param node: the JSON representing the node to be inserted
    :param merge: whether to merge the node with an existing node with the same id
    :param node_type: the node type
    :return: a cypher query, and the parameters
    """
    node_args_mapping = get_args_mapping(obj=node)
    query = f'MERGE (n:{node_type} {{ {c.ARANGO_NODE_ID_PROP}: "{node[c.ARANGO_NODE_ID_PROP]}" }}) '
    query += f'ON CREATE SET n = {node_args_mapping} '
    if merge:
        query += f'ON MATCH SET n += {node_args_mapping}'
    else:
        query += f'ON MATCH SET n = {node_args_mapping}'
    return query, make_primitives(node)


def create_neo4j_edge_query(edge: dict, src_type: str, dst_type: str, relation: str, merge: bool) -> tuple[str, dict]:
    """
    Creates a query to insert the edge to a neo4j database.
    :param edge: the JSON representing the edge to be inserted
    :param src_type: the node type of the source node in the edge
    :param dst_type: the node type of the destination node in the edge
    :param relation: few words that describes the connection (written on the edge)
    :param merge: whether to merge the edge with an existing edge from the same type that connects the same nodes
    :return: a cypher query, and the parameters
    """
    edge_args_mapping = get_args_mapping(obj=edge)
    query = f'MATCH (src:{src_type} {{ {c.ARANGO_NODE_ID_PROP}: "{edge[c.ARANGO_EDGE_FROM_ID_PROP]}" }}) '
    query += f'MATCH (dst:{dst_type} {{ {c.ARANGO_NODE_ID_PROP}: "{edge[c.ARANGO_EDGE_TO_ID_PROP]}" }}) '
    query += f'WHERE src IS NOT NULL AND dst IS NOT NULL '
    query += f'MERGE (src)-[r:{relation}]->(dst) '
    query += f'ON CREATE SET r = {edge_args_mapping} '
    if merge:
        query += f'ON MATCH SET r += {edge_args_mapping}'
    else:
        query += f'ON MATCH SET r = {edge_args_mapping}'
    return query, make_primitives(edge)


class ArangoToNeo4j(ABC):
    def __init__(self, graph_name: str, ignored_node_types: tuple[str, ...] = (), ignored_edge_types: tuple[str, ...] = ()):
        self.graph_file_path: str = os.path.join(c.GRAPHS_DIR, f'{graph_name}{c.GRAPH_FILE_EXTENSION}')
        self.instructions_dir_path: str = os.path.join(c.INSTRUCTIONS_DIR, graph_name)
        self.ignored_node_types: tuple[str, ...] = ignored_node_types
        self.ignored_edge_types: tuple[str, ...] = ignored_edge_types

        self.arango_nodes: dict[str, list[dict]] = dict()
        self.arango_edges: dict[str, list[dict]] = dict()
        self.is_loaded: bool = False

    def load_graph(self, from_file: bool = True, rewrite_file: bool = False) -> None:
        """
        Loads the arango graph into memory
        :param from_file: whether to load from existing file (self.graph_file_path)
        :param rewrite_file: whether to rewrite the current graph file with the loaded data
        """
        if from_file and os.path.exists(self.graph_file_path):
            self.__load_graph_from_file()
        else:
            self.__load_graph_from_host()

        self.is_loaded = True
        if rewrite_file:
            self.save_graph_to_file()

    def __nodes_count(self) -> int:
        count = 0
        for nodes in self.arango_nodes.values():
            count += len(nodes)
        return count

    def __edges_count(self) -> int:
        count = 0
        for edges in self.arango_edges.values():
            count += len(edges)
        return count

    def __load_graph_from_file(self) -> None:
        """
        Loads the arango graph from existing file (self.graph_file_path)
        :return:
        """
        graph_data: dict = read_file(file_path=self.graph_file_path, is_json=True)

        assert c.GRAPH_FILE_NODES_KEY in graph_data and c.GRAPH_FILE_EDGES_KEY in graph_data, \
            'incorrect graph file format.'

        self.arango_nodes = graph_data[c.GRAPH_FILE_NODES_KEY]
        self.arango_edges = graph_data[c.GRAPH_FILE_EDGES_KEY]

        log.info(f"Loaded {self.__nodes_count()} nodes and {self.__edges_count()} edges from '{self.graph_file_path}'")

    def __load_graph_from_host(self) -> None:
        """
        Loads the arango graph from the host to the memory
        """
        arango_graph: ArangoGraph = get_arango_graph()

        self.__load_nodes_from_host(arango_graph=arango_graph)
        self.__load_edges_from_host(arango_graph=arango_graph)

        log.info(f"Loaded {self.__nodes_count()} nodes and {self.__edges_count()} edges from host")

    def __load_nodes_from_host(self, arango_graph: ArangoGraph) -> None:
        """
        Loads only nodes from the arango host into memory
        :param arango_graph: The arango graph object
        """
        for node_type in tqdm(arango_graph.vertex_collections(), desc='Loading nodes'):
            if node_type not in self.ignored_node_types:
                self.arango_nodes[node_type] = [arango_node for arango_node in tqdm(arango_graph.vertex_collection(node_type), desc=node_type)]

    def __load_edges_from_host(self, arango_graph: ArangoGraph) -> None:
        """
        Loads only edges from the arango host into memory
        :param arango_graph: The arango graph object
        """
        for edge_def in tqdm(arango_graph.edge_definitions(), desc='Loading edges'):
            edge_type = edge_def[c.ARANGO_EDGE_DEF_TYPE_PROP]
            from_node_types = edge_def[c.ARANGO_EDGE_DEF_FROM_TYPE_PROP]
            to_node_types = edge_def[c.ARANGO_EDGE_DEF_TO_TYPE_PROP]
            relevant_node_types = set(from_node_types + to_node_types)
            if (len(relevant_node_types.intersection(set(self.ignored_node_types))) == 0 and
                    edge_type not in self.ignored_edge_types):
                self.arango_edges[edge_type] = [arango_edge for arango_edge in tqdm(arango_graph.edge_collection(edge_type), desc=edge_type)]

    def save_graph_to_file(self) -> None:
        assert self.is_loaded, "graph data must be loaded before saving to file"

        file_data = dict()
        file_data[c.GRAPH_FILE_NODES_KEY] = self.arango_nodes
        file_data[c.GRAPH_FILE_EDGES_KEY] = self.arango_edges

        write_file(file_path=self.graph_file_path, data=file_data, is_json=True, create_path_if_not_exist=True)

        log.info(f"Saved {self.__nodes_count()} nodes and {self.__edges_count()} edges to '{self.graph_file_path}'")

    def add_embeddings_to_transformed_nodes(self, node_type: str, transformed_nodes: list[dict]) -> list[list[float]] | None:
        """
        Override this function to add embeddings to the transformed nodes, before generating instructions (automatic)
        :param node_type: the node type
        :param transformed_nodes: a chunk transformed nodes (after calling the transformers)
        :return: a list of embeddings that correspond to the transformed nodes (same order), in case no embeddings are needed, return None
        """
        return None

    def generate_nodes_instructions(self, node_type: str, arango_nodes: list[dict], merge: bool) -> list[tuple[str, dict]]:
        transformed_nodes = [self._arango_node_transformer(arango_node=n, node_type=node_type)
                             for n in tqdm(arango_nodes, desc=f'transforming {node_type} nodes')]
        embeddings = self.add_embeddings_to_transformed_nodes(node_type=node_type, transformed_nodes=transformed_nodes)
        if embeddings is not None:
            transformed_nodes = [tn | {c.EMBEDDING_TOKEN: embeddings[i]} for i, tn in enumerate(transformed_nodes)]
        instructions = [create_neo4j_node_query(node=tn, node_type=node_type, merge=merge)
                        for tn in tqdm(transformed_nodes, desc=f'generating instructions for {node_type} nodes')]
        return instructions

    def generate_edges_instructions(self, edge_type: str, arango_edges: list[dict], merge: bool) -> list[tuple[str, dict]]:
        transformed_edges = [self._arango_edge_transformer(arango_edge=e, edge_type=edge_type)
                             for e in tqdm(arango_edges, desc=f'transforming {edge_type} edges')]
        instructions = [create_neo4j_edge_query(edge=te, src_type=s, dst_type=d, relation=r, merge=merge)
                        for te, r, s, d in
                        tqdm(transformed_edges, desc=f'generating instructions for {edge_type} edges')]
        return instructions

    def generate_instructions(self, merge: bool = True, load_if_exists: bool = False) -> None:
        """
        Generates instructions that will create the graph in neo4j to later be executed. Saves all the instructions
        to the instructions folder.
        :param merge:
        :return:
        """
        assert self.is_loaded, "graph data must be loaded before generating instructions"

        for node_type, arango_nodes in self.arango_nodes.items():
            node_type_file_path = os.path.join(self.instructions_dir_path,
                                               c.INSTRUCTIONS_NODES_DIR_NAME,
                                               node_type + c.INSTRUCTIONS_FILE_EXTENSION)
            if load_if_exists and os.path.exists(node_type_file_path):
                log.info(f"File '{node_type_file_path}' already exists, skipping generation of {node_type} nodes")
                continue
            instructions = self.generate_nodes_instructions(node_type=node_type, arango_nodes=arango_nodes, merge=merge)
            write_file(file_path=node_type_file_path, data=instructions, is_json=True, create_path_if_not_exist=True)
            log.info(f"Saved {len(instructions)} {node_type} instructions to '{node_type_file_path}'")

        for edge_type, arango_edges in self.arango_edges.items():
            edge_type_file_path = os.path.join(self.instructions_dir_path,
                                               c.INSTRUCTIONS_EDGES_DIR_NAME,
                                               edge_type + c.INSTRUCTIONS_FILE_EXTENSION)
            if load_if_exists and os.path.exists(edge_type_file_path):
                log.info(f"File '{edge_type_file_path}' already exists, skipping generation of {edge_type} edges")
                continue
            instructions = self.generate_edges_instructions(edge_type=edge_type, arango_edges=arango_edges, merge=merge)
            write_file(file_path=edge_type_file_path, data=instructions, is_json=True, create_path_if_not_exist=True)
            log.info(f"Saved {len(instructions)} {edge_type} instructions to '{edge_type_file_path}'")

    def validate_build_successful(self):
        """
        Checks whether all nodes and all edges were loaded successfully from arango to neo4j
        """
        assert self.is_loaded, "graph data must be loaded before validating that the build was successful"

        log.info("Begin neo4j graph build validation...")

        driver = get_neo4j_driver()

        with driver.session(database=os.getenv(c.NEO4J_DB_NAME)) as session:
            for node_type, nodes in self.arango_nodes.items():
                arango_id_to_node = {node['_id']: node for node in nodes}
                arango_ids = set(arango_id_to_node.keys())
                while True:
                    records = session.run(f"MATCH (n:{node_type}) RETURN n._id")
                    neo4j_ids = set([record[0] for record in records])
                    missing_nodes = [arango_id_to_node[node_id] for node_id in set(arango_ids - neo4j_ids)]
                    excessive_nodes = [arango_id_to_node[node_id] for node_id in set(neo4j_ids - arango_ids)]
                    if len(missing_nodes) == len(excessive_nodes) == 0:
                        log.info(f"All nodes of type '{node_type}' have been created.")
                        break
                    if len(missing_nodes) > 0:
                        log.warning(f"Found {len(missing_nodes)} missing nodes of type '{node_type}'. Adding...")
                        instructions = self.generate_nodes_instructions(node_type=node_type, arango_nodes=missing_nodes, merge=False)
                        execute_queries(queries=instructions)
                    if len(excessive_nodes) > 0:
                        log.warning(f"Found {len(excessive_nodes)} excessive nodes of type '{node_type}'. Deleting...")
                        instructions = [
                            (
                                f"MATCH (n:{node_type} {{ _id: $_id }}) DETACH DELETE n",
                                {'_id': node[c.ARANGO_NODE_ID_PROP]} 
                            )
                            for node in excessive_nodes
                        ]
                        execute_queries(queries=instructions)

            for edge_type, edges in self.arango_edges.items():
                arango_id_to_edge = {edge['_id']: edge for edge in edges}
                arango_ids = set(arango_id_to_edge.keys())
                rel_type, src_types, dst_types = self._arango_edge_type_info(edge_type=edge_type)
                src_types_str = ':' + ':'.join(src_types)
                dst_types_str = ':' + ':'.join(dst_types)
                while True:
                    records = session.run(f"MATCH (src{src_types_str})-[r:{rel_type}]->(dst{dst_types_str}) RETURN r._id")
                    neo4j_ids = set([record[0] for record in records])
                    missing_edges = [arango_id_to_edge[edge_id] for edge_id in set(arango_ids - neo4j_ids)]
                    excessive_edges = [arango_id_to_edge[edge_id] for edge_id in set(neo4j_ids - arango_ids)]
                    if len(missing_edges) == len(excessive_edges) == 0:
                        log.info(f"All edges of type '{edge_type}' have been created.")
                        break
                    if len(missing_edges) > 0:
                        log.info(f"Found {len(missing_edges)} missing edges of type '{edge_type}'! Creating...")
                        instructions = self.generate_edges_instructions(edge_type=edge_type, arango_edges=missing_edges, merge=False)
                        execute_queries(queries=instructions)
                    if len(excessive_edges) > 0:
                        log.warning(f"Found {len(excessive_edges)} excessive edges of type '{edge_type}'! Deleting...")
                        instructions = [
                            (
                                f"MATCH (src{src_types_str})-[r:{rel_type}]->(dst{dst_types_str}) WHERE r._id = $_id DELETE r",
                                {'_id': edge[c.ARANGO_EDGE_ID_PROP]}
                            )
                            for edge in excessive_edges
                        ]
                        execute_queries(queries=instructions)

    def build_neo4j(self, ignore_files: Optional[List[str]] = None) -> None:
        """
        Build the Neo4j graph from the nodes and edges instructions files (.cypher)
        """
        nodes_dir = os.path.join(self.instructions_dir_path, c.INSTRUCTIONS_NODES_DIR_NAME)
        edges_dir = os.path.join(self.instructions_dir_path, c.INSTRUCTIONS_EDGES_DIR_NAME)

        for directory in [nodes_dir, edges_dir]:
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                if ignore_files and file_path in ignore_files:p
                    log.info(f"Ignoring file '{file_path}'")
                    continue
                instructions = read_file(file_path=file_path, is_json=True)
                execute_queries(instructions, file_path)

    @abstractmethod
    def _arango_node_transformer(self, arango_node: dict, node_type: str) -> dict:
        """
        Transforms a given node from the arango db to a dict containing only relevant features.
        DO NOT remove the id field - this field will be used for identifying nodes in the revised edition.
        :param arango_node: a given dict representing a node in the arango graph
        :param node_type: the node type in the arango database
        :return: a dict representing only the relevant features of this node
        """

    @abstractmethod
    def _arango_edge_transformer(self, arango_edge: dict, edge_type: str) -> tuple[dict, str, str, str]:
        """
        Transforms a given edge from the arango db to a dict containing only relevant features.
        DO NOT remove the id field - this field will be used for identifying edges in the revised edition.
        :param arango_edge: a given dict representing an edge in the arango graph
        :param edge_type: the edge type in the arango database
        :return: a dict representing only the relevant features of this edge, a string representing the
                 relation the edge represents, a string representing the source node type, and a string representing
                 the destination node type
        """

    @abstractmethod
    def _arango_edge_type_info(self, edge_type: str) -> tuple[str, list[str], list[str]]:
        """
        Given an edge type, this function returns the source and destination types this edge type represents, and also
        the relation type (what is written on the neo4j edges)
        :param edge_type: the edge collection type
        :return: relation type, list of source node types, list of destination node types
        """

