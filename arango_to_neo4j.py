import os
import logging as log
from abc import ABC, abstractmethod
from threading import Lock
from typing import Any
import json
from tqdm import tqdm

from dotenv import load_dotenv
from arango import ArangoClient
from arango.graph import Graph as ArangoGraph

import constants as c
from utils import read_file, write_file
from neo4j import GraphDatabase


load_dotenv()
log.getLogger().setLevel(log.INFO)


def get_arango_graph() -> ArangoGraph:
    """
    Connects to the arango host and database, and retrieves the graph
    :return: The arango graph object representing the wanted graph
    """
    arango_host = os.getenv(c.ARANGO_HOST)
    db_name = os.getenv(c.ARANGO_DB_NAME)
    db_username = os.getenv(c.ARANGO_USERNAME)
    db_password = os.getenv(c.ARANGO_PASSWORD)
    db_graph_name = os.getenv(c.ARANGO_GRAPH_NAME)

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


def is_neo4j_primitive(obj: Any):
    primitives = str | int | float | None
    return isinstance(obj, primitives) or (isinstance(obj, list) and all([isinstance(v, primitives) for v in obj]))


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


def create_neo4j_node_query(node: dict, node_type: str, merge: bool) -> (str, dict):
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


def create_neo4j_edge_query(edge: dict, src_type: str, dst_type: str, relation: str, merge: bool) -> (str, dict):
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
    query = f'MATCH (src:{src_type} {{ {c.ARANGO_NODE_ID_PROP}: "{edge[c.ARANGO_EDGE_FROM_ID_PROP]}" }} '
    query += f'MATCH (dst:{dst_type} {{ {c.ARANGO_NODE_ID_PROP}: "{edge[c.ARANGO_EDGE_TO_ID_PROP]}" }} '
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
        self.__lock = Lock()

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
        for node_type in arango_graph.vertex_collections():
            if node_type not in self.ignored_node_types:
                self.arango_nodes[node_type] = [arango_node for arango_node in arango_graph.vertex_collection(node_type)]

    def __load_edges_from_host(self, arango_graph: ArangoGraph) -> None:
        """
        Loads only edges from the arango host into memory
        :param arango_graph: The arango graph object
        """
        for edge_def in arango_graph.edge_definitions():
            edge_type = edge_def[c.ARANGO_EDGE_DEF_TYPE_PROP]
            from_node_types = edge_def[c.ARANGO_EDGE_DEF_FROM_TYPE_PROP]
            to_node_types = edge_def[c.ARANGO_EDGE_DEF_TO_TYPE_PROP]
            relevant_node_types = set(from_node_types + to_node_types)
            if (len(relevant_node_types.intersection(set(self.ignored_node_types))) == 0 and
                    edge_type not in self.ignored_edge_types):
                self.arango_edges[edge_type] = [arango_edge for arango_edge in arango_graph.edge_collection(edge_type)]

    def save_graph_to_file(self) -> None:
        assert self.is_loaded, "graph data must be loaded before saving to file"

        file_data = dict()
        file_data[c.GRAPH_FILE_NODES_KEY] = self.arango_nodes
        file_data[c.GRAPH_FILE_EDGES_KEY] = self.arango_edges

        write_file(file_path=self.graph_file_path, data=file_data, is_json=True, create_path_if_not_exist=True)

        log.info(f"Saved {self.__nodes_count()} nodes and {self.__edges_count()} edges to '{self.graph_file_path}'")

    def generate_instructions(self, merge: bool = True) -> None:
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
            transformed_nodes = [self._arango_node_transformer(arango_node=n, node_type=node_type)
                                 for n in tqdm(arango_nodes, desc=f'transforming {node_type} nodes')]
            instructions = [create_neo4j_node_query(node=tn, node_type=node_type, merge=merge)
                            for tn in tqdm(transformed_nodes, desc=f'generating instructions for {node_type} nodes')]
            write_file(file_path=node_type_file_path, data=instructions, is_json=True, create_path_if_not_exist=True)
            log.info(f"Saved {len(instructions)} {node_type} instructions to '{node_type_file_path}'")

        for edge_type, arango_edges in self.arango_edges.items():
            edge_type_file_path = os.path.join(self.instructions_dir_path,
                                               c.INSTRUCTIONS_EDGES_DIR_NAME,
                                               edge_type + c.INSTRUCTIONS_FILE_EXTENSION)
            transformed_edges = [self._arango_edge_transformer(arango_edge=e, edge_type=edge_type)
                                 for e in tqdm(arango_edges, desc=f'transforming {edge_type} edges')]
            instructions = [create_neo4j_edge_query(edge=te, src_type=s, dst_type=d, relation=r, merge=merge)
                            for te, r, s, d in tqdm(transformed_edges, desc=f'generating instructions for {edge_type} edges')]
            write_file(file_path=edge_type_file_path, data=instructions, is_json=True, create_path_if_not_exist=True)
            log.info(f"Saved {len(instructions)} {edge_type} instructions to '{edge_type_file_path}'")

    def build_neo4j(self) -> None:
        """
        Build the Neo4j graph from the instructions files (.cypher)
        """
        neo4j_uri = os.getenv(c.NEO4J_HOST)
        neo4j_auth = (os.getenv(c.NEO4J_BRON_USERNAME), os.getenv(c.NEO4J_BRON_PASSWORD))
        labels_counter = 0

        def execute_instructions(tx, filename: str, instructions: list[tuple[str, dict]]) -> None:
            for query, params in tqdm(instructions, desc=f"Executing instructions from '{filename}'"):
                tx.run(query, **params)

        with GraphDatabase.driver(neo4j_uri, auth=neo4j_auth) as driver:
            driver.verify_connectivity()

            nodes_dir = os.path.join(self.instructions_dir_path, c.INSTRUCTIONS_NODES_DIR_NAME)
            for node_type_filename in os.listdir(nodes_dir):
                node_type_file = os.path.join(nodes_dir, node_type_filename)
                instructions = read_file(file_path=node_type_file, is_json=True)
                with driver.session() as session:
                    def transaction_function(tx): return execute_instructions(tx, node_type_file, instructions)
                    session.write_transaction(transaction_function=transaction_function)

            edges_dir = os.path.join(self.instructions_dir_path, c.INSTRUCTIONS_EDGES_DIR_NAME)
            for edge_type_filename in os.listdir(edges_dir):
                edge_type_file = os.path.join(edges_dir, edge_type_filename)
                instructions = read_file(file_path=edge_type_file, is_json=True)
                with driver.session() as session:
                    def transaction_function(tx): return execute_instructions(tx, edge_type_file, instructions)
                    session.write_transaction(transaction_function=transaction_function)



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

