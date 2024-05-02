import os
import json
import logging as log
from abc import ABC, abstractmethod
from threading import Lock

from dotenv import load_dotenv
from arango import ArangoClient
from arango.graph import Graph as ArangoGraph

import constants as c
from utils import read_file, write_file

load_dotenv()
log.getLogger().setLevel(log.INFO)


def get_arango_graph() -> ArangoGraph:
    """
    Connects to the arango host and database, and retrieves the graph
    :return: The arango graph object representing the wanted graph
    """
    arango_host = os.environ.get(c.ARANGO_HOST)
    db_name = os.environ.get(c.ARANGO_DB_NAME)
    db_username = os.environ.get(c.ARANGO_USERNAME)
    db_password = os.environ.get(c.ARANGO_PASSWORD)
    db_graph_name = os.environ.get(c.ARANGO_GRAPH_NAME)

    arango_client = ArangoClient(hosts=arango_host)
    db = arango_client.db(name=db_name, username=db_username, password=db_password)
    return db.graph(name=db_graph_name)


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

    def __create_neo4j_node_query(self, node: dict, merge: bool) -> str:
        """
        Creates a query to insert the node to a neo4j database.
        :param node: the JSON representing the node to be inserted
        :param merge: whether to merge the node with an existing node with the same id
        :return: a cypher query
        """
        # TODO implement

    def __create_neo4j_edge_query(self, edge: dict, relation: str, merge: bool) -> str:
        """
        Creates a query to insert the edge to a neo4j database.
        :param edge: the JSON representing the edge to be inserted
        :param relation: few words that describes the connection (written on the edge)
        :param merge: whether to merge the node with an existing node with the same id
        :return: a cypher query
        """
        # TODO implement

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
            transformed_nodes = [self._arango_node_transformer(arango_node=n, node_type=node_type) for n in arango_nodes]
            instructions = [self.__create_neo4j_node_query(node=tn, merge=merge) for tn in transformed_nodes]
            write_file(file_path=node_type_file_path, data=instructions, is_json=False, create_path_if_not_exist=True)
            log.info(f"Generated {len(instructions)} for node type {node_type}, and saved to '{node_type_file_path}'")

        for edge_type, arango_edges in self.arango_edges.items():
            edge_type_file_path = os.path.join(self.instructions_dir_path,
                                               c.INSTRUCTIONS_EDGES_DIR_NAME,
                                               edge_type + c.INSTRUCTIONS_FILE_EXTENSION)
            transformed_edges = [self._arango_edge_transformer(arango_edge=e, edge_type=edge_type) for e in arango_edges]
            instructions = [self.__create_neo4j_edge_query(edge=te, relation=r, merge=merge) for te, r in transformed_edges]
            write_file(file_path=edge_type_file_path, data=instructions, is_json=False, create_path_if_not_exist=True)
            log.info(f"Generated {len(instructions)} for edge type {edge_type}, and saved to '{edge_type_file_path}'")

    def build_neo4j(self) -> None:
        """
        Build the Neo4j graph from the instructions files (.cypher)
        """
        # TODO execute all instructions

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
    def _arango_edge_transformer(self, arango_edge: dict, edge_type: str) -> tuple[dict, str]:
        """
        Transforms a given edge from the arango db to a dict containing only relevant features.
        DO NOT remove the id field - this field will be used for identifying edges in the revised edition.
        :param arango_edge: a given dict representing an edge in the arango graph
        :param edge_type: the edge type in the arango database
        :return: a dict representing only the relevant features of this edge, and a string representing the
                 relation the edge represents
        """

