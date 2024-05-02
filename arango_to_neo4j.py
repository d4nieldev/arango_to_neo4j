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
    arango_client = ArangoClient(hosts=arango_host)

    db_name = os.environ.get(c.ARANGO_DB_NAME)
    db_username = os.environ.get(c.ARANGO_USERNAME)
    db_password = os.environ.get(c.ARANGO_PASSWORD)
    db_graph_name = os.environ.get(c.ARANGO_GRAPH_NAME)

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
        :param from_file: Whether to load from existing file (self.graph_file_path)
        :param rewrite_file: Whether to rewrite the current graph file with the loaded data
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

        log.info(f"Loaded {self.__nodes_count()} nodes and {self.__edges_count()} edges from {self.graph_file_path}")

    def __load_graph_from_host(self) -> None:
        """
        Loads the arango graph from the host to the memory
        :param save_after_load: save the graph to a file for faster loading in the future (recommended: True)
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

        log.info(f"Saved {self.__nodes_count()} nodes and {self.__edges_count()} edges to {self.graph_file_path}")

    def generate_instructions(self) -> None:
        assert self.is_loaded, "graph data must be loaded before generating instructions"
        # TODO generate all instructions and write to file (use some kind of CREATE IF NOT EXISTS)
        pass

    def build_neo4j(self) -> None:
        """
        Build the Neo4j graph from the instructions files (.cypher)
        """
        # TODO execute all instructions
        pass

    @abstractmethod
    def _arango_node_transformer(self, arango_node: dict) -> dict:
        """
        Transforms a given node from the arango db to a dict containing only relevant features.
        DO NOT remove the id field - this field will be used for identifying nodes in the revised edition.
        :param arango_node: a given dict representing a node in the arango graph
        :return: a dict representing only the relevant features of this node
        """

    @abstractmethod
    def _arango_edge_transformer(self, arango_edge: dict) -> dict:
        """
        Transforms a given edge from the arango db to a dict containing only relevant features.
        DO NOT remove the id field - this field will be used for identifying edges in the revised edition.
        :param arango_edge: a given dict representing an edge in the arango graph
        :return: a dict representing only the relevant features of this edge
        """

