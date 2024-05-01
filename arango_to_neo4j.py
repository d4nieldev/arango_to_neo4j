import os
import logging as log
from typing import Callable
from abc import ABC, abstractmethod

from dotenv import load_dotenv
from arango import ArangoClient
from arango.graph import Graph as ArangoGraph

import constants as c

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
    def __init__(self, graph_filename: str, ignored_node_types: list[str], ignored_edge_types: list[str]):
        self.path_to_graph_file: str = os.path.join(c.PATH_TO_GRAPHS_DIR, graph_filename)
        self.ignored_node_types: list[str] = ignored_node_types
        self.ignored_edge_types: list[str] = ignored_edge_types

        self.arango_nodes: dict[str, list[dict]] = dict()
        self.arango_edges: dict[str, list[dict]] = dict()
        self.__load_graph()

    def __load_graph(self, from_existing: bool = True) -> None:
        if from_existing and os.path.exists(self.path_to_graph_file):
            self.__load_graph_from_existing()
        else:
            self.__load_graph_from_host(save_after_load=True)

    def __load_graph_from_existing(self) -> None:
        # TODO use the same logic as in save_graph_to_file to load from the file
        pass

    def __load_graph_from_host(self, save_after_load: bool) -> None:
        """
        Loads the arango graph from the host to the memory
        :param save_after_load: save the graph to a file for faster loading in the future (recommended: True)
        """
        arango_graph: ArangoGraph = get_arango_graph()

        for node_type in arango_graph.vertex_collections():
            if node_type not in self.ignored_node_types:
                self.arango_nodes[node_type] = [arango_node for arango_node in arango_graph.vertex_collection(node_type)]

        for edge_def in arango_graph.edge_definitions():
            edge_type = edge_def[c.ARANGO_EDGE_DEF_TYPE_PROP]
            from_node_type = edge_def[c.ARANGO_EDGE_DEF_FROM_TYPE_PROP]
            to_node_type = edge_type[c.ARANGO_EDGE_DEF_TO_TYPE_PROP]
            if from_node_type not in self.ignored_node_types and \
                    to_node_type not in self.ignored_node_types and \
                    edge_type not in self.ignored_edge_types:
                self.arango_edges[edge_type] = [arango_edge for arango_edge in arango_graph.edge_collection(edge_type)]

        if save_after_load:
            self.save_graph_to_file()

    def save_graph_to_file(self):
        # TODO use the same logic as in __load_graph_from_existing to save to the file
        pass

    def __generate_instructions(self):
        # TODO generate all instructions (use some kind of CREATE IF NOT EXISTS)
        pass

    def build_neo4j(self, instructions_file: str):
        """
        Build the Neo4j graph from an instructions file (.cypher)
        :return:
        """
        # TODO execute all instructions
        pass

    @abstractmethod
    def arango_node_transformer(self, arango_node: dict) -> dict:
        """
        Transforms a given node from the arango db to a dict containing only relevant features.
        DO NOT remove the id field - this field will be used for identifying nodes in the revised edition.
        :param arango_node: a given dict representing a node in the arango graph
        :return: a dict representing only the relevant features of this node
        """

    @abstractmethod
    def arango_edge_transformer(self, arango_edge: dict) -> dict:
        """
        Transforms a given edge from the arango db to a dict containing only relevant features.
        DO NOT remove the id field - this field will be used for identifying edges in the revised edition.
        :param arango_edge: a given dict representing an edge in the arango graph
        :return: a dict representing only the relevant features of this edge
        """

