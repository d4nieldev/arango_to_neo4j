# Arango to Neo4j

This mini-project transfers graph databases written in [arango](https://arangodb.com/) to the newer [neo4j](https://neo4j.com/).

## Customize

To implement your own transformer you should create a class that extends [ArangoToNeo4j](arango_to_neo4j.py) and implement the methods `_arango_node_transformer()`, `_arango_edge_transformer()`, `_arango_edge_type_info()`

As an example, look at [BronArangoNeo4j](bron/bron_arango_to_neo4j.py) that transfers the [bron graph](https://github.com/ALFA-group/BRON/tree/master) to a neo4j version of it, along with some data enrichment.

## Usage

### Requirements

To install all the requirements simply with uv run `uv sync`

### Environment Variables

Create a `.env` file at the root project or export the following environment variables:

```bash
# arango login details
ARANGO_HOST={host}:{port}
ARANGO_USERNAME=
ARANGO_PASSWORD=
ARANGO_DB_NAME=
ARANGO_GRAPH_NAME=

# neo4j login details
NEO4J_HOST={host}:{port}
NEO4J_DB_NAME=
NEO4J_BRON_USERNAME=
NEO4J_BRON_PASSWORD=
```

### Run

To run your transformer, run the following sequence of operations:

```python
# instantiate the transformer object
transformer = MyArangoToNeo4j()

# load the graph either from an arango database or from a file if loaded previously using this method
transformer.load_arango_graph(from_file=True, rewrite_file=False)

# generate instructions to be run on an empty neo4j database
transformer.generate_instructions(load_if_exists=True)

# build the neo4j graph from the generated instructions
transformer.build_neo4j_from_instructions()

# verify that the creation was successful, add any new nodes
transformer.verify_build_successful()
```
