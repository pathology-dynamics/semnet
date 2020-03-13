#!bin/bash

# Import CSVs with subject, object, and predicate information into Neo4j database
# Assumes this file is run in the Neo4j directory and the CSV are in this directory

bin/neo4j-import.bat --into data/graph.db --nodes '180621_nodes.txt' --relationships '180618_edges.txt' --delimiter 'TAB'
