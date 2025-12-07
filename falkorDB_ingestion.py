import queue
import time
from falkordb import FalkorDB
from embedding_generation import create_embedding_text, generate_embeddings
from scripts.symbol_graph_extractor import run_symbol_graph_extractor
from typing import List, Dict, Optional, Any
import os
import json
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

from types_for_agent import RelationshipType, ParsedRelationship

load_dotenv(override=True)
FALKOR_HOST = os.environ.get("FALKORDB_HOST", "localhost")
FALKOR_PORT = os.environ.get("FALKORDB_PORT", 6379)
JSON_FILE_PATH = "./enriched_symbols.json"
GRAPH_NAME = os.environ.get("GRAPH_NAME", "solana_knowledge_graph")

NUM_KEYS = int(os.environ.get("NUM_GEMINI", 1))
CONCURRENT_WORKERS = max(45, NUM_KEYS)

def escape_string(s: str) -> str:
    """Escape strings for Cypher queries"""
    if s is None:
        return ""
    return s.replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"')

def create_graph_schema(db: FalkorDB):
    graph = db.select_graph(GRAPH_NAME)

    try:
        graph.query("CREATE INDEX FOR (s:Symbol) ON (s.id)")
        graph.query("CREATE INDEX FOR (m:Module) ON (m.name)")
        db.create_constraint(
            GRAPH_NAME, "UNIQUE", "NODE", "Symbol", ["id"]
        )

        db.create_constraint(
            GRAPH_NAME, "UNIQUE", "NODE", "Module", ["name"]
        )

        graph.query("CREATE INDEX ON :Symbol(kind)")
        graph.query("CREATE INDEX ON :Symbol(moduleName)")
        graph.query("CREATE INDEX ON :Symbol(type)")
        graph.query("CREATE INDEX ON :Symbol(returnType)")
        graph.query("CREATE INDEX ON :Symbol(functionSignature)")
        graph.query("CREATE INDEX ON :Module(name)")

        # ✅ Vector index
        graph.query("""
            CREATE VECTOR INDEX ON :Symbol(embedding)
            OPTIONS {dimension: 3072, similarityFunction: 'cosine'}
        """)

        print("Graph schema and indexes created successfully")

    except Exception as e:
        print(f"Note: {e}")


def load_symbols_to_falkordb(symbols: List[Dict[str, Any]], offset = 0):
    """Load symbols with embeddings into FalkorDB"""
    
    # Connect to FalkorDB
    db = FalkorDB(host=FALKOR_HOST, port=FALKOR_PORT)
    graph = db.select_graph(GRAPH_NAME)
    
    print(f"Connected to FalkorDB at {FALKOR_HOST}:{FALKOR_PORT}")
    print(f"Using graph: {GRAPH_NAME}")
    
    # Create schema
    create_graph_schema(db)

    # task queue + workers
    task_queue = queue.Queue()
    for symbol in symbols[offset:]:
        task_queue.put(symbol)

    total = task_queue.qsize()

    progress = {"done": 0}

    def worker(api_key: str):
        while not task_queue.empty():
            try:
                symbol = task_queue.get_nowait()
            except queue.Empty:
                return

            idx = progress["done"] + 1
            print(f"\nProcessing symbol {idx}/{total}: {symbol.get('name', 'unknown')}")
            success = ingest_single_symbol(graph, symbol, api_key)

            if not success:
                print(f"Data ingestion of symbol with id {symbol.get('id', '')} failed")

            progress["done"] += 1

            if progress["done"] % 10 == 0:
                print(f"\n--- Processed {progress['done']}/{total} symbols ---")

            task_queue.task_done()

    with ThreadPoolExecutor(max_workers=CONCURRENT_WORKERS) as executor:
        for i in range(CONCURRENT_WORKERS):
            executor.submit(worker, f"GEMINI_API_KEY_{i}")

    task_queue.join()

    print(f"\n All {total} symbols processed successfully!")
    print(f"\nGraph '{GRAPH_NAME}' is ready for GraphRAG queries!")

def ingest_single_symbol(graph, symbol: Dict[str, Any], api_key: str):
    embed_text = create_embedding_text(symbol)

    print("  - Getting embedding from Gemini...")
    embedding = generate_embeddings(embed_text, api_key=api_key)

    # if embedding is None:
    #     print("Skipping symbol due to embedding error")
    #     return False

    if not create_symbol_node(graph, symbol, embedding):
        print("Creation of node failed")
        return False

    create_module_relationship(graph, symbol)
    create_symbol_relationships(graph, symbol)

    time.sleep(1)
    return True

#helpers
def parse_relationships(rel_list: List[str]) -> List[ParsedRelationship]:
    """Parse relationship strings into structured data. Relationships are specified by id"""
    
    relationships: List[ParsedRelationship] = []
    if not rel_list:
        return relationships

    rel_mapping: dict[str, RelationshipType] = {
        "Is a member of:": "MEMBER_OF",
        "Is an optional member of:": "OPTIONAL_MEMBER_OF",
        "Inherits from:": "INHERITS_FROM",
        "Conforms to:": "CONFORMS_TO",
        "Overrides:": "OVERRIDES",
        "Is a requirement of:": "REQUIREMENT_OF",
        "Is an optional requirement of:": "OPTIONAL_REQUIREMENT_OF",
        "Is a default implementation of:": "DEFAULT_IMPLEMENTATION_OF",
        "Extends:": "EXTENSION_TO",
        "References:": "REFERENCES",
        "Is an overload of:": "OVERLOAD_OF"
    }

    for rel in rel_list:
        rel = rel.strip()
        for prefix, rel_type in rel_mapping.items():
            if rel.startswith(prefix):
                target = rel[len(prefix):].strip()
                relationships.append({
                    "type": rel_type,
                    "target": target
                })
                break

    return relationships

def create_symbol_node(graph, symbol: Dict[str, Any], embedding: list):
    symbol_id = escape_string(symbol.get('id', ''))
    name = escape_string(symbol.get('name', ''))
    kind = escape_string(symbol.get('kind', ''))
    type_ = escape_string(symbol.get('type', ''))
    file_path = escape_string(symbol.get('filePath', ''))
    module_name = escape_string(symbol.get('moduleName', ''))
    declaration = escape_string(symbol.get('cleanDeclaration', ''))
    doc_string = escape_string(symbol.get('cleanDocString', ''))
    generics = escape_string(symbol.get('cleanGenerics', ''))
    function_signature = escape_string(symbol.get('functionSignature', ''))
    parameter_names = symbol.get('parameterNames', [])
    return_type = escape_string(symbol.get('returnType', ''))

    embedding_str = str(embedding)
    parameter_names = [p for p in parameter_names if p is not None]
    parameter_names_array_str = "[" + ",".join(f"'{escape_string(p)}'" for p in parameter_names) + "]"

    query = f"""
    MERGE (s:Symbol {{id: '{symbol_id}'}})
    SET s.name = '{name}',
        s.kind = '{kind}',
        s.type = '{type_}',
        s.filePath = '{file_path}',
        s.declaration = '{declaration}',
        s.documentation = '{doc_string}',
        s.moduleName = '{module_name}',
        s.cleanGenerics = '{generics}',
        s.functionSignature = '{function_signature}',
        s.parameterNames = {parameter_names_array_str},
        s.returnType = '{return_type}',
        s.embedding = vecf32({embedding_str})
    """

    try:
        graph.query(query)
        print("Symbol node created")
        return True
    except Exception as e:
        print(f"ERROR: Error creating symbol: {e}")
        return False

def create_module_relationship(graph, symbol: Dict[str, Any]):
    symbol_id = escape_string(symbol.get('id', ''))
    module_name = escape_string(symbol.get('moduleName', ''))

    if not module_name:
        return

    module_query = f"""
    MERGE (m:Module {{name: '{module_name}'}})
    WITH m
    MATCH (s:Symbol {{id: '{symbol_id}'}})
    MERGE (s)-[:BELONGS_TO_MODULE]->(m)
    """

    try:
        graph.query(module_query)
        print(" Module relationship created")
    except Exception as e:
        print(f" Error creating module: {e}")

def create_symbol_relationships(graph, symbol: Dict[str, Any]):
    symbol_id = escape_string(symbol.get('id', ''))
    relationships = parse_relationships(symbol.get('relationships', []))

    for rel in relationships:
        rel_type = rel['type']
        target_id = escape_string(rel['target'])

        rel_query = f"""
        MATCH (s:Symbol {{id: '{symbol_id}'}})
        MERGE (t:Symbol {{id: '{target_id}'}})
        MERGE (s)-[:{rel_type}]->(t)
        """
        try:
            graph.query(rel_query)
            print(f"  Relationship {rel_type} created")
        except Exception as e:
            print(f" Error creating relationship: {e}")



def ingestion():
    
    if input("Rebuild Swift Project and Update Symbol Graphs based on your latest project in drive? (y/N): ").strip().lower() != "y":
        print(f"Loading symbols from {JSON_FILE_PATH}...")
        with open(JSON_FILE_PATH, 'r') as f:
            symbols = json.load(f)
    else:
        symbols = run_symbol_graph_extractor()
    
    print(f"Loaded {len(symbols)} symbols")
    if input("Continue to load symbols into Falkor? (y/N): ").strip().lower() != "y":
        return
    # Load to FalkorDB

    load_symbols_to_falkordb(symbols)
    
if __name__ == "__main__":
    ingestion()
