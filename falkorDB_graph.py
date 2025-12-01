from falkordb import FalkorDB
from symbol_graph_embedding import create_embedding_text, generate_embeddings
from symbol_graph_extractor_script import run_symbol_graph_extractor
from typing import List, Dict, Optional, Any
import os
import json
import time


FALKOR_HOST = os.environ.get("FALKORDB_HOST", "localhost")
FALKOR_PORT = os.environ.get("FALKORDB_PORT", 6379)
JSON_FILE_PATH = "./enriched_symbols.json"
GRAPH_NAME = "solana_knowledge_graph"

def escape_string(s: str) -> str:
    """Escape strings for Cypher queries"""
    if s is None:
        return ""
    return s.replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"')

def parse_relationships(rel_list: List[str]) -> List[Dict[str, str]]:
    """Parse relationship strings into structured data"""
    relationships = []
    if not rel_list:
        return relationships
    
    # Map of relationship prefixes to their types
    rel_mapping = {
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
        for prefix, rel_type in rel_mapping.items():
            if rel.startswith(prefix):
                target = rel.replace(prefix, "").strip()
                relationships.append({"type": rel_type, "target": target})
                break
    
    return relationships

def create_graph_schema(db: FalkorDB):
    """Create indexes for the graph"""
    graph = db.select_graph(GRAPH_NAME)
    
    try:
        # Create indexes for faster lookups
        graph.query("CREATE INDEX FOR (s:Symbol) ON (s.id)")
        graph.query("CREATE INDEX FOR (s:Symbol) ON (s.name)")
        graph.query("CREATE INDEX FOR (m:Module) ON (m.name)")
        
        # Create vector index for embeddings (768 dimensions for Gemini)
        graph.query("""
            CREATE VECTOR INDEX FOR (s:Symbol) ON (s.embedding) 
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
    
    # Process each symbol
    countdown_length = 990
    total = len(symbols) 
    key_number = 1
    key = f"GEMINI_API_KEY_{key_number}"
    countdown = countdown_length
    for idx, symbol in enumerate(symbols[offset:], offset + 1):
        print(f"\nProcessing symbol {idx}/{total}: {symbol.get('name', 'unknown')}")
        # Switch API key every 1000 symbols
        if countdown == 0:
            key_number += 1
            key = f"GEMINI_API_KEY_{key_number}"
            countdown = countdown_length
            print(f"\n🔄 Switching to {key}\n")
        # Create embedding text
        embed_text = create_embedding_text(symbol)
        
        # Get embedding from Gemini
        print("  - Getting embedding from Gemini...")
        embedding = generate_embeddings(embed_text, api_key=key)
        
        if embedding is None:
            print("Skipping symbol due to embedding error")
            countdown -= 1
            continue
        
        # Prepare symbol data
        symbol_id = escape_string(symbol.get('id', ''))
        name = escape_string(symbol.get('name', ''))
        kind = escape_string(symbol.get('kind', ''))
        file_path = escape_string(symbol.get('filePath', ''))
        declaration = escape_string(symbol.get('cleanDeclaration', ''))
        doc_string = escape_string(symbol.get('cleanDocString', ''))
        module_name = escape_string(symbol.get('moduleName', ''))
        
        # Create Symbol node with embedding
        embedding_str = str(embedding)
        
        query = f"""
        MERGE (s:Symbol {{id: '{symbol_id}'}})
        SET s.name = '{name}',
            s.kind = '{kind}',
            s.filePath = '{file_path}',
            s.declaration = '{declaration}',
            s.documentation = '{doc_string}',
            s.moduleName = '{module_name}',
            s.embedding = vecf32({embedding_str})
        """
        
        try:
            graph.query(query)
            print("Symbol node created")
        except Exception as e:
            print(f"ERROR: Error creating symbol: {e}")
            countdown -= 1
            continue
        
        # Create Module node and relationship
        if module_name:
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
        
        # Parse and create relationships
        relationships = parse_relationships(symbol.get('relationships', []))
        for rel in relationships:
            rel_type = rel['type']
            target_name = escape_string(rel['target'])
            
            rel_query = f"""
            MATCH (s:Symbol {{id: '{symbol_id}'}})
            MERGE (t:Symbol {{name: '{target_name}'}})
            MERGE (s)-[:{rel_type}]->(t)
            """
            try:
                graph.query(rel_query)
                print(f"  Relationship {rel_type} created")
            except Exception as e:
                print(f"  ✗ Error creating relationship: {e}")
        
        # Rate limiting to avoid API throttling
        time.sleep(0.6)
        countdown -= 1
        if idx % 10 == 0:
            print(f"\n--- Processed {idx}/{total} symbols ---")
    
    print(f"\n All {total} symbols processed successfully!")
    print(f"\nGraph '{GRAPH_NAME}' is ready for GraphRAG queries!")

def query_similar_symbols(query_text: str, top_k: int = 5):
    """Query for similar symbols using vector similarity"""
    
    # Get embedding for query
    print(f"\nSearching for: {query_text}")
    query_embedding = generate_embeddings(query_text, task_type="RETRIEVAL_QUERY")
    
    if query_embedding is None:
        print("Error getting query embedding")
        return
    
    # Connect to FalkorDB
    db = FalkorDB(host=FALKOR_HOST, port=FALKOR_PORT)
    graph = db.select_graph(GRAPH_NAME)
    
    # Vector similarity search
    embedding_str = str(query_embedding)
    cypher_query = f"""
    CALL db.idx.vector.queryNodes('Symbol', 'embedding', {top_k}, vecf32({embedding_str})) 
    YIELD node, score
    RETURN node.name, node.kind, node.declaration, score
    ORDER BY score DESC
    """
    
    results = graph.query(cypher_query)
    
    print(f"\nTop {top_k} similar symbols:")
    for idx, record in enumerate(results.result_set, 1):
        print(f"\n{idx}. {record[0]} ({record[1]})")
        print(f"   Declaration: {record[2]}")
        print(f"   Similarity: {record[3]:.4f}")


def main():
    # Load JSON data
    # print(f"Loading symbols from {JSON_FILE_PATH}...")
    # with open(JSON_FILE_PATH, 'r') as f:
    #     symbols = json.load(f)

    symbols = run_symbol_graph_extractor()
    
    print(f"Loaded {len(symbols)} symbols")
    input("Continue to load symbols into folkor?: ")
    # Load to FalkorDB

    load_symbols_to_falkordb(symbols)
    
    # Example query
    print("\n" + "="*60)
    print("Example: Querying similar symbols")
    print("="*60)
    query_similar_symbols("error handling enum cases", top_k=5)

if __name__ == "__main__":
    main()