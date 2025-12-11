import queue
import time
import os
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Any, cast
from enum import Enum

from falkordb import FalkorDB
from dotenv import load_dotenv

from types_for_agent import RelationshipType, ParsedRelationship
from embedding_generation import create_embedding_text, generate_embeddings
from scripts.symbol_graph_extractor import run_symbol_graph_extractor


load_dotenv(override=True)

FALKOR_HOST = os.environ.get("FALKORDB_HOST", "localhost")
FALKOR_PORT = int(os.environ.get("FALKORDB_PORT", 6379))
FALKOR_USER = os.environ.get("FALKORDB_USERNAME", "")
FALKOR_PASSWORD = os.environ.get("FALKORDB_PASSWORD", "")
JSON_FILE_PATH = "./enriched_symbols.json"
GRAPH_NAME = os.environ.get("GRAPH_NAME", "solana_knowledge_graph")

NUM_KEYS = int(os.environ.get("NUM_GEMINI", 1))
CONCURRENT_WORKERS = min(30, NUM_KEYS)

@dataclass
class SymbolNode:
    """
    Represents the raw data structure from the JSON file.
    """
    id: str
    moduleName: str
    name: str
    kind: str
    absolutePath: str
    filePath: str
    accessLevel: str
    cleanDocString: str
    cleanDeclaration: str
    relationships: List[str]  # Raw strings from JSON
    
    # Optionals with defaults
    type: Optional[str] = None
    cleanGenerics: Optional[str] = None
    functionSignature: Optional[str] = None
    returnType: Optional[str] = None
    parameterNames: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict):
        """Safe initializer that handles missing keys gracefully."""
        return cls(
            id=data.get("id", ""),
            moduleName=data.get("moduleName", ""),
            name=data.get("name", ""),
            kind=data.get("kind", ""),
            absolutePath=data.get("absolutePath", ""),
            filePath=data.get("filePath", ""),
            accessLevel=data.get("accessLevel", ""),
            cleanDocString=data.get("cleanDocString", ""),
            cleanDeclaration=data.get("cleanDeclaration", ""),
            relationships=data.get("relationships", []),
            type=data.get("type"),
            cleanGenerics=data.get("cleanGenerics"),
            functionSignature=data.get("functionSignature"),
            returnType=data.get("returnType"),
            parameterNames=data.get("parameterNames") or []
        )

#Helpers
def escape_string(s: str) -> str:
    """Escape strings for Cypher queries"""
    if s is None:
        return ""
    return s.replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"')

def parse_relationships(rel_list: List[str]) -> List[ParsedRelationship]:
    """
    Parse raw relationship strings into structured ParsedRelationship TypedDicts.
    """
    parsed: List[ParsedRelationship] = []
    if not rel_list:
        return parsed

    # Mapping raw string prefixes to your RelationshipType Literal
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
        rel = rel.strip()
        for prefix, rel_type_str in rel_mapping.items():
            if rel.startswith(prefix):
                target = rel[len(prefix):].strip()
                
                # Cast string to your Literal type for type safety
                r_type = cast(RelationshipType, rel_type_str)
                
                # Create the TypedDict
                relationship: ParsedRelationship = {
                    "type": r_type,
                    "target": target
                }
                parsed.append(relationship)
                break

    return parsed

# DB Logic
def create_graph_schema(db: FalkorDB):
    graph = db.select_graph(GRAPH_NAME)

    def create_index_safely(query, description):
        try:
            graph.query(query)
            print(f"  Created: {description}")
        except Exception as e:
            # Check for common "already exists" error messages
            if "already exists" in str(e).lower():
                print(f"  Exists: {description}")
            else:
                print(f"  Failed: {description} | Error: {e}")
    index_queries = [
        ("CREATE INDEX FOR (s:Symbol) ON (s.id)", "Symbol ID Index"),
        ("CREATE INDEX FOR (m:Module) ON (m.name)", "Module Name Index"),
        ("CREATE INDEX ON :Symbol(kind)", "Symbol Kind Index"),
        ("CREATE INDEX ON :Symbol(moduleName)", "Symbol ModuleName Index"),
        ("CREATE INDEX ON :Symbol(type)", "Symbol Type Index"),
        ("CREATE INDEX ON :Symbol(returnType)", "Symbol ReturnType Index"),
        ("CREATE INDEX ON :Symbol(filePath)", "Symbol FilePath Index"),
        ("CREATE INDEX ON :Symbol(functionSignature)", "Symbol Signature Index"),
    ]

    for q, desc in index_queries:
        create_index_safely(q, desc)

    try:
        db.create_constraint(GRAPH_NAME, "UNIQUE", "NODE", "Symbol", ["id"])
        print("  Created: Unique Symbol ID Constraint")
    except Exception:
        print("  Exists: Unique Symbol ID Constraint")

    try:
        db.create_constraint(GRAPH_NAME, "UNIQUE", "NODE", "Module", ["name"])
        print("  Created: Unique Module Name Constraint")
    except Exception:
        print("  Exists: Unique Module Name Constraint")
    
    vector_query = """
        CREATE VECTOR INDEX FOR (s:Symbol) ON (s.embedding)
        OPTIONS {dimension: 3072, similarityFunction: 'cosine'}
    """
    create_index_safely(vector_query, "Vector Index (3072 dim)")

    print("Schema verification complete.\n")

def ingest_single_symbol(graph, symbol: SymbolNode, api_key: str, skip_embedding: bool = False) -> bool:
    # 1. Generate Embedding (unless skipped)
    embedding = None
    
    if not skip_embedding:
        # Convert to dict for the embedding function
        embed_text = create_embedding_text(asdict(symbol))
        
        # IMPORTANT: We allow this to RAISE an exception if it fails.
        embedding = generate_embeddings(embed_text, api_key=api_key)

    if not create_symbol_node(graph, symbol, embedding):
        raise Exception(f"Database write failed for {symbol.id}")

    create_module_relationship(graph, symbol)
    create_symbol_relationships(graph, symbol)

    time.sleep(0.6) 
    return True

def create_symbol_node(graph, symbol: SymbolNode, embedding: Optional[list]) -> bool:
    # Prepare data for Cypher
    props = {
        "id": escape_string(symbol.id),
        "name": escape_string(symbol.name),
        "kind": escape_string(symbol.kind),
        "type": escape_string(symbol.type),
        "absolutePath": escape_string(symbol.absolutePath),
        "filePath": escape_string(symbol.filePath),
        "declaration": escape_string(symbol.cleanDeclaration),
        "documentation": escape_string(symbol.cleanDocString),
        "moduleName": escape_string(symbol.moduleName),
        "cleanGenerics": escape_string(symbol.cleanGenerics),
        "functionSignature": escape_string(symbol.functionSignature),
        "returnType": escape_string(symbol.returnType),
    }

    valid_params = [p for p in symbol.parameterNames if p]
    param_str = "[" + ",".join(f"'{escape_string(p)}'" for p in valid_params) + "]"
    
    embedding_clause = ""
    if embedding:
        embedding_clause = f", s.embedding = vecf32({str(embedding)})"

    query = f"""
    MERGE (s:Symbol {{id: '{props['id']}'}})
    SET 
        s.name = '{props['name']}',
        s.kind = '{props['kind']}',
        s.type = '{props['type']}',
        s.absolutePath = '{props['absolutePath']}',
        s.filePath = '{props['filePath']}',
        s.declaration = '{props['declaration']}',
        s.documentation = '{props['documentation']}',
        s.moduleName = '{props['moduleName']}',
        s.cleanGenerics = '{props['cleanGenerics']}',
        s.functionSignature = '{props['functionSignature']}',
        s.parameterNames = {param_str},
        s.returnType = '{props['returnType']}'
        {embedding_clause}
    """

    try:
        graph.query(query)
        return True
    except Exception as e:
        print(f"ERROR creating symbol {symbol.name}: {e}")
        return False

def create_module_relationship(graph, symbol: SymbolNode):
    if not symbol.moduleName:
        return

    module_name = escape_string(symbol.moduleName)
    symbol_id = escape_string(symbol.id)

    query = f"""
    MERGE (m:Module {{name: '{module_name}'}})
    WITH m
    MATCH (s:Symbol {{id: '{symbol_id}'}})
    MERGE (s)-[:BELONGS_TO_MODULE]->(m)
    """
    try:
        graph.query(query)
    except Exception as e:
        print(f"Error linking module {module_name}: {e}")

def create_symbol_relationships(graph, symbol: SymbolNode):
    # Parse the list of raw strings into typed dictionaries
    rels = parse_relationships(symbol.relationships)
    symbol_id = escape_string(symbol.id)

    for rel in rels:
        # Access TypedDict keys via ['key']
        target_id = escape_string(rel['target'])
        rel_type = rel['type']
        
        query = f"""
        MATCH (s:Symbol {{id: '{symbol_id}'}})
        MERGE (t:Symbol {{id: '{target_id}'}})
        MERGE (s)-[:{rel_type}]->(t)
        """
        try:
            graph.query(query)
        except Exception as e:
            print(f"Error linking relationship {rel_type}: {e}")

def load_symbols_to_falkordb(symbols: List[SymbolNode], offset: int = 0):
    
    db = FalkorDB(host=FALKOR_HOST, port=FALKOR_PORT, username=FALKOR_USER, password=FALKOR_PASSWORD)
    graph = db.select_graph(GRAPH_NAME)
    
    print(f"Connected to FalkorDB at {FALKOR_HOST}:{FALKOR_PORT}")
    print(f"Using graph: {GRAPH_NAME}")
    
    create_graph_schema(db)

    task_queue = queue.Queue()
    for sym in symbols[offset:]:
        task_queue.put(sym)

    total_items = task_queue.qsize()
    progress = {"done": 0}

    key_pool = queue.Queue()
    for i in range(NUM_KEYS):
        key_pool.put(f"GEMINI_API_KEY_{i}")

    total_items = task_queue.qsize()
    progress = {"done": 0}

    print(f" Starting ingestion: {total_items} symbols | {CONCURRENT_WORKERS} threads | {NUM_KEYS} keys rotating")
    def worker():
        while True:
            try:
                symbol : SymbolNode = task_queue.get_nowait()
            except queue.Empty:
                return
            
            max_retries = 25
            for attempt in range(1, max_retries + 1):
                current_api_key = key_pool.get()

                try:
                    ingest_single_symbol(graph, symbol, current_api_key, skip_embedding=False)
                    break # Success
                except Exception as e:
                    if attempt < max_retries:
                        print(f"   Retry {attempt}/{max_retries} for {symbol.name} using new key... (Error: {e})")
                        time.sleep(0.6)
                    else:
                        print(f" Failed embedding for {symbol.name} after {max_retries} tries. Saving metadata only.")
                        try:
                            ingest_single_symbol(graph, symbol, current_api_key, skip_embedding=True)
                        except Exception as final_e:
                            print(f"   CRITICAL: Could not save {symbol.name} even without embedding. Error: {final_e}")
                finally:
                    key_pool.put(current_api_key)
            
            done_count = progress["done"] + 1
            progress["done"] = done_count
            if done_count % 10 == 0:
                print(f"   Ingestion progress: {done_count}/{total_items} - Last Symbol: {symbol.name}")
            
            task_queue.task_done()

    
    with ThreadPoolExecutor(max_workers=CONCURRENT_WORKERS) as executor:
        for _ in range(CONCURRENT_WORKERS):
            executor.submit(worker)

    task_queue.join()
    print(f"\nAll {total_items} symbols processed successfully!")

def ingestion():
    rebuild = input("Rebuild Swift Project and Update Symbol Graphs? (y/N): ").strip().lower() == "y"
    
    raw_data = []
    if not rebuild:
        print(f"Loading symbols from {JSON_FILE_PATH}...")
        try:
            with open(JSON_FILE_PATH, 'r') as f:
                raw_data = json.load(f)
        except FileNotFoundError:
            print("JSON file not found. Please rebuild.")
            return
    else:
        raw_data = run_symbol_graph_extractor()
    
    # Transform Raw Dicts -> SymbolNode Objects
    symbols = [SymbolNode.from_dict(item) for item in raw_data]
    
    print(f"Loaded {len(symbols)} symbols")
    
    if input("Continue to load symbols into Falkor? (y/N): ").strip().lower() == "y":
        load_symbols_to_falkordb(symbols)

if __name__ == "__main__":
    ingestion()