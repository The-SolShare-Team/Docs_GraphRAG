"""
Swift Symbol GraphRAG Agent
Uses Agno with Cerebras + FalkorDB for hybrid vector + graph search
"""
from falkordb import FalkorDB
from embedding_generation import generate_embeddings
import os
from typing import List, Dict, Any, Optional
import json

from agno.agent import Agent
from agno.models.cerebras import Cerebras
from agno.models.google import Gemini
from agno.tools import Toolkit
from agno.utils.pprint import pprint_run_response
from agno.db.sqlite import SqliteDb
from random import randint

# Configuration
FALKOR_HOST = os.environ.get("FALKORDB_HOST", "localhost")
FALKOR_PORT = os.environ.get("FALKORDB_PORT", 6379)
GRAPH_NAME = os.environ.get("GRAPH_NAME", "solana_knowledge_graph")
NUM_KEYS = os.environ.get("NUM_KEYS", 1)

instructions = """
You are an expert GraphRAG agent specialized in the Solana Swift SDK codebase.

# Your Knowledge Domain
You have deep knowledge of a Swift SDK for Solana blockchain development with 3 main components:

## Core Components
- **SolanaWalletAdapterKit**: Connect deeplink wallets and use provider methods for Solflare, Phantom, and Backpack
- **SolanaRPC**: RPC functions for blockchain interactions (sendTransaction, getBalance, etc.)
- **SolanaTransactions**: Build transactions with systemProgram and other utilities

## Your Expertise
You understand Swift wallet provider protocols, Solana transaction lifecycle, RPC methods, and mobile-specific considerations for wallet connections.

# Your Role
Help developers navigate the codebase, understand implementations, debug issues, build features, and compare with web3.js patterns.

# Your Tool Usage Instructions

You must use these tools to query the knowledge graph before answering:

1. **For concept/functionality questions**: Use `vector_search` with the user's full query
2. **For specific symbol names**: Use `find_by_name` with the exact symbol name
3. **To explore symbol relationships**: First find the symbol ID, then use `graph_traverse`
4. **For module contents**: Use `get_module_symbols` with the module name

**Crucial Rule**: Always use tools first. Synthesize answers based only on returned data. If no results, state that clearly.

# How You Operate
You leverage a knowledge graph of Swift symbols containing declarations, documentation, relationships, and file paths. Provide specific symbol names, file paths, relationships, and practical examples. Be concise but thorough, focusing on actionable information.
"""


class FalkorDBTools(Toolkit):
    """Tools for interacting with FalkorDB knowledge graph"""
    
    def __init__(self, graph_name = GRAPH_NAME, **kwargs):
        self.db = FalkorDB(host=FALKOR_HOST, port=FALKOR_PORT)
        self.graph = self.db.select_graph(graph_name)
        self.schema = self.get_graph_schema_context(self.db, graph_name)
        print(f"✓ Connected to FalkorDB graph: {graph_name}")
        
        tools = [
            self.vector_search,
            self.find_by_name,
            self.graph_traverse,
            self.get_module_symbols
        ]
        super().__init__(name="falkordb_tools", tools=tools, **kwargs)
    
    #pre-hook
    def get_graph_schema_context(self, db: FalkorDB, graph_name: str) -> str:
        """
        Fetches the database schema/ontology and formats it as a string
        for AI agent context.
        """

        graph = db.select_graph(graph_name)

        schema_str = f"Graph Schema for '{graph_name}':\n\n"

        # 1. Node labels and their properties
        labels_query = """
        CALL db.labels() YIELD label
        RETURN label
        """
        labels_result = graph.query(labels_query)

        schema_str += "Nodes:\n"
        for (label,) in labels_result.result_set:
            props_query = f"""
            MATCH (n:{label})
            UNWIND keys(n) AS key
            RETURN DISTINCT key
            LIMIT 100
            """
            props_result = graph.query(props_query)
            props = [row[0] for row in props_result.result_set]
            schema_str += f" - {label}: properties = {props}\n"

        # 2. Relationship types and their properties
        rels_query = "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"
        rels_result = graph.query(rels_query)
        schema_str += "\nRelationships:\n"
        for (rel_type,) in rels_result.result_set:
            rel_props_query = f"""
            MATCH ()-[r:{rel_type}]->()
            UNWIND keys(r) AS key
            RETURN DISTINCT key
            LIMIT 100
            """
            rel_props_result = graph.query(rel_props_query)
            rel_props = [row[0] for row in rel_props_result.result_set]

            if rel_props:
                schema_str += f" - {rel_type}: properties = {rel_props}\n"
            else:
                schema_str += f" - {rel_type} \n"

        return schema_str


    def vector_search(self, query: str, top_k: int = 20) -> str:
        """
        Search for similar Swift symbols using semantic similarity.
        Use this when the user asks about concepts, functionality, or describes what they're looking for.
        
        Args:
            query (str): The search query describing what to find
            top_k (int): Number of results to return. The default value is 20 
        
        Returns:
            Formatted string with symbol information
        """
        # print(f"\n Vector search: '{query}'")
        
        # Get query embedding
        query_embedding = generate_embeddings(
            query, 
            api_key=f"GEMINI_API_KEY_{randint(1, NUM_KEYS)}",
            task_type="RETRIEVAL_DOCUMENT"
        )
        # print("query embedding dimensions:", len(query_embedding))
        if not query_embedding:
            return "Error: Could not generate embedding for query"
        
        # Vector similarity search
        embedding_str = str(query_embedding)
        params = {"embedding": embedding_str, "top_k": top_k}
        cypher_query = """
        CALL db.idx.vector.queryNodes('Symbol', 'embedding', $top_k, vecf32($embedding))
        YIELD node, score
        RETURN node.id, node.name, node.kind, node.declaration, node.documentation, node.filePath, node.moduleName, score
        ORDER BY score DESC
        """
        results = self.graph.query(cypher_query, params=params)

        
        if not results.result_set:
            return "No symbols found matching your query."
        
        # Format results
        formatted = [f"Found {len(results.result_set)} similar symbols:\n"]
        for idx, record in enumerate(results.result_set, 1):
            node_id, name, kind, declaration, documentation, file_path, module_name, score = record

            formatted.append(f"\n{idx}. **{name}** ({kind})")
            formatted.append(f"   Symbol ID: {node_id}")
            formatted.append(f"   Declaration: `{declaration}`")
            
            if documentation:
                formatted.append(f"   Documentation: {documentation}")
            
            formatted.append(f"   File Path: {file_path}")
            formatted.append(f"   Module Name: {module_name}")
            formatted.append(f"   Relevance Score: {score:.3f}")

        return "\n".join(formatted)
    
    def find_by_name(self, name: str) -> str:
        return
    
    def graph_traverse(self, symbol_id: str, relationship_type: Optional[str] = None, direction: str = "outgoing") -> str:
        return
    
    def get_module_symbols(self, module_name: str, limit: int = 100) -> str:
        return


def create_graphrag_agent():
    """Create Agno agent with FalkorDB tools"""  
    # Create agent with Cerebras model
    agent = Agent(
        name="Solana Swift SDK Expert",
        id="graph_rag_agent",
        description="An agent integrated with FalkorDB to do graphRAG.",
        instructions=instructions,
        tools=[
            FalkorDBTools(),
        ],
        model=Cerebras(id="qwen-3-235b-a22b-instruct-2507"),
        # model=Cerebras(id="zai-glm-4.6"),
        # model=Gemini(id="gemini-2.5-flash"),
        markdown=True,
        stream=True,
        stream_events=True,
        debug_mode=False,
        db=SqliteDb(db_file="db/graph_rag_agent.db")
    )
    
    return agent




if __name__ == "__main__":
    print("=" * 60)
print("Swift Symbol GraphRAG Agent")
print("Powered by Agno + Cerebras + FalkorDB")
print("=" * 60)

agent = create_graphrag_agent()

try:
    agent.cli_app(
        input="Hello! Ask me anything about your Solana Swift SDK.",
        user="You",
        emoji="💬",
        stream=True,
    )
except KeyboardInterrupt:
    print("\n\n👋 Goodbye!")
