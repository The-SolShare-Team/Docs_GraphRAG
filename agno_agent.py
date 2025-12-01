"""
Swift Symbol GraphRAG Agent
Uses Agno with Cerebras + FalkorDB for hybrid vector + graph search
"""

from falkordb import FalkorDB
from symbol_graph_embedding import generate_embeddings
import os
from typing import List, Dict, Any, Optional
import json

from agno.agent import Agent
from agno.models.cerebras import Cerebras
from agno.tools import Toolkit
from agno.utils.pprint import pprint_run_response
from agno.db.sqlite import SqliteDb

# Configuration
FALKOR_HOST = os.environ.get("FALKORDB_HOST", "localhost")
FALKOR_PORT = os.environ.get("FALKORDB_PORT", 6379)
GRAPH_NAME = "solana_knowledge_graph"


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
    
    def __init__(self, **kwargs):
        self.db = FalkorDB(host=FALKOR_HOST, port=FALKOR_PORT)
        self.graph = self.db.select_graph(GRAPH_NAME)
        print(f"✓ Connected to FalkorDB graph: {GRAPH_NAME}")
        
        tools = [
            self.vector_search,
            self.find_by_name,
            self.graph_traverse,
            self.get_module_symbols
        ]
        super().__init__(name="falkordb_tools", tools=tools, **kwargs)
    def vector_search(self, query: str, top_k: int = 25) -> str:
        """
        Search for similar Swift symbols using semantic similarity.
        Use this when the user asks about concepts, functionality, or describes what they're looking for.
        
        Args:
            query (str): The search query describing what to find
            top_k (int): Number of results to return. Your default value should be 25. You should aim to return a lot as each vector is a small piece of code.
        
        Returns:
            Formatted string with symbol information
        """
        print(f"\n Vector search: '{query}'")
        
        # Get query embedding
        query_embedding = generate_embeddings(
            query, 
            api_key="GEMINI_API_KEY_1",
            task_type="RETRIEVAL_DOCUMENT"
        )
        print("query embedding dimensions:", len(query_embedding))
        if not query_embedding:
            return "Error: Could not generate embedding for query"
        
        # Vector similarity search
        embedding_str = str(query_embedding)
        cypher_query = f"""
        CALL db.idx.vector.queryNodes('Symbol', 'embedding', {top_k}, vecf32({embedding_str})) 
        YIELD node, score
        RETURN node.id, node.name, node.kind, node.declaration, 
               node.documentation, node.filePath, score
        ORDER BY score DESC
        """
        
        results = self.graph.query(cypher_query)
        
        if not results.result_set:
            return "No symbols found matching your query."
        
        # Format results
        formatted = [f"Found {len(results.result_set)} similar symbols:\n"]
        for idx, record in enumerate(results.result_set, 1):
            formatted.append(f"\n{idx}. **{record[1]}** ({record[2]})")
            formatted.append(f"   Declaration: `{record[3]}`")
            if record[4]:
                doc_preview = record[4][:200] + "..." if len(record[4]) > 200 else record[4]
                formatted.append(f"   Documentation: {doc_preview}")
            formatted.append(f"   File: {record[5]}")
            formatted.append(f"   Relevance Score: {record[6]:.3f}")
            formatted.append(f"   Symbol ID: {record[0]}")
        
        return "\n".join(formatted)
    
    def find_by_name(self, name: str) -> str:
        """
        Find Swift symbols by exact or partial name match.
        Use when the user mentions a specific symbol name.
        
        Args:
            name (str): The symbol name or partial name to search for
        
        Returns:
            Formatted string with symbol information
        """
        print(f"\n🔎 Name search: '{name}'")
        
        cypher_query = f"""
        MATCH (s:Symbol)
        WHERE s.name CONTAINS '{name}'
        RETURN s.id, s.name, s.kind, s.declaration, s.documentation, s.filePath
        LIMIT 10
        """
        
        results = self.graph.query(cypher_query)
        
        if not results.result_set:
            return f"No symbols found with name containing '{name}'"
        
        # Format results
        formatted = [f"Found {len(results.result_set)} symbols matching '{name}':\n"]
        for idx, record in enumerate(results.result_set, 1):
            formatted.append(f"\n{idx}. **{record[1]}** ({record[2]})")
            formatted.append(f"   Declaration: `{record[3]}`")
            if record[4]:
                doc_preview = record[4][:200] + "..." if len(record[4]) > 200 else record[4]
                formatted.append(f"   Documentation: {doc_preview}")
            formatted.append(f"   File: {record[5]}")
            formatted.append(f"   Symbol ID: {record[0]}")
        
        return "\n".join(formatted)
    
    def graph_traverse(self, symbol_id: str, relationship_type: Optional[str] = None) -> str:
        """
        Explore relationships of a specific symbol (inheritance, conformance, member relationships, etc.).
        
        Args:
            symbol_id (str): The unique ID of the symbol to explore
            relationship_type (str): Optional filter by relationship type (
    "MEMBER_OF",
    "OPTIONAL_MEMBER_OF",
    "INHERITS_FROM",
    "CONFORMS_TO",
    "OVERRIDES",
    "REQUIREMENT_OF",
    "OPTIONAL_REQUIREMENT_OF",
    "DEFAULT_IMPLEMENTATION_OF",
    "EXTENSION_TO",
    "REFERENCES",
    "OVERLOAD_OF"
)
        
        Returns:
            JSON string with symbol and its relationships
        """
        print(f"\n🕸️  Graph traverse from: {symbol_id}")
        
        cypher_query = f"""
        MATCH (s:Symbol {{id: '{symbol_id}'}})
        OPTIONAL MATCH (s)-[r]->(related:Symbol)
        RETURN s.name, s.kind, s.declaration, s.documentation,
               collect({{
                   type: type(r),
                   name: related.name,
                   kind: related.kind,
                   declaration: related.declaration,
                   id: related.id
               }}) as relationships
        """
        
        results = self.graph.query(cypher_query)
        
        if not results.result_set:
            return f"Symbol with ID '{symbol_id}' not found"
        
        record = results.result_set[0]
        result = {
            "name": record[0],
            "kind": record[1],
            "declaration": record[2],
            "documentation": record[3],
            "relationships": [r for r in record[4] if r.get('name')]
        }
        
        # Filter by relationship type if specified
        if relationship_type and result["relationships"]:
            result["relationships"] = [
                r for r in result["relationships"] 
                if r.get("type") == relationship_type
            ]
        
        # Format output
        formatted = [f"Symbol: **{result['name']}** ({result['kind']})"]
        formatted.append(f"Declaration: `{result['declaration']}`")
        if result['documentation']:
            formatted.append(f"Documentation: {result['documentation'][:300]}...")
        
        if result['relationships']:
            formatted.append(f"\nRelationships ({len(result['relationships'])}):")
            for rel in result['relationships']:
                formatted.append(f"  - {rel['type']}: {rel['name']} ({rel['kind']})")
                formatted.append(f"    Declaration: `{rel['declaration']}`")
                formatted.append(f"    ID: {rel['id']}")
        else:
            formatted.append("\nNo relationships found.")
        
        return "\n".join(formatted)
    
    def get_module_symbols(self, module_name: str, limit: int = 100) -> str:
        """
        Get all symbols from a specific Swift module.
        
        Args:
            module_name: The name of the module
            limit: Maximum number of symbols to return (default: 20)
        
        Returns:
            Formatted string with symbols in the module
        """
        print(f"\n📦 Module search: '{module_name}'")
        
        cypher_query = f"""
        MATCH (s:Symbol)-[:BELONGS_TO_MODULE]->(m:Module {{name: '{module_name}'}})
        RETURN s.id, s.name, s.kind, s.declaration, s.documentation
        LIMIT {limit}
        """
        
        results = self.graph.query(cypher_query)
        
        if not results.result_set:
            return f"No symbols found in module '{module_name}'"
        
        # Format results
        formatted = [f"Found {len(results.result_set)} symbols in module '{module_name}':\n"]
        for idx, record in enumerate(results.result_set, 1):
            formatted.append(f"\n{idx}. **{record[1]}** ({record[2]})")
            formatted.append(f"   Declaration: `{record[3]}`")
            if record[4]:
                doc_preview = record[4][:150] + "..." if len(record[4]) > 150 else record[4]
                formatted.append(f"   Documentation: {doc_preview}")
            formatted.append(f"   Symbol ID: {record[0]}")
        
        return "\n".join(formatted)




def create_graphrag_agent():
    """Create Agno agent with FalkorDB tools"""  
    # Create agent with Cerebras model
    agent = Agent(
        name="Solana Swift SDK Expert",
        id="graph_rag_agent",
        description="An agent integrated with FalkorDB to do graphRAG.",
        model=Cerebras(id="llama3.3-70b"),
        instructions=instructions,
        tools=[
            FalkorDBTools(),
        ],
        markdown=True,
        stream=True,
        stream_events=True,
        debug_mode=False,
        db=SqliteDb(db_file="db/graph_rag_agent.db")
    )
    
    return agent


def main():
    """Interactive GraphRAG chat interface"""
    print("=" * 60)
    print("Swift Symbol GraphRAG Agent")
    print("Powered by Agno + Cerebras + FalkorDB")
    print("=" * 60)
    
    agent = create_graphrag_agent()
    
    print("\n💡 Ask me anything about your Solana Swift SDK!")
    print("Examples:")
    print("  - Give me an example code snippet on how to create a transaction and send it")
    print("  - Are there any other ways to send a transaction?")
    print("  - What wallet providers are available?")
    print("  - How do I connect to Phantom wallet?")
    print("  - Show me RPC methods for getting account info")
    print("  - What symbols are in the Solana module?")
    print("\nType 'quit' to exit.\n")
    
    while True:
        try:
            user_input = input("\n💬 You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            # Run agent
            agent.print_response(user_input)
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()