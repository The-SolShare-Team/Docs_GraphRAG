"""
Swift Symbol GraphRAG Agent
Uses Agno with Cerebras + FalkorDB for hybrid vector + graph search
"""
from falkordb import FalkorDB
from embedding_generation import generate_embeddings
import os
from typing import List, Dict, Any, Optional, Union, get_args
from functools import wraps
import cohere
from dotenv import load_dotenv

from agno.agent import Agent
from agno.models.cerebras import Cerebras
from agno.tools import Toolkit
from agno.db.sqlite import SqliteDb
from random import randint

from types_for_agent import *
from falkorDB_ingestion import ingestion

load_dotenv(override=True)

# Configuration
FALKOR_HOST = os.environ.get("FALKORDB_HOST", "localhost")
FALKOR_PORT = os.environ.get("FALKORDB_PORT", 6379)
GRAPH_NAME = os.environ.get("GRAPH_NAME", "solana_knowledge_graph")
NUM_KEYS = os.environ.get("NUM_KEYS", 1)

instructions = """
You are an expert GraphRAG agent specialized in navigating Swift SDK codebases through a knowledge graph.

# Your Role
Help developers navigate codebases, understand implementations, debug issues, and build features by intelligently querying and traversing a graph of code symbols.

# Your Tool Strategy

## Entry Points: Getting Into the Graph
You have TWO ways to enter the graph and find starting symbols:

1. **vector_search(query)**: Semantic search for concepts/functionality
   - Use for: "How do I connect a wallet?", "transaction signing", "RPC methods"
   - Returns symbols ranked by semantic similarity
   - Best for exploratory or concept-based queries

2. **query_symbols(filters)**: Property-based filtering
   - Use for: Finding specific symbol types, modules, or names
   - Examples: `{"kind": "class"}`, `{"moduleName": "SolanaRPC"}`, `{"name_contains": "Transaction"}`
   - Best for targeted searches with known properties

## Exploration: Traversing the Graph
Once you have a symbol, explore its connections:

3. **graph_traverse(symbol_id, relationship_types, direction, result_filters)**: Follow relationships
   - Use after finding a starting symbol to explore its context
   - Examples:
     - Find class members: `graph_traverse(id, "MEMBER_OF", "incoming")`
     - Find what a function calls: `graph_traverse(id, "REFERENCES", "outgoing")`
     - Filter results: `graph_traverse(id, "MEMBER_OF", "incoming", result_filters={"kind": "method"})`
   - Can traverse multiple hops with `depth` parameter

## Your Search Strategy

**Standard Workflow:**
1. **Enter** the graph using vector_search OR query_symbols
2. **Traverse** from those symbols using graph_traverse to gather context
3. **Repeat traversal** from different symbols until you have sufficient information
4. **Synthesize** your answer from all gathered data

**If stuck after multiple attempts:**
- Re-enter the graph with a different entry method
- Try different search terms or filters
- Adjust relationship types or traversal direction

**Critical Rules:**
- ALWAYS use tools before answering - never guess
- Cite specific symbols, file paths, and relationships from tool results
- If tools return no results after multiple attempts, clearly state information is not found
- Combine multiple tool calls to build comprehensive answers

# Response Style
Be concise but thorough. Provide:
- Specific symbol names and their locations (file paths)
- Code declarations when relevant
- Relationship context (what inherits/implements/references what)
- Practical examples drawn from the actual codebase
"""


class FalkorDBTools(Toolkit):
    """Tools for interacting with FalkorDB knowledge graph"""
    
    def __init__(self, graph_name = GRAPH_NAME, NUM_KEYS : str = NUM_KEYS, **kwargs):
        self.graph_name = graph_name
        self.NUM_KEYS = int(NUM_KEYS)
        self.db = FalkorDB(host=FALKOR_HOST, port=FALKOR_PORT)
        self.graph = self.db.select_graph(graph_name)
        self.schema = self.get_graph_schema_context(graph_name)


        # print(self.schema.model_dump_json(indent=4))
        print(f"✓ Connected to FalkorDB graph: {graph_name}")
        
        tools = [
            self.vector_search,
            self.query_symbols,
            self.graph_traverse,
            # self.get_module_symbols,
            self.clear_graph,
            self.populate_graph,
        ]
        # instructions="" #to add
        super().__init__(name="falkordb_tools", tools=tools, **kwargs) #instructions=instructions
    #inject SymbolKind and RelationshipType information
    @staticmethod
    def inject_schema_info(func):
        """Decorator to inject schema information into docstrings"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        # Get schema values
        symbol_kinds = list(get_args(SymbolKind))
        relationship_types = list(get_args(RelationshipType))
        
        symbol_properties = []
        for field_name, field_info in Symbol.model_fields.items():
            if field_name not in ['embedding', 'id']:
                description = field_info.description or ''
                symbol_properties.append(f"{field_name}: {description}")
        
        properties_formatted = '\n            '.join(symbol_properties)
        
        # Inject into docstring
        if func.__doc__:
            doc = func.__doc__
            doc = doc.replace("{symbol_kinds}", ', '.join(f'"{k}"' for k in symbol_kinds))
            doc = doc.replace("{relationship_types}", ', '.join(f'"{r}"' for r in relationship_types))
            doc = doc.replace("{symbol_properties}", properties_formatted)
            wrapper.__doc__ = doc
        
        return wrapper

    #pre-hook
    def get_graph_schema_context(self, graph_name: Optional[str] = None) -> GraphSchema:
        """
        Fetches the database schema/ontology in a structured format for AI agent context.
        Returns kinds of symbols and relationship types as those are special properties.
        """
        graph_name = graph_name or self.graph_name
        graph = self.db.select_graph(graph_name)
        # --- Symbol Node ---
        symbol_props_query = """
        MATCH (s:Symbol)
        UNWIND keys(s) AS key
        RETURN DISTINCT key
        """
        symbol_props_result = graph.query(symbol_props_query)
        symbol_properties = [row[0] for row in symbol_props_result.result_set if row[0] is not None]

        symbol_kinds_query = "MATCH (s:Symbol) RETURN DISTINCT s.kind"
        symbol_kinds_result = graph.query(symbol_kinds_query)
        symbol_kinds = [row[0] for row in symbol_kinds_result.result_set if row[0] is not None]

        symbol_schema = SymbolSchema(
            properties=symbol_properties,
            kinds=symbol_kinds
        )

        module_query = "MATCH (m:Module) RETURN DISTINCT m.name"
        module_result = graph.query(module_query)
        module_names = [row[0] for row in module_result.result_set if row[0] is not None]

        # --- Relationship Types ---
        rels_query = "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"
        rels_result = graph.query(rels_query)
        relationships = [row[0] for row in rels_result.result_set]

        return GraphSchema(
            symbol_schema=symbol_schema,
            module_names=module_names,
            relationships=relationships
        )

    def clear_graph(self, graph_name: Optional[str]) -> str:
        """
        Clear all nodes and relationships from a graph. Only call when user explicitly says to delete the graph.
        DANGER: This will delete ALL data in the graph.
        
        Args:
            graph_name (str): Name of the graph to clear. If you are unsure of the name, leave this parameter blank.
                The toolkit is enabled with a default graph name.
        Returns:
            str: Confirmation message about the clearing operation.
        """
        target_graph = graph_name or self.graph_name
        
        # Request confirmation
        print(f"\n  WARNING: You are about to delete ALL data from graph '{target_graph}'!")
        print("This action cannot be undone.")
        confirmation = input("Type 'DELETE' to confirm: ")
        
        if confirmation != "DELETE":
            return f" Graph clearing cancelled. You must type 'DELETE' to confirm."
        
        # Perform the deletion
        graph_to_clear = self.db.select_graph(target_graph)
        
        # Delete all relationships first, then all nodes
        delete_query = "MATCH (n) DETACH DELETE n"
        result = graph_to_clear.query(delete_query)
        
        return f" Successfully cleared graph '{target_graph}'. All nodes and relationships have been deleted."
    
    def populate_graph(self):
        """
        Repopulates the graph using a prebuilt script.
        
        """
        ingestion()

    def vector_search(self, 
        query: str, 
        top_k: int = 10,
        rerank: bool = True,
        initial_retrieval_multiplier: Optional[int] = 3) -> List[VectorSearchResult]:
        """
        Search for similar Swift symbols using semantic similarity. 
        Never all unless the user explicitly says they want to populate the graph.

        Args:
            query (str): The search query describing what to find.
            top_k (int): Number of results to return. Default is 10.
            rerank (bool): Whether to apply reranking (default True)
            initial_retrieval_multiplier (Optional[int]): How many candidates to retrieve 
                before reranking (default 3x final results)

        Returns:
            List[VectorSearchResult]: A list of dictionaries containing the structured Symbol
                and its similarity score, e.g.,
                [{"symbol": Symbol(...), "score": 0.93}, ...]
        """
        
        # Determine retrieval size
        if rerank:
            retrieve_k = (top_k * initial_retrieval_multiplier)
        else:
            retrieve_k = top_k
        

        # Generate embedding
        query_embedding = generate_embeddings(
            query,
            api_key=f"GEMINI_API_KEY_{randint(1, self.NUM_KEYS)}",
            task_type="RETRIEVAL_QUERY"
        )

        if not query_embedding:
            return []

        # Vector similarity search
        params = {"embedding": query_embedding, "top_k": retrieve_k}
        cypher_query = """
        CALL db.idx.vector.queryNodes('Symbol', 'embedding', $top_k, vecf32($embedding))
        YIELD node, score
        RETURN node.id, node.name, node.kind, node.type, node.declaration, node.documentation,
               node.filePath, node.moduleName, node.cleanGenerics, node.functionSignature,
               node.parameterNames, node.returnType, score
        ORDER BY score DESC
        """

        results = self.graph.query(cypher_query, params=params)
        symbols_with_scores: List[Dict[str, object]] = []
        for record in results.result_set:
            (
                node_id, name, kind, type, declaration, documentation,
                filePath, moduleName, cleanGenerics, functionSignature,
                parameterNames, returnType, score
            ) = record

            symbol = Symbol(
                # id=node_id,
                name=name,
                kind=kind,
                type=type,  
                filePath=filePath,
                moduleName=moduleName,
                documentation=documentation,
                declaration=declaration,
                # cleanGenerics=cleanGenerics,
                functionSignature=functionSignature,
                parameterNames=parameterNames,
                returnType=returnType,
            )
            symbols_with_scores.append({"symbol": symbol, "score": score})
        
        if rerank and len(symbols_with_scores) > 0:
            symbols_with_scores = self._rerank_results(
                query, symbols_with_scores, top_k
            )
        else:
            symbols_with_scores = symbols_with_scores[:top_k]
        return symbols_with_scores
    
    def _rerank_results(self, query: str, results: List[dict], top_k: int, ) -> List[dict]:
        """
        Rerank results using Cohere's reranker.
        
        Args:
            query: Original search query
            results: List of {symbol, score, initial_rank} dicts
            top_k: Number of final results to return
            
        Returns:
            Reranked list of results with updated scores
        """
        cohere_api_key = os.getenv("COHERE_API_KEY")
        try:
            if not cohere_api_key:
                raise ValueError("Missing Cohere API key")
            co = cohere.Client(cohere_api_key)
        except Exception as e:
            print(f"Failed to initialize Cohere: {e}. Skipping reranking.")
            return results[:top_k]
        
        documents = []
        for result in results:
            symbol = result["symbol"]

            doc_parts = [
                f"Name: {symbol.name}",
                f"Kind: {symbol.kind}",
                f"Module: {symbol.moduleName}",
            ]
            
            if symbol.type:
                doc_parts.append(f"Type: {symbol.type}")
            
            if symbol.declaration:
                doc_parts.append(f"Declaration: {symbol.declaration}")
            
            if symbol.functionSignature:
                doc_parts.append(f"Signature: {symbol.functionSignature}")
            
            if symbol.documentation:
                # Truncate very long docs for reranking
                doc = symbol.documentation[:500]
                doc_parts.append(f"Documentation: {doc}")
            
            documents.append("\n".join(doc_parts))

        try:
            rerank_response = co.rerank(
                query=query,
                documents=documents,
                top_n=top_k,
                model="rerank-v3.5",  # "rerank-english-v3.0"
            )

            reranked_results = []
            for rank_result in rerank_response.results:
                original_idx = rank_result.index
                reranked_results.append({
                    "symbol": results[original_idx]["symbol"],
                    "score": rank_result.relevance_score  # Replace with reranker score
                })

            return reranked_results
        except Exception as e:
            print(f" Reranking failed: {e}. Returning original results.")
            return results[:top_k]

    def parse_json_string(self, json_str: str) -> dict | None:
        """
        Parse a JSON string that may be wrapped in markdown code blocks.
        
        Args:
            json_str: A JSON string, possibly wrapped in ```json``` code blocks
            
        Returns:
            Parsed dictionary if successful, None otherwise
        """
        try:
            # Remove markdown code block wrapper if present
            cleaned = json_str.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned.replace("```json", "", 1)
            if cleaned.startswith("```"):
                cleaned = cleaned.replace("```", "", 1)
            if cleaned.endswith("```"):
                cleaned = cleaned.rsplit("```", 1)[0]
            cleaned = cleaned.strip()
            
            # Parse the JSON
            parsed = json.loads(cleaned)
            
            # Validate it's a dictionary
            if not isinstance(parsed, dict):
                print(f"❌ Error: JSON must be an object (dictionary). Got: {type(parsed).__name__}")
                return None
                
            return parsed
            
        except json.JSONDecodeError as e:
            print(f"❌ Error: Could not decode JSON string. {e}")
            print(f"Received string: {json_str}")
            return None
    
    def build_where_statement(self, json_str, params: Dict[str, Any]) -> str:
        filters = {}
        if json_str:
            filters = self.parse_json_string(json_str)

        # Build WHERE clauses dynamically
        where_clauses = []
        param_counter = 0
        
        # Properties that should use case-insensitive partial match
        case_insensitive_props = {"kind", "documentation", "declaration", "name","moduleName"}
        # Properties that should use case-sensitive partial match (default)
        case_sensitive_props = {"type", "filePath", "returnType", 
                            "functionSignature", "cleanGenerics"}
        # Array property
        array_props = {"parameterNames"}
        
        for key, value in filters.items():
            param_name = f"param_{param_counter}"
            param_counter += 1
            
            # Handle special operators
            if key.endswith("_contains"):
                # Explicit partial match
                base_key = key[:-9]  # Remove "_contains" suffix
                if base_key in case_insensitive_props:
                    where_clauses.append(f"toLower(s.{base_key}) CONTAINS toLower(${param_name})")
                else:
                    where_clauses.append(f"s.{base_key} CONTAINS ${param_name}")
                params[param_name] = value
                
            elif key.endswith("_in"):
                # Multiple values (OR condition with partial matching)
                base_key = key[:-3]  # Remove "_in" suffix
                if not isinstance(value, list):
                    value = [value]
                
                # Create OR conditions for each value with partial matching
                or_conditions = []
                for i, val in enumerate(value):
                    sub_param = f"{param_name}_{i}"
                    if base_key in case_insensitive_props:
                        or_conditions.append(f"toLower(s.{base_key}) CONTAINS toLower(${sub_param})")
                    else:
                        or_conditions.append(f"s.{base_key} CONTAINS ${sub_param}")
                    params[sub_param] = val
                
                where_clauses.append(f"({' OR '.join(or_conditions)})")
                
            elif key.endswith("_any"):
                # Check if any element in array contains value
                base_key = key[:-4]  # Remove "_any" suffix
                if base_key in array_props:
                    where_clauses.append(f"ANY(param IN s.{base_key} WHERE param CONTAINS ${param_name})")
                    params[param_name] = value
                else:
                    # Fall back to regular contains for non-array properties
                    where_clauses.append(f"s.{base_key} CONTAINS ${param_name}")
                    params[param_name] = value
                    
            else:
                # Default behavior: partial match based on property type
                if key in array_props:
                    # For array properties without suffix, check if any element contains value
                    if isinstance(value, list):
                        # Multiple values to check against array
                        or_conditions = []
                        for i, val in enumerate(value):
                            sub_param = f"{param_name}_{i}"
                            or_conditions.append(f"ANY(param IN s.{key} WHERE param CONTAINS ${sub_param})")
                            params[sub_param] = val
                        where_clauses.append(f"({' OR '.join(or_conditions)})")
                    else:
                        # Single value to check against array
                        where_clauses.append(f"ANY(param IN s.{key} WHERE param CONTAINS ${param_name})")
                        params[param_name] = value
                        
                elif key in case_insensitive_props:
                    # Case-insensitive partial match
                    where_clauses.append(f"toLower(s.{key}) CONTAINS toLower(${param_name})")
                    params[param_name] = value
                    
                elif key in case_sensitive_props:
                    # Case-sensitive partial match
                    where_clauses.append(f"s.{key} CONTAINS ${param_name}")
                    params[param_name] = value
                    
                else:
                    # Default to case-sensitive partial match for unknown properties
                    where_clauses.append(f"s.{key} CONTAINS ${param_name}")
                    params[param_name] = value
        
        # Construct the WHERE clause
        where_statement = " AND ".join(where_clauses) if where_clauses else "true"
        return where_statement

    @inject_schema_info
    def query_symbols(self, filters_json: Optional[str], limit: int = 50) -> List[Symbol]:
        """
        Query symbols by any combination of properties. All matches use partial/fuzzy matching.

        Args:
            filters_json (Optional[str]): A JSON string representing a dictionary of property:value pairs for filtering.
                Do not use a dictionary object, provide a JSON string.
                
                Available properties include: name, kind, type, moduleName, documentation, etc.
                
                Operators:
                - Use "_contains" for partial string matching.
                - Use "_in" to match any value in a list.
                - Use "_any" to search within array properties.
                
                Example JSON strings:
                - '{"name_contains": "Transaction"}'
                - '{"kind_in": ["Class", "Struct"]}'
                - '{"moduleName_contains": "Foundation", "kind": "function"}'
                - '{"parameterNames_any": "completion"}'
                
                If no filters are needed, omit this parameter or provide an empty JSON string '{}'.
            
            limit (int): Max results (default 50)
        
        Returns:
            List[Symbol]: Matching symbols
        """

        params = {"limit": limit}
        where_statement = self.build_where_statement(filters_json, params=params)

        # Build Cypher query
        cypher_query = f"""
        MATCH (s:Symbol)
        WHERE {where_statement}
        RETURN s.id, s.name, s.kind, s.type, s.declaration, s.documentation,
            s.filePath, s.moduleName, s.cleanGenerics, s.functionSignature,
            s.parameterNames, s.returnType
        LIMIT $limit
        """
        # Execute query
        try:
            # print(f"Generated Cypher query:\n{cypher_query}")
            # print(f"Parameters: {params}")
            results = self.graph.query(cypher_query, params=params)
        except Exception as e:
            print(f"❌ Query execution failed: {e}")
            print(f"Query: {cypher_query}")
            print(f"Params: {params}")
            return []
        
        # Parse results into Symbol objects
        symbols: List[Symbol] = []
        for record in results.result_set:
            (
                node_id, name, kind, type, declaration, documentation,
                filePath, moduleName, cleanGenerics, functionSignature,
                parameterNames, returnType,
            ) = record
            
            symbol = Symbol(
                id=node_id,
                name=name,
                kind=kind,
                type=type,
                filePath=filePath,
                moduleName=moduleName,
                documentation=documentation,
                declaration=declaration,
                cleanGenerics=cleanGenerics,
                functionSignature=functionSignature,
                parameterNames=parameterNames,
                returnType=returnType,
            )
            symbols.append(symbol)
        return symbols

    @inject_schema_info
    def graph_traverse(
        self,
        symbol_id: str,
        relationship_types: Optional[List[str]],
        direction: str = "outgoing",
        result_filters_json: Optional[str] = None, 
        depth: int = 1,
        limit: int = 50
    ) -> List[GraphTraversalResult]:
        """
        Traverse relationships from a symbol node and optionally filter the connected symbols. BFS-style.
        
        Args:
            symbol_id (str): Starting symbol node ID
            relationship_types (List[str]): Filter by relationship type (partial match). 
                Valid types: {relationship_types}
                Example: "MEMBER_OF", "INHERITS_FROM", "REFERENCES"
            direction (str): Traversal direction
                - "outgoing": Follow relationships from this symbol
                - "incoming": Follow relationships to this symbol  
                - "both": Follow in both directions
            result_filters_json (Optional[str]): A JSON string representing a dictionary of property:value pairs to filter the final connected symbols. Do not use a dictionary object, provide a JSON string.
            Example JSON strings:
                - '{"kind": "function"}'
                - '{"type": "Bool", "kind_in": ["method", "property"]}'
            
            If no filters are needed, omit this parameter or provide an empty JSON string '{}'
            depth (int): Number of relationship hops (default 1, max 5 for safety)
            limit (int): Max results (default 50)
        
        Returns:
            List[GraphTraversalResult]: A list of traversal results, each containing:
                - symbol: The connected Swift symbol
                - relationships: List of relationship types in the traversal path
                - relationship_directions: List of directions for each relationship 
                - depth: Number of hops from the starting symbol
        
        Examples:
            # Get all members of a class
            graph_traverse(symbol_id="MyClass_123", relationship_types="MEMBER_OF", direction="incoming")
            
            # Get all methods that are members of a class
            graph_traverse(symbol_id="MyClass_123", "MEMBER_OF", "incoming", 
                        result_filters={{"kind": "method"}})
            
            # Find what a function references, but only Bool-returning functions
            graph_traverse("myFunc_123", "REFERENCES", "outgoing",
                        result_filters={{"kind": "function", "returnType": "Bool"}})
        """

        if depth > 5:
            depth = 5  # Safety limit to prevent expensive queries
        
        # Build relationship direction clause
        if direction == "outgoing":
            rel_pattern = f"-[r*1..{depth}]->"
        elif direction == "incoming":
            rel_pattern = f"<-[r*1..{depth}]-"
        elif direction == "both":
            rel_pattern = f"-[r*1..{depth}]-"
        else:
            print(f"❌ Invalid direction: {direction}. Use 'outgoing', 'incoming', or 'both'")
            return []
        
        # Build WHERE clauses
        where_clauses = ["start.id = $symbol_id"]
        params = {"symbol_id": symbol_id, "limit": limit}
        # Filter by relationship type (partial match for flexibility)
        if relationship_types:
            where_clauses.append("ANY(t IN [rel IN relationships(path) | type(rel)] WHERE t IN $relationship_types)")
            params["relationship_types"] = relationship_types
        # Build filters for connected symbols (reuse logic from query_symbols)
        where_clauses.append(self.build_where_statement(result_filters_json, params=params))
        where_statement = " AND ".join(where_clauses)
        
        # Build Cypher query with variable depth and direction tracking
        cypher_query = f"""
MATCH (start:Symbol)
WHERE start.id = $symbol_id
MATCH path = (start){rel_pattern}(connected:Symbol)
WHERE {where_statement}
WITH connected, 
     relationships(path) as rels, 
     nodes(path) as nodes, 
     length(path) as depth
WITH connected, rels, nodes, depth,
     [rel IN rels | type(rel)] as relationship_types,
     [i IN range(0, size(rels)-1) | 
        CASE WHEN startNode(rels[i]) = nodes[i] THEN 'outgoing' ELSE 'incoming' END
     ] as relationship_directions
RETURN DISTINCT 
    connected.id, connected.name, connected.kind, connected.type, 
    connected.declaration, connected.documentation,
    connected.filePath, connected.moduleName, connected.cleanGenerics, 
    connected.functionSignature, connected.parameterNames, connected.returnType, 
    relationship_types,
    relationship_directions,
    depth
ORDER BY depth, connected.name
LIMIT $limit
"""
        # Execute query
        try:
            results = self.graph.query(cypher_query, params=params)
        except Exception as e:
            print(f"❌ Graph traversal failed: {e}")
            print(f"Query: {cypher_query}")
            print(f"Params: {params}")
            return []
        
        # Parse results
        traversal_results = []
        for record in results.result_set:
            (
                node_id, name, kind, type_, declaration, documentation,
                filePath, moduleName, cleanGenerics, functionSignature,
                parameterNames, returnType, relationship_types, relationship_directions, depth_val
            ) = record
            
            symbol = Symbol(
                id=node_id, name=name,
                kind=kind, type=type_, filePath=filePath, moduleName=moduleName,
                documentation=documentation, declaration=declaration, cleanGenerics=cleanGenerics,
                functionSignature=functionSignature, parameterNames=parameterNames, returnType=returnType,
                # embedding=embedding if embedding else []
            )
            
            traversal_results.append({
                "symbol": symbol,
                "relationships": relationship_types,  # Path of relationship types
                "directions": relationship_directions,  # Direction for each relationship
                "depth": depth_val
            })
        
        return traversal_results
        
        # def get_module_symbols(self, module_name: str, limit: int = 100) -> str:
        #     return


def create_graphrag_agent(graph_name = GRAPH_NAME, debug_mode = False, debug_level = 2, model="qwen-3-235b-a22b-instruct-2507"):
    """Create Agno agent with FalkorDB tools"""  
    # Create agent with Cerebras model
    dbTools = FalkorDBTools()
    agent = Agent(
        name="Solana Swift SDK Expert",
        id="graph_rag_agent",
        description="An agent integrated with FalkorDB to do graphRAG.",
        instructions=instructions,
        tools=[
            dbTools,
        ],
        model=Cerebras(id=model,
            retries=5,
            delay_between_retries=1,
            exponential_backoff=True,
            api_key=os.environ.get("CEREBRAS_API_KEY")
        ),
        # dependencies={"graph_name" : graph_name},
        # model=Cerebras(id="zai-glm-4.6"),
        # model=Gemini(id="gemini-2.5-flash"),
        pre_hooks=[dbTools.get_graph_schema_context],
        markdown=True,
        stream=True,
        stream_events=True,
        debug_mode=debug_mode,
        debug_level=debug_level,
        db=SqliteDb(db_file="db/graph_rag_agent.db"),
        add_history_to_context=True,
        num_history_runs=2,
        read_chat_history=True,
        # num_history_sessions=2 # prev sessions
    )
    
    return agent




if __name__ == "__main__":
    print("=" * 60)
    print("Swift Symbol GraphRAG Agent")
    print("Powered by Agno + Cerebras + FalkorDB")
    print("=" * 60)

    agent = create_graphrag_agent(debug_level=2, debug_mode=True)

    try:
        agent.cli_app(
            user="You",
            emoji="💬",
            stream=True,
        )
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
