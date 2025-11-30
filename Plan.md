# GraphRAG Documentation System for Swift SDK

## Project Overview

An automated documentation generation system for a Solana Swift SDK that combines knowledge graph representation with vector embeddings to enable AI-powered, comprehensive documentation generation.

---

## System Architecture

```
┌─────────────────┐
│ Swift Analyzer  │ (SourceKit-LSP)
│ + Vector Embed  │
└────────┬────────┘
         │
         ├─────────────────┐
         │                 │
         ▼                 ▼
┌─────────────────┐  ┌──────────────┐
│  Graphiti API   │  │  Vector DB   │
│   (Neo4j KG)    │  │ (Code chunks)│
└────────┬────────┘  └──────┬───────┘
         │                  │
         └────────┬─────────┘
                  │
                  ▼
         ┌────────────────┐
         │ GraphRAG Agent │
         │  (AI-powered)  │
         └────────────────┘
                  │
                  ▼
         ┌────────────────┐
         │ Documentation  │
         └────────────────┘
```

---

## Core Components

### 1. Swift Code Analyzer

**Purpose:** Extract semantic information from Swift codebase using SourceKit-LSP

**What it does:**
- Starts SourceKit-LSP as a subprocess
- Queries the Language Server Protocol for semantic information
- Extracts entities (classes, functions, properties) with full metadata
- Generates code embeddings for vector database

**Outputs:**
- JSON file with complete code structure and relationships
- Code chunks embedded and sent to vector database

**Key Technology:** SourceKit-LSP (Apple's official Swift language server)

---

### 2. Graphiti REST API Server

**Purpose:** Manage knowledge graph storage and provide query interface

**What it does:**
- Exposes HTTP REST endpoints for code entity ingestion
- Uses Graphiti library to convert entities into graph nodes
- Stores structured relationships in Neo4j
- Provides search endpoints for querying code structure

**Key Endpoints:**
- `POST /add_entity` - Add individual code entities
- `POST /batch_add` - Bulk entity ingestion
- `GET /search` - Query the knowledge graph
- `GET /documentation` - Generate documentation for topics

**Key Technology:** Graphiti + Neo4j

---

### 3. Vector Database

**Purpose:** Store semantic embeddings of code for similarity search

**What it stores:**
- Individual functions/methods with signatures
- Class definitions with context
- Documentation comments
- Usage examples
- Code blocks with surrounding context

**Recommended Options:**
- **Qdrant** - Fast, Rust-based, optimized for code
- **ChromaDB** - Simple, Python-native, embeddable

---

### 4. GraphRAG Documentation Agent

**Purpose:** Combine graph structure with code semantics to generate comprehensive documentation

**Multi-Stage Retrieval Process:**

**Stage 1: Graph Query (Structure)**
- Query knowledge graph for relevant entities
- Understand relationships and dependencies
- Map architectural structure

**Stage 2: Vector Search (Implementation)**
- Semantic search for relevant code chunks
- Find concrete examples and patterns
- Retrieve actual implementations

**Stage 3: Graph Traversal (Relationships)**
- Follow edges to understand call chains
- Map execution flows
- Identify dependencies

**Stage 4: Synthesis**
- AI agent combines all retrieved context
- Generates comprehensive documentation
- Includes architecture diagrams, code examples, and explanations
- Cites sources (files, lines, graph nodes)

---

## What Each Technology Provides

### SourceKit-LSP (Semantic Analysis)
✅ Compiler-accurate type information  
✅ Cross-file relationship understanding  
✅ Symbol resolution (definitions, references)  
✅ Function signatures and return types  
✅ Protocol conformances  
✅ Generic type resolutions  

### Knowledge Graph (Structural Understanding)
✅ Entity relationships (calls, inherits, uses)  
✅ Architectural overview  
✅ Dependency chains  
✅ Type hierarchies  
✅ Module organization  

### Vector Database (Semantic Code Search)
✅ Semantic similarity matching  
✅ Code example retrieval  
✅ Pattern recognition  
✅ Context-aware search  
✅ Implementation details  

### GraphRAG Agent (Intelligent Synthesis)
✅ Multi-source reasoning  
✅ Natural language queries  
✅ Contextual documentation generation  
✅ Example-driven explanations  
✅ Architectural insights  

---

## Complete Data Flow

### Phase 1: Initial Setup (One-time)
1. Install Neo4j database
2. Deploy Graphiti REST API server
3. Set up vector database (Qdrant/ChromaDB)
4. Build Swift SDK to generate SourceKit-LSP index

### Phase 2: Parse Codebase (Run on updates)
1. Swift analyzer starts SourceKit-LSP
2. For each Swift file:
   - Query for symbols, types, relationships
   - Extract semantic information
   - Generate embeddings for code chunks
3. Output JSON structure with complete metadata
4. Send to both Graphiti API and Vector DB

### Phase 3: Ingest into Knowledge Systems
1. Graphiti API receives code structure JSON
2. Creates nodes for entities (classes, functions, etc.)
3. Creates edges for relationships (calls, uses, inherits)
4. Deduplicates and enriches automatically
5. Vector DB stores embedded code chunks with metadata

### Phase 4: Generate Documentation
1. User requests documentation: "Explain transaction building"
2. GraphRAG agent executes multi-stage retrieval:
   - Queries knowledge graph for structure
   - Searches vector DB for implementations
   - Traverses graph for relationships
3. AI synthesizes results into documentation:
   - Architecture diagrams from graph
   - Code examples from vector DB
   - Explanations combining both sources
4. Outputs markdown/HTML with citations

---

## Key Advantages of This Approach

### Superior to Pure Vector RAG
- ✅ Graph provides structural understanding vectors miss
- ✅ Explicit relationships vs. just semantic similarity
- ✅ Can answer architectural questions accurately

### Superior to Pure Knowledge Graph
- ✅ Vectors provide semantic search capability
- ✅ Actual code examples, not just metadata
- ✅ Handles nuanced, meaning-based queries

### Combined GraphRAG Benefits
- ✅ **Semantic code search** - Find patterns without exact keywords
- ✅ **Concrete examples** - Real code backing every concept
- ✅ **Structural understanding** - Big picture from knowledge graph
- ✅ **AI-quality output** - Explanatory prose, not just facts
- ✅ **Handles complexity** - Nuanced questions like "What's the difference between X and Y?"

---

## Example Use Cases

### Query: "Document how to build and send a transaction"

**Knowledge Graph provides:**
- Transaction class structure and methods
- Relationships: Transaction → CALLS → RpcClient.send()
- Dependencies: Uses Account, Keypair, Message

**Vector DB provides:**
- Actual usage examples: `let tx = Transaction(...)`
- Real implementation code
- Inline documentation and comments

**AI Agent generates:**
- Complete guide with architecture overview
- Step-by-step code examples
- Best practices and common patterns

### Query: "What are all the RPC methods?"

**Knowledge Graph provides:**
- Complete list of RpcClient methods
- Method relationships and call chains
- Parameter and return types

**Vector DB provides:**
- Implementation of each method
- Usage examples from tests
- Documentation comments

**AI Agent generates:**
- Comprehensive RPC reference
- Grouped by functionality
- With examples for each method

---

## Implementation Checklist

### Component 1: Swift Code Analyzer
- [ ] Set up SourceKit-LSP integration
- [ ] Implement LSP protocol communication (JSON-RPC)
- [ ] Extract entities and relationships
- [ ] Generate code embeddings
- [ ] Output JSON structure

### Component 2: Graphiti API Server
- [ ] Set up Neo4j database
- [ ] Install and configure Graphiti
- [ ] Create FastAPI REST endpoints
- [ ] Implement entity ingestion logic
- [ ] Add search and query endpoints

### Component 3: Vector Database
- [ ] Choose and install vector DB (Qdrant/ChromaDB)
- [ ] Define chunking strategy for code
- [ ] Implement embedding pipeline
- [ ] Set up metadata filtering
- [ ] Test semantic search queries

### Component 4: GraphRAG Agent
- [ ] Design multi-stage retrieval logic
- [ ] Implement graph query integration
- [ ] Implement vector search integration
- [ ] Build AI synthesis pipeline
- [ ] Add source citation system
- [ ] Create documentation templates

---

## Technical Recommendations

### Embedding Strategy
**What to embed:**
- Functions with full signatures and docstrings
- Class definitions with immediate context
- Usage examples from tests
- Documentation blocks
- Knowledge graph facts themselves (as text)

**Why embed graph facts:**
- Enables semantic search of relationships
- Example: "What validates transactions?" finds "Transaction.sign() calls validation helpers"

### Chunking Best Practices
- Keep function boundaries intact
- Include surrounding context (3-5 lines)
- Preserve documentation comments
- Use entity names as metadata for filtering

### Source Citation
- Reference specific files and line numbers
- Link to graph nodes for relationships
- Include confidence scores for AI-generated content
- Make documentation actionable and verifiable

---

## Future Enhancements

- **Incremental updates**: Only re-parse changed files
- **Multi-version support**: Compare API versions in knowledge graph
- **Interactive exploration**: Web UI for graph visualization
- **Usage analytics**: Track which docs are queried most
- **Code generation**: Use agent to generate SDK usage code
- **Test generation**: Auto-generate tests from knowledge graph

---

## Technology Stack Summary

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Code Analysis | SourceKit-LSP | Semantic Swift understanding |
| Knowledge Graph | Neo4j + Graphiti | Structural relationships |
| Vector Storage | Qdrant/ChromaDB | Semantic code search |
| API Layer | FastAPI (Python) | REST interface |
| AI Agent | GraphRAG pattern | Documentation synthesis |
| Output Format | Markdown/HTML | Human-readable docs |

---

## Getting Started

1. **Set up infrastructure**: Neo4j, Vector DB, Graphiti API
2. **Build Swift analyzer**: Implement SourceKit-LSP integration
3. **Parse codebase**: Run analyzer on Solana Swift SDK
4. **Test queries**: Verify both graph and vector retrieval work
5. **Build agent**: Implement multi-stage GraphRAG pipeline
6. **Generate docs**: Start with simple topics, iterate on quality

---

## Success Metrics

- **Coverage**: % of codebase entities in knowledge graph
- **Accuracy**: Correctness of generated documentation
- **Retrieval quality**: Relevance of graph + vector results
- **Response time**: End-to-end documentation generation speed
- **Usefulness**: Developer feedback on documentation quality
