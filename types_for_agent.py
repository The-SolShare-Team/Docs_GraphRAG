import json
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal, TypedDict


SymbolKind = Literal[
    "associated type",
    "case",
    "class",
    "enumeration",
    "initializer",
    "instance method",
    "instance property",
    "macro",
    "operator",
    "protocol",
    "structure",
    "type alias",
    "type method",
    "type property",
    "func.op",
    "init",
    "method",
    "property",
    "struct",
    "subscript",
    "typealias"
]

RelationshipType = Literal[
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
    "OVERLOAD_OF",
    "BELONGS_TO_MODULE",
]

Direction = Literal[
    "incoming",
    "outgoing"
]

class Symbol(BaseModel):
    id: Optional[str] = Field(None, description="Unique identifier for the symbol node.")
    name: Optional[str] = Field(None, description="The name of the symbol (e.g., function, class, variable).")
    kind: Optional[str] = Field(None, 
        description="The kind of symbol. Must be one of the predefined SymbolKind literals (e.g., 'Class', 'Instance Method')."
    )
    type: Optional[str] = Field(None, 
        description="The data type of the symbol, if applicable (e.g., for variables or return types)."
    )
    declaration: Optional[str] = Field(None, description="The full declaration of the symbol, as it appears in code.")
    documentation: Optional[str] = Field(None, description="The documentation string or comments associated with the symbol.")
    filePath: Optional[str] = Field(None, description="The file path where the symbol is defined.")
    moduleName: Optional[str] = Field(None, description="The module or namespace the symbol belongs to.")
    cleanGenerics: Optional[str] = Field(None, description="The generic parameters of the symbol, cleaned and simplified.")
    functionSignature: Optional[str] = Field(None, description="The function or method signature, if the symbol is callable.")
    parameterNames: Optional[List[str]] = Field(None, description="List of parameter names for functions or methods.")
    returnType: Optional[str] = Field(None, description="The return type for functions or methods, if applicable.")
    embedding: Optional[List[float]] = Field(None, description="The semantic vector embedding of the symbol used for similarity search.")

    class Config:
        extra = "forbid"

class SymbolSchema(BaseModel):
    properties: List[str] = Field(
        ...,
        description="List of all properties present on Symbol nodes."
    )
    kinds: List[str] = Field(
        ...,
        description="All unique values for the 'kind' property of Symbol nodes."
    )
    class Config:
        extra = "forbid"

class GraphSchema(BaseModel):
    symbol_schema: SymbolSchema = Field(
        ...,
        description="Schema information for Symbol nodes, including properties and kinds."
    )
    module_names: List[str] = Field(
        ...,
        description="List of module names in the project."
    )
    relationships: List[str] = Field(
        ...,
        description="All relationship types in the graph. Defines semantic connections between nodes."
    )
    class Config:
        extra = "forbid"

# SET s.name = '{name}',
#         s.kind = '{kind}',
#         s.type = '{type_}',
#         s.filePath = '{file_path}',
#         s.declaration = '{declaration}',
#         s.documentation = '{doc_string}',
#         s.moduleName = '{module_name}',
#         s.cleanGenerics = '{generics}',
#         s.functionSignature = '{function_signature}',
#         s.parameterNames = {parameter_names_array_str},
#         s.returnType = '{return_type}',
#         s.embedding = vecf32({embedding_str})
#     """

class VectorSearchResult(BaseModel):
    """Result from vector search containing a symbol and its similarity score."""
    
    symbol: Symbol = Field(
        ...,
        description="The matched Swift symbol with all its properties including name, kind, type, etc."
    )
    
    score: float = Field(
        ...,
        ge=0.0,  # Similarity scores should be non-negative
        le=1.0,  # Typically similarity scores are 0-1
        description="Similarity score between the query and the symbol (0.0 to 1.0, higher is more similar)"
    )

class TraversalStep(BaseModel):
    """Represents a single step in a graph traversal path."""
    node: "Symbol" = Field(
        ...,
        description="The Swift symbol visited at this step"
    )
    relationship_type: str = Field(
        ...,
        description="The type of relationship used to reach the next node"
    )
    direction: str = Field(
        ...,
        description="Direction of the relationship from this node's perspective ('outgoing' or 'incoming')"
    )

class GraphTraversalResult(BaseModel):
    """Represents a full traversal path from a starting symbol to a connected symbol."""
    path: List[TraversalStep] = Field(
        ...,
        description="Ordered list of steps from the start node to the end node"
    )
    depth: int = Field(
        ...,
        ge=1,
        le=5,
        description="Number of hops from the start node to the end node"
    )
    end_node: "Symbol" = Field(
        ...,
        description="The final node reached at the end of the traversal path"
    )

class ParsedRelationship(TypedDict):
    type: RelationshipType
    target: str


if __name__ == "__main__":
    # 1. Create a dummy Symbol instance
    dummy_symbol = Symbol(
        id="sym_12345",
        name="calculate_total",
        kind="function",  # Using a generic string as allowed by the model definition (str), though usually matches SymbolKind
        type="Function",
        declaration="def calculate_total(items: List[Item]) -> float:",
        documentation="Calculates the sum of prices for a list of items.",
        filePath="src/utils/math_helpers.py",
        moduleName="math_helpers",
        cleanGenerics=None,
        functionSignature="(items: List[Item]) -> float",
        parameterNames=["items"],
        returnType="float",
        embedding=[0.12, 0.45, -0.23, 0.98]
    )

    # 2. Create a SymbolSchema instance
    dummy_symbol_schema = SymbolSchema(
        properties=[
            "id", "name", "kind", "type", "declaration", 
            "documentation", "filePath", "moduleName", "embedding"
        ],
        kinds=[
            "class", "function", "variable", "interface"
        ]
    )

    # 3. Create a GraphSchema instance
    dummy_graph_schema = GraphSchema(
        symbol_node=dummy_symbol_schema,
        module_names=["math_helpers", "core", "api"],
        relationships=[
            "MEMBER_OF", "INHERITS_FROM", "REFERENCES"
        ]
    )

    print("--- Symbol JSON ---")
    # Using model_dump_json() for Pydantic V2, use .json() for V1
    try:
        print(dummy_symbol.model_dump_json(indent=2))
    except AttributeError:
        print(dummy_symbol.json(indent=2))

    print("\n--- SymbolSchema JSON ---")
    try:
        print(dummy_symbol_schema.model_dump_json(indent=2))
    except AttributeError:
        print(dummy_symbol_schema.json(indent=2))

    print("\n--- GraphSchema JSON ---")
    try:
        print(dummy_graph_schema.model_dump_json(indent=2))
    except AttributeError:
        print(dummy_graph_schema.json(indent=2))

    print("\n--- Symbol Class Definition (JSON Schema) ---")
    try:
        # Pydantic V2
        print(json.dumps(Symbol.model_json_schema(), indent=2))
    except AttributeError:
        # Pydantic V1
        print(Symbol.schema_json(indent=2))

    print("\n--- GraphSchema Class Definition (JSON Schema) ---")
    try:
        # Pydantic V2
        print(json.dumps(GraphSchema.model_json_schema(), indent=2))
    except AttributeError:
        # Pydantic V1
        print(GraphSchema.schema_json(indent=2))
