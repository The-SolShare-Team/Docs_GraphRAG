from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Literal, TypedDict

class Symbol(BaseModel):
    id: str
    name: str
    kind: str
    type: Optional[str] = None
    filePath: Optional[str] = None
    moduleName: str
    cleanDocString: str
    cleanDeclaration: str
    cleanGenerics: Optional[str] = None
    functionSignature: Optional[str] = None
    parameterNames: Optional[List[str]] = None
    returnType: Optional[str] = None

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

class ParsedRelationship(TypedDict):
    type: RelationshipType
    target: str