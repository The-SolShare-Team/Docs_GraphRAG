from typing import List, Dict, Any
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


load_dotenv(override=True)
@retry(
    stop=stop_after_attempt(10),  # Give up after 5 failed attempts for a single item
    wait=wait_exponential(multiplier=1, min=1, max=10), # Wait 1s, then 2s, 4s, etc.
    retry=retry_if_exception_type(Exception) # Or a more specific exception from your API client
)
def generate_embeddings(text: str, api_key, task_type: str ="RETRIEVAL_DOCUMENT") -> List[float]:
    """
    Calls Google Gemini to generate a vector for the Golden Chunk.
    Uses 'RETRIEVAL_DOCUMENT' optimized for database storage.
    """
    client = genai.Client(api_key=os.environ.get(api_key))
    EMBED_DIM = 3072

    key_value = os.environ.get(api_key)
    if key_value:
        masked_key = f"{key_value[:10]}...{key_value[-10:]}"
        print(f"  🔑 Using key {api_key}: {masked_key}")
    else:
        print(f"  ⚠️  Key {api_key} not found in environment!")
    
    if not text or len(text.strip()) < 5:
        # Return empty vector if text is too short/empty
        return [0.0] * EMBED_DIM
    
    try:
        response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=[text],
            config=types.EmbedContentConfig(
                task_type=task_type, # Critical for storage / query
            )
        )
        return response.embeddings[0].values
    except Exception as e:
        print(f"Embedding failed: {e}, retrying")
        # Return zero vector so the graph write doesn't fail, 
        # but log it so you can retry later.
        raise
    
def create_embedding_text(symbol: Dict[str, Any]) -> str:
    parts = []

    if symbol.get("name"):
        parts.append(f"Name: {symbol['name']}")

    if symbol.get("kind"):
        parts.append(f"Kind: {symbol['kind']}")

    if symbol.get("moduleName"):
        parts.append(f"Module: {symbol['moduleName']}")

    if symbol.get("cleanDeclaration"):
        parts.append(f"Declaration: {symbol['cleanDeclaration']}")

    if symbol.get("functionSignature"):
        parts.append(f"Signature: {symbol['functionSignature']}")

    if symbol.get("returnType"):
        parts.append(f"Return Type: {symbol['returnType']}")

    if symbol.get("cleanGenerics"):
        parts.append(f"Generics: {symbol['cleanGenerics']}")

    if symbol.get("cleanDocString"):
        parts.append(f"Documentation: {symbol['cleanDocString']}")

    return " | ".join(parts)


