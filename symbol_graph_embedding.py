from typing import List, Dict, Any
import subprocess
import shutil
import glob
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()
def build_symbol_graphs(project_path: str = "/Users/williamjin/Documents/solanaProj/SolanaWalletAdapterKit", output_dir: str = None):
    """
    Runs the swift build command and outputs symbol graphs to a specific directory.
    """
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), "symbol-graphs")
    print("Output Dir (default): ", output_dir)
    print(f"Building Symbol Graphs for {project_path}...")

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    cmd = [
        "swift", "build",
        "-Xswiftc", "-emit-symbol-graph",
        "-Xswiftc", "-emit-symbol-graph-dir",
        "-Xswiftc", output_dir
    ]

    try:
        subprocess.run(cmd, cwd=project_path, check=True, capture_output=True, text=True)
        print(f"Build success! JSON files are in {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Build failed:\n{e.stderr}")
        raise e


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
                title="Swift Symbol"            # Optional metadata for the model
            )
        )
        return response.embeddings[0].values
    except Exception as e:
        print(f"Embedding failed: {e}")
        # Return zero vector so the graph write doesn't fail, 
        # but log it so you can retry later.
        return [0.0] * EMBED_DIM
    
def create_embedding_text(symbol: Dict[str, Any]) -> str:
    """Create a rich text representation for embedding"""
    parts = []
    
    # Add the declaration
    if symbol.get('cleanDeclaration'):
        parts.append(f"Declaration: {symbol['cleanDeclaration']}")
    
    # Add documentation
    if symbol.get('cleanDocString'):
        parts.append(f"Documentation: {symbol['cleanDocString']}")
    
    # Add kind and name
    parts.append(f"Kind: {symbol.get('kind', '')}")
    parts.append(f"Name: {symbol.get('name', '')}")
    
    # Add module info
    if symbol.get('moduleName'):
        parts.append(f"Module: {symbol['moduleName']}")
    
    # Add file path
    if symbol.get('filePath'):
        parts.append(f"Path: {symbol['filePath']}")
    
    return " | ".join(parts)

