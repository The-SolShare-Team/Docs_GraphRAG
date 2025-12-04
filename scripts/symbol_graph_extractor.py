import subprocess
from pathlib import Path
import json
from scripts.build_symbol_graphs import build_symbol_graphs
import os

def run_symbol_graph_extractor():

    build_symbol_graphs()

    project_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../SwiftSymbolKit")  # <-- change this

    result = subprocess.run(
        ["swift", "run", "SymbolGraphExtractor"],
        cwd=project_dir,            # run inside the Swift package dir
        text=True,
        capture_output=True         # capture stdout and stderr
    )
    
    if result.returncode != 0:
        print("Swift tool failed:")
        print(result.stderr)
        return None

    try:
        parsed_json = json.loads(result.stdout)
        return parsed_json
    except json.JSONDecodeError as e:
        print("Failed to parse JSON:", e)
        print("Raw output was:\n", result.stdout[:500])
        return None
