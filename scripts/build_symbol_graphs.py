import subprocess
import shutil
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path="../.env")
PROJECT_PATH = os.environ.get("PROJECT_PATH", "/Users/williamjin/Documents/solanaProj/SolanaWalletAdapterKit")


def build_symbol_graphs(project_path: str = PROJECT_PATH, output_dir: str = None):
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
    clean_cmd = ["swift", "package", "clean"]
    build_cmd = [
        "swift", "build",
        "-Xswiftc", "-emit-symbol-graph",
        "-Xswiftc", "-emit-symbol-graph-dir",
        "-Xswiftc", output_dir
    ]

    try:
        for cmd in [clean_cmd, build_cmd]:
            subprocess.run(cmd, cwd=project_path, check=True, capture_output=True, text=True)
        print(f"Build success! JSON files are in {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Build failed:\n{e.stderr}")
        raise e

