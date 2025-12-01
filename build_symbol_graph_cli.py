from symbol_graph_embedding import build_symbol_graphs
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Swift Symbol Graphs")
    parser.add_argument(
        "--project-path",
        default="/Users/williamjin/Documents/solanaProj/SolanaWalletAdapterKit",
        help="Path to the Swift package root"
    )
    parser.add_argument(
        "--output",
        default=os.path.join(os.getcwd(), "symbol-graphs"),
        help="Output directory for JSON files"
    )
    
    args = parser.parse_args()
    build_symbol_graphs(args.project_path, args.output)