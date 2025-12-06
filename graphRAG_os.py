from agno.os.app import AgentOS
from agent_graphRAG import create_graphrag_agent
from dotenv import load_dotenv


load_dotenv()


agent_os = AgentOS(
    id="graph_rag_os",
    description="Agent OS interface for writing documentation leveraging graphRAG",
    agents=[create_graphrag_agent(debug_level=2, debug_mode=True, model="qwen-3-235b-a22b-instruct-2507")],
    # run_hooks_in_background=True,
)


app = agent_os.get_app()

if __name__ == "__main__":
    # Default port is 7777; change with port=...
    agent_os.serve(app="graphRAG_os:app", reload=True)