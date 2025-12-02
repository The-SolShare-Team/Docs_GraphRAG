from agno.os.app import AgentOS
from agent_graphRAG import create_graphrag_agent
from dotenv import load_dotenv


load_dotenv()


agent_os = AgentOS(
    id="graph_rag_os",
    description="Agent OS interface for writing documentation leveraging graphRAG",
    agents=[create_graphrag_agent()]

)


app = agent_os.get_app()

if __name__ == "__main__":
    # Default port is 7777; change with port=...
    agent_os.serve(app="agno_agent_os:app", reload=True)