from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os
from typing import Dict, Any
from pydantic import BaseModel
from src.history_parser import HistoryParser
from src.graph_builder import GraphBuilder
from src.miner import Miner
from src.visualizer import Visualizer
from src.simulator import Simulator
from src.utils import get_github_token

app = FastAPI(
    title="RepoInsightMiner API",
    description="API for analyzing GitHub repositories with AI insights.",
    version="0.1.0"
)

security = HTTPBearer()


class RepoRequest(BaseModel):
    repo_url: str
    max_commits: int = 1000
    include_prs: bool = True


class SimulationRequest(BaseModel):
    scenario: str
    params: Dict[str, Any]


def get_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    if token != os.getenv("API_TOKEN"):  # Set API_TOKEN env for auth
        raise HTTPException(status_code=401, detail="Invalid token")
    return token


@app.post("/analyze_repo", response_model=Dict[str, Any])
async def analyze_repo(request: RepoRequest, token: str = Depends(get_token)):
    """
    Analyze a GitHub repo: Parse history, build graph, mine insights.
    Requires GITHUB_TOKEN env var.
    """
    try:
        parser = HistoryParser(request.repo_url)
        history = parser.parse_history(request.max_commits, request.include_prs)

        builder = GraphBuilder(history)
        nx_g = builder.get_networkx_graph()
        pyg_d = builder.to_pytorch_geometric()

        miner = Miner(nx_g, pyg_d)
        insights = miner.mine_insights(history)

        return {"insights": insights, "metadata": history["metadata"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/simulate", response_model=Dict[str, Any])
async def simulate_scenario(request: SimulationRequest, repo_url: str, token: str = Depends(get_token)):
    """
    Run a what-if simulation on a repo.
    First analyzes the repo, then simulates.
    """
    try:
        # Re-analyze or load from cache (simplified: re-analyze)
        parser = HistoryParser(repo_url)
        history = parser.parse_history()

        builder = GraphBuilder(history)
        nx_g = builder.get_networkx_graph()
        pyg_d = builder.to_pytorch_geometric()

        miner = Miner(nx_g, pyg_d)

        simulator = Simulator(nx_g, pyg_d, miner)
        result = simulator.run_simulation(request.scenario, request.params)

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)