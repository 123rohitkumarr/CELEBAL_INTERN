from fastapi import APIRouter

from app.agents.agent import agent
from app.models.schema import AssistantRequest

router = APIRouter()


@router.post("/assistant")
def assistant(request: AssistantRequest):
    return agent.run(request.query)