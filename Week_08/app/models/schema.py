from typing import Any

from pydantic import BaseModel, Field


class AssistantRequest(BaseModel):
    query: str = Field(..., min_length=1)


class AssistantResponse(BaseModel):
    success: bool
    intent: str
    tool: str
    data: Any | None = None
    error: str | None = None



class ToolResponse(BaseModel):
    success: bool
    result: float | int | None = None
    error: str | None = None