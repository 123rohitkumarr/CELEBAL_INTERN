from app.models.schemas import AssistantResponse


def success(intent, tool, data):
    return AssistantResponse(
        success=True,
        intent=intent,
        tool=tool,
        data=data,
    )


def failure(intent, tool, error):
    return AssistantResponse(
        success=False,
        intent=intent,
        tool=tool,
        error=error,
    )