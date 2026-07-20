import time

from app.services.llm_service import ask_llm
from app.services.planner import create_plan
from app.tools.calculator import calculate
from app.tools.keyword_tool import extract_keywords

TOOLS = {
    "calculator": calculate,
    "keyword_tool": extract_keywords,
}


class SmartAgent:
    def run(self, query: str):
        plan = create_plan(query)
        tool_name = plan.get("tool", "GENERAL")
        arguments = plan.get("arguments", {})

        start = time.perf_counter()

        if tool_name == "GENERAL":
            result = ask_llm(query)
            execution_time = round((time.perf_counter() - start) * 1000, 2)
            return {
                "success": result.get("success", True),
                "intent": "GENERAL",
                "tool": "Groq LLM",
                "execution_time_ms": execution_time,
                **result,
            }

        if tool_name not in TOOLS:
            return {"success": False, "error": f"Unknown tool '{tool_name}'"}

        tool = TOOLS[tool_name]
        result = tool(**arguments)
        execution_time = round((time.perf_counter() - start) * 1000, 2)
        return {
            "success": result.get("success", True),
            "intent": tool_name.upper(),
            "tool": tool_name,
            "execution_time_ms": execution_time,
            **result,
        }


agent = SmartAgent()