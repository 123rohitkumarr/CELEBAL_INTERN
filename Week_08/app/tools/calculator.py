import ast
import operator

# Supported operations
OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


class SafeEvaluator:
    def evaluate(self, expression: str):
        node = ast.parse(expression, mode="eval").body
        return self._eval(node)

    def _eval(self, node):
        if isinstance(node, ast.Constant):  # Python 3.8+
            return node.value

        if isinstance(node, ast.Num):       # Backward compatibility
            return node.n

        if isinstance(node, ast.BinOp):
            left = self._eval(node.left)
            right = self._eval(node.right)
            return OPERATORS[type(node.op)](left, right)

        if isinstance(node, ast.UnaryOp):
            operand = self._eval(node.operand)
            return OPERATORS[type(node.op)](operand)

        raise ValueError("Unsupported expression")


evaluator = SafeEvaluator()


def calculate(expression: str):
    try:
        result = evaluator.evaluate(expression)

        return {
            "success": True,
            "result": result
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }