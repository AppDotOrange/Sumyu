import ast
import json
import math
import operator
import sys


BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def checked(value):
    if type(value) not in (int, float):
        raise ValueError("Only real numeric results are supported")

    if type(value) is int:
        if value.bit_length() > 512:
            raise ValueError("Integer result is too large")
    else:
        if not math.isfinite(value) or abs(value) > 1e100:
            raise ValueError("Numeric result is outside the supported range")

    return value


def evaluate(node, depth=0):
    if depth > 32:
        raise ValueError("Expression is too deeply nested")

    if isinstance(node, ast.Expression):
        return evaluate(node.body, depth + 1)

    if isinstance(node, ast.Constant):
        if type(node.value) not in (int, float):
            raise ValueError("Only numeric constants are supported")

        return checked(node.value)

    if isinstance(node, ast.UnaryOp):
        operation = UNARY_OPERATORS.get(type(node.op))

        if operation is None:
            raise ValueError("Unsupported unary operator")

        return checked(
            operation(evaluate(node.operand, depth + 1))
        )

    if isinstance(node, ast.BinOp):
        operation = BINARY_OPERATORS.get(type(node.op))

        if operation is None:
            raise ValueError("Unsupported arithmetic operator")

        left = evaluate(node.left, depth + 1)
        right = evaluate(node.right, depth + 1)

        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("Exponent is too large")

        return checked(operation(left, right))

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Invalid function")

        if node.keywords:
            raise ValueError("Keyword arguments are not supported")

        name = node.func.id
        args = [
            evaluate(arg, depth + 1)
            for arg in node.args
        ]

        if name == "round":
            if len(args) not in (1, 2):
                raise ValueError("round expects one or two arguments")

            digits = 0 if len(args) == 1 else args[1]

            if type(digits) is not int or abs(digits) > 12:
                raise ValueError("Invalid rounding precision")

            return checked(round(args[0], digits))

        if name == "abs":
            if len(args) != 1:
                raise ValueError("abs expects one argument")

            return checked(abs(args[0]))

        if name in ("min", "max"):
            if not args:
                raise ValueError(f"{name} expects at least one argument")

            operation = min if name == "min" else max
            return checked(operation(args))

        raise ValueError(f"Unsupported function: {name}")

    raise ValueError(
        f"Unsupported expression element: {type(node).__name__}"
    )


def calculate(expression):
    if not isinstance(expression, str):
        raise ValueError("expression must be a string")

    if not expression or len(expression) > 2048:
        raise ValueError("Invalid expression length")

    tree = ast.parse(expression, mode="eval")

    if sum(1 for _ in ast.walk(tree)) > 128:
        raise ValueError("Expression is too complex")

    return evaluate(tree)


def main():
    request = json.load(sys.stdin)

    if not isinstance(request, dict):
        raise ValueError("Invalid request")

    if request.get("name") != "calculate":
        raise ValueError("Incorrect tool name")

    arguments = request.get("arguments")

    if not isinstance(arguments, dict):
        raise ValueError("arguments must be an object")

    result = calculate(arguments.get("expression"))

    # stdout must contain minified JSON only.
    json.dump(
        result,
        sys.stdout,
        allow_nan=False,
        separators=(",", ":"),
    )
    sys.stdout.write("\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"calculate: {error}", file=sys.stderr)
        sys.exit(1)