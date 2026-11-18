import json
import sys


def main():
    request = json.load(sys.stdin)

    if not isinstance(request, dict):
        raise ValueError("Invalid request")

    if request.get("name") != "reason":
        raise ValueError("Incorrect tool name")

    arguments = request.get("arguments")

    if not isinstance(arguments, dict):
        raise ValueError("arguments must be an object")

    cot = arguments.get("cot")

    if not isinstance(cot, str):
        raise ValueError("cot must be a string")

    if len(cot) > 8192:
        raise ValueError("Reasoning note is too long")

    json.dump(
        {
            "stdout": "",
            "stdout_truncated": False,
        },
        sys.stdout,
        separators=(",", ":"),
    )

    sys.stdout.write("\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"reason: {error}", file=sys.stderr)
        sys.exit(1)