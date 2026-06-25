import json
from pydantic import TypeAdapter

from models.function_call import FunctionCall
from models.promt import Prompt

from models.function import Function


def get_functions(path: str) -> list[Function]:
    adapter = TypeAdapter(list[Function])
    with open(path, "r") as file:
        data = json.load(file)
    return adapter.validate_python(data)


def get_promts(path: str) -> list[str]:
    adapter = TypeAdapter(list[Prompt])
    with open(path, "r") as file:
        data = json.load(file)
    return adapter.validate_python(data)


def save_functions(path: str, functions: list[FunctionCall]) -> None:
    adapter = TypeAdapter(list[FunctionCall])
    with open(path, "w") as file:
        json.dump(adapter.dump_python(functions), file, indent=4)
