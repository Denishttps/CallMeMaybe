import json
import logging
from pathlib import Path

from pydantic import TypeAdapter

from models.function import Function
from models.function_call import FunctionCall
from models.promt import Prompt


logger = logging.getLogger(__name__)


def get_functions(path: str | Path) -> list[Function]:
    """Load function definitions from a JSON file."""
    adapter = TypeAdapter(list[Function])
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return adapter.validate_python(data)


def get_promts(path: str | Path) -> list[Prompt]:
    """Load prompts from a JSON file."""
    adapter = TypeAdapter(list[Prompt])
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return adapter.validate_python(data)


def save_functions(path: str | Path, functions: list[FunctionCall]) -> None:
    """Persist function calls to disk as pretty-printed JSON."""
    adapter = TypeAdapter(list[FunctionCall])
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(adapter.dump_python(functions), file, indent=4)
        file.write("\n")


def load_json_file(path: str | Path) -> list[dict]:
    """Load a JSON array from disk and return it as a list of dicts."""
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError:
        logger.error("Input file was not found: %s", path)
        return []
    except json.JSONDecodeError as error:
        logger.error("Invalid JSON in %s: %s", path, error)
        return []
    except OSError as error:
        logger.error("Unable to read %s: %s", path, error)
        return []

    if not isinstance(data, list):
        logger.error("Expected a JSON array in %s", path)
        return []
    return data
