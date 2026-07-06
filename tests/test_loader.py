import json
from pathlib import Path

from src.loader import get_functions, get_promts, save_functions
from src.models.function_call import FunctionCall


def test_get_functions_reads_valid_definition(tmp_path: Path) -> None:
    """Test that get_functions correctly reads a valid function definition from a JSON file.""" # noqa
    definition_path = tmp_path / "functions.json"
    definition_path.write_text(
        json.dumps(
            [
                {
                    "name": "fn_test",
                    "description": "Test function",
                    "parameters": {"value": {"type": "string"}},
                    "returns": {"type": "string"},
                }
            ]
        ),
        encoding="utf-8",
    )

    functions = get_functions(str(definition_path))

    assert len(functions) == 1
    assert functions[0].name == "fn_test"


def test_get_promts_reads_valid_prompts(tmp_path: Path) -> None:
    """Test that get_promts correctly reads valid prompts from a JSON file.""" # noqa
    prompts_path = tmp_path / "prompts.json"
    prompts_path.write_text(
        json.dumps([{"prompt": "hello"}]), encoding="utf-8"
    )

    prompts = get_promts(str(prompts_path))

    assert len(prompts) == 1
    assert prompts[0].prompt == "hello"


def test_save_functions_writes_expected_json(tmp_path: Path) -> None:
    """Test that save_functions correctly writes a list of FunctionCall objects to a JSON file.""" # noqa
    output_path = tmp_path / "results.json"
    payload = [
        FunctionCall(prompt="hi", name="fn_test", parameters={"value": "x"})
    ]

    save_functions(str(output_path), payload)

    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written[0]["name"] == "fn_test"
    assert written[0]["parameters"]["value"] == "x"
