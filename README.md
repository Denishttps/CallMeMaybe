This project has been created as part of the 42 curriculum by dbobrov

## Description

CallMeMaybe is a small Python project that translates natural-language requests into structured function calls using a local LLM and constrained decoding. The goal is to produce valid JSON that matches a schema rather than rely on free-form text generation.

## Instructions

1. Create and activate a virtual environment with uv.
2. Install dependencies with `uv sync`.
3. Run the pipeline with `uv run python -m src`.
4. Optional custom paths can be supplied with `--functions_definition`, `--input`, and `--output`.

## Algorithm explanation

The implementation builds a prompt for the model, then constrains token selection at every step so the output remains valid for the target schema. For numbers, booleans and strings, only tokens that can still produce a valid value are kept available. This makes the final JSON reliable even for a small model.

## Design decisions

- The project uses Pydantic models to validate the input and output schemas.
- The generation logic is separated into a dedicated generator component.
- File I/O is handled through a small loader module with explicit error handling.

## Performance analysis

The current approach is lightweight and suitable for small function sets. Accuracy depends on the model and the quality of the constrained decoding masks; the implementation prioritises reliability over unconstrained generation.

## Challenges faced

The main challenge was ensuring that the generated output stays both syntactically correct JSON and compatible with the function schema. This was addressed by filtering invalid tokens at generation time and by validating the final payload.

## Testing strategy

The repository includes a small regression test suite covering JSON loading and persistence. Additional prompt and function definitions can be added under the input data folder and processed by the main script.

## Example usage

```bash
uv sync
uv run python -m src
uv run python -m src --input data/input/function_calling_tests.json --output data/output/function_calling_results.json
```

## Resources

- The official documentation for Pydantic.
- The Hugging Face Transformers documentation.
- The project task description in the repository's task.md file.

AI was used to help structure the project, draft the documentation, and identify places where constrained decoding could be applied.
