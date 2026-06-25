import argparse
import logging
from pathlib import Path

from llm_sdk import Small_LLM_Model
from generator import Generator
from loader import get_promts, save_functions


DEFAULT_FUNCTIONS = "data/input/functions_definition.json"
DEFAULT_INPUT = "data/input/function_calling_tests.json"
DEFAULT_OUTPUT = "data/output/function_calling_results.json"

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CallMeMaybe - function calling with constrained decoding"
    )
    parser.add_argument("--functions_definition", default=DEFAULT_FUNCTIONS)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    model = Small_LLM_Model()
    generator = Generator(
        model=model,
        path_functions=args.functions_definition,
        output_file=args.output,
    )

    prompts = get_promts(args.input)
    if not prompts:
        logging.error("No prompts to process")
        return

    results = []
    for prompt in prompts:
        result = generator.generate(prompt.prompt)
        if result is not None:
            results.append(result)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_functions(args.output, results)
    print(f"Saved {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()
