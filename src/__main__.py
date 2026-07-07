import argparse
import logging

from generator import Generator
from llm_sdk import Small_LLM_Model  # type: ignore[import-untyped]

from loader import get_promts, save_functions
from config import DEFAULT_FUNCTIONS, DEFAULT_INPUT, DEFAULT_OUTPUT


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def main() -> None:
    """Run the function-calling pipeline from the command line."""
    parser = argparse.ArgumentParser(
        description="CallMeMaybe - function calling with constrained decoding"
    )
    parser.add_argument("--functions_definition", default=DEFAULT_FUNCTIONS)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    try:
        model = Small_LLM_Model()
    except Exception as error:
        logging.error("Unable to initialize the language model: %s", error)
        return

    generator = Generator(
        model=model,
        path_functions=args.functions_definition,
        output_file=args.output,
    )

    try:
        prompts = get_promts(args.input)
    except Exception as error:
        logging.error("Unable to load prompts: %s", error)
        return

    if not prompts:
        logging.error("No prompts to process")
        return

    results = []
    for prompt in prompts:
        try:
            result = generator.generate(prompt.prompt)
        except Exception as error:
            logging.error(
                "Failed to process prompt '%s': %s",
                prompt.prompt,
                error
            )
            continue
        if result is not None:
            results.append(result)

    save_functions(args.output, results)
    print(f"Saved {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()
