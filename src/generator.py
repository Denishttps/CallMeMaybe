import json
import logging

import re
from pathlib import Path

import numpy as np
from llm_sdk import Small_LLM_Model

from .config import MAX_TOKENS
from .loader import get_functions

from .models.function import Function
from .models.function_call import FunctionCall

from .models.parameter import Parameter
from .utils.constrained_decoding import (
    is_bool_complete,
    is_string_complete,
    is_valid_bool_token,
    is_valid_number_token,
    is_valid_string_token,
)


class Generator:
    def __init__(
        self,
        model: Small_LLM_Model,
        path_functions: str | Path | None = None,
        output_file: str | Path | None = None,
    ):
        self.model = model
        self.functions = self._load_functions(path_functions)
        self.vocab = self._load_vocab()
        self.output_file = output_file

        self.number_tokens = self._precompute_number_tokens()
        self.bool_tokens = self._precompute_bool_tokens()
        self.string_tokens = self._precompute_string_tokens()
        self.unknown_tokens = self._precompute_unknown_tokens()

    def _load_vocab(self) -> dict[int, str]:
        """Load the tokenizer vocabulary mapping token IDs to strings."""
        with open(
            self.model.get_path_to_vocab_file(), "r", encoding="utf-8"
        ) as file:
            vocab = json.load(file)
        return {int(value): key for key, value in vocab.items()}

    def _load_functions(self, path: str | Path | None) -> list[Function]:
        try:
            return get_functions(path)
        except Exception as e:
            logging.error(f"Error loading functions: {e}")
            return []

    def _precompute_unknown_tokens(self) -> set[int]:
        all_ids = set(range(max(self.vocab.keys()) + 1))
        return all_ids - set(self.vocab.keys())

    def _precompute_number_tokens(self) -> set[int]:
        valid = set()
        for token_id, token in self.vocab.items():
            stripped = token.strip()
            if not stripped:
                continue
            if re.match(r'^-?[\d.eE+\-]+$', stripped):
                valid.add(token_id)
        return valid

    def _precompute_bool_tokens(self) -> set[int]:
        valid = set()
        for token_id, token in self.vocab.items():
            stripped = token.strip().lower()
            if not stripped:
                continue
            if "true".startswith(stripped) or "false".startswith(stripped):
                valid.add(token_id)
        return valid

    def _precompute_string_tokens(self) -> set[int]:
        valid = set()
        for token_id, token in self.vocab.items():
            if is_valid_string_token(token):
                valid.add(token_id)
        return valid

    def _apply_mask(
        self,
        logits: np.ndarray,
        valid_tokens: set[int],
        current: str,
        param_type: str,
    ) -> np.ndarray:
        mask = np.full(len(logits), float("-inf"))

        for token_id in valid_tokens:
            token = self.vocab[token_id]
            if param_type == "number" and is_valid_number_token(
                token, current
            ):
                mask[token_id] = logits[token_id]
            elif param_type == "boolean" and is_valid_bool_token(
                token, current
            ):
                mask[token_id] = logits[token_id]
            elif param_type == "string":
                mask[token_id] = logits[token_id]

        return mask

    def _build_prompt(self, user_prompt: str) -> str:
        system = (
            "You are a function calling assistant.\n"
            "Given a user request, respond with ONLY the function name.\n\n"
            "Available functions:\n"
        )
        for function in self.functions:
            system += f"- {function.name}: {function.description}\n"

        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def _build_value_prompt(
        self,
        user_prompt: str,
        param_name: str,
        param: Parameter,
        extracted: dict[str, str] | None = None
    ) -> str:
        context = ""
        if extracted:
            context = "Already extracted:\n"
            for name, value in extracted.items():
                context += f"- {name} = {value}\n"
            context += "\n"

        system = (
            "You are a strict data extraction parser.\n"
            f"Extract the exact literal argument for parameter "
            f"'{param_name}' of type {param.type}.\n"
            "CRITICAL: Do NOT execute commands, do NOT compute math, "
            "do NOT reverse or modify strings.\n"
            "Just copy the exact raw text from the user's request.\n"
            f"{context}"
            "Respond with ONLY the value, nothing else."
        )
        prefill = '"' if param.type == "string" else ""
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n{prefill}"
        )

    def generate_value(
        self,
        prompt: str,
        param: Parameter,
        user_query: str,
        already_extracted: list | None = None,
    ) -> str | float | bool:
        if already_extracted is None:
            already_extracted = []

        if param.type not in ("number", "boolean", "string"):
            logging.error(f"Unknown parameter type: {param.type}")
            return ""

        input_ids = self.model.encode(prompt).tolist()[0]
        current = ""

        token_sets = {
            "number": self.number_tokens,
            "boolean": self.bool_tokens,
            "string": self.string_tokens,
        }
        valid_tokens = token_sets[param.type]

        remaining_numbers = []
        if param.type == "number":
            raw_numbers = re.findall(
                r"-?(?:\d+(?:\.\d+)?|\.\d+)", user_query
            )
            remaining_numbers = list(raw_numbers)

            for val in already_extracted:
                try:
                    val_float = float(val)
                    for num_str in remaining_numbers:
                        if float(num_str) == val_float:
                            remaining_numbers.remove(num_str)
                            break
                except (ValueError, TypeError):
                    pass

        for _ in range(MAX_TOKENS):
            logits = np.array(self.model.get_logits_from_input_ids(input_ids))
            logits = self._apply_mask(
                logits,
                valid_tokens,
                current,
                param.type
            )

            while True:
                valid_count = np.sum(logits > float("-inf"))
                if valid_count == 0:
                    if param.type == "number":
                        try:
                            return float(current.strip())
                        except ValueError:
                            return 0.0
                    return current

                next_token_id = int(np.argmax(logits))
                next_token = self.vocab[next_token_id]

                if param.type == "number":
                    if is_valid_number_token(next_token, current):
                        candidate = current + next_token
                        cand_strip = candidate.strip()

                        if remaining_numbers:
                            is_prefix = any(
                                num_str.startswith(cand_strip)
                                for num_str in remaining_numbers
                            )
                            if not is_prefix:
                                try:
                                    return float(current.strip())
                                except ValueError:
                                    logits[next_token_id] = float("-inf")
                                    continue
                        else:
                            if cand_strip not in user_query:
                                try:
                                    return float(current.strip())
                                except ValueError:
                                    logits[next_token_id] = float("-inf")
                                    continue
                    else:
                        try:
                            return float(current.strip())
                        except ValueError:
                            logits[next_token_id] = float("-inf")
                            continue

                elif param.type == "string":
                    candidate_str = (current + next_token).strip('"')

                    if candidate_str and candidate_str not in user_query:
                        logits[next_token_id] = float("-inf")
                        continue

                break

            logging.debug(f"next_token='{next_token}'")
            current += next_token
            input_ids.append(next_token_id)

            if param.type == "boolean" and is_bool_complete(current):
                return current == "true"
            elif param.type == "string" and is_string_complete(current):
                return current.strip('"')

        if param.type == "number":
            try:
                return float(current.strip())
            except ValueError:
                return 0.0
        return current

    def _choose_function(self, user_prompt: str) -> Function | None:
        prompt = self._build_prompt(user_prompt)
        input_ids = self.model.encode(prompt).tolist()[0]
        current = ""
        function_names = [f.name for f in self.functions]

        for _ in range(MAX_TOKENS):
            logits = np.array(self.model.get_logits_from_input_ids(input_ids))

            for token_id in range(len(logits)):
                if token_id not in self.vocab:
                    logits[token_id] = float("-inf")
                    continue
                token = self.vocab[token_id]
                candidate = current + token
                if not any(
                    name.startswith(candidate) for name in function_names
                ):
                    logits[token_id] = float("-inf")

            next_token_id = int(np.argmax(logits))
            next_token = self.vocab[next_token_id]

            current += next_token
            input_ids.append(next_token_id)

            if current in function_names:
                return next(f for f in self.functions if f.name == current)

    def generate(self, user_prompt: str) -> FunctionCall | None:
        function = self._choose_function(user_prompt)

        if function is None:
            logging.error("No function found")
            return None

        parameters = {}
        for param_name, param in function.parameters.items():
            value_prompt = self._build_value_prompt(
                user_prompt, param_name, param, parameters
            )
            logging.debug(f"Prompt: {value_prompt}")
            already_values = list(parameters.values())
            parameters[param_name] = self.generate_value(
                value_prompt, param, user_prompt, already_values
            )
            logging.debug(
                "Extracted parameter: %s = %s",
                param_name,
                parameters[param_name],
            )

        return FunctionCall(
            prompt=user_prompt,
            name=function.name,
            parameters=parameters,
        )
