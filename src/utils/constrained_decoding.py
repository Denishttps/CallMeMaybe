import re


def is_valid_number_token(token: str, current: str) -> bool:
    candidate = current + token
    candidate = candidate.strip()
    if not candidate:
        return True
    try:
        float(candidate)
        return True
    except ValueError:
        return (
            bool(re.match(
                r'^-?(\d+\.?\d*|\d*\.\d+)([eE][+-]?\d*)?$', candidate
            )) or
            candidate in ("-", ".", "-.")
        )


def is_valid_bool_token(token: str, current: str) -> bool:
    candidate = (current + token).lower()
    return "true".startswith(candidate) or "false".startswith(candidate)


def is_valid_string_token(token: str) -> bool:
    return '"' not in token or token == '"'


def is_number_complete(current: str) -> bool:
    try:
        float(current)
        return True
    except ValueError:
        return False


def is_bool_complete(current: str) -> bool:
    return current.lower() in ("true", "false")


def is_string_complete(current: str) -> bool:
    return current.endswith('"')
