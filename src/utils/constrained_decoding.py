import re


def is_valid_number_token(token: str, current: str) -> bool:
    """Check if appending the token to the current string can still form a valid number representation.""" # noqa
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
    """Check if appending the token to the current string can still form a valid boolean representation.""" # noqa
    candidate = (current + token).lower()
    return "true".startswith(candidate) or "false".startswith(candidate)


def is_valid_string_token(token: str) -> bool:
    """Check if the token is valid for a string parameter, allowing any token except unescaped double quotes.""" # noqa
    return '"' not in token or token == '"'


def is_number_complete(current: str) -> bool:
    """Check if the current string represents a complete number.""" # noqa
    try:
        float(current)
        return True
    except ValueError:
        return False


def is_bool_complete(current: str) -> bool:
    """Check if the current string represents a complete boolean value.""" # noqa
    return current.lower() in ("true", "false")


def is_string_complete(current: str) -> bool:
    """Check if the current string represents a complete string value, which is complete if it ends with a double quote.""" # noqa
    return current.endswith('"')
