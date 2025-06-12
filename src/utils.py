from __future__ import annotations

from typing import Union # Union is used in the type hint

def to_bool(value: Union[str, bool, None]) -> bool:
    """
    Converts a string or boolean value to a boolean, strictly.
    'true' (case-insensitive) -> True
    'false' (case-insensitive) -> False
    boolean -> itself
    None -> False (as a default for missing optional params)
    Other string values -> ValueError
    Other types -> TypeError
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        val_lower = value.lower()
        if val_lower == 'true':
            return True
        if val_lower == 'false':
            return False
        raise ValueError(f"Cannot convert string '{value}' to boolean. Expected 'true' or 'false'.")
    # If it's not bool, None, or str, it's an unexpected type for this conversion
    raise TypeError(f"Cannot convert value of type {type(value)} to boolean. Expected str, bool, or None.")
