"""
Severity Mapper Module
Maps stuttering severity levels to exercise difficulty levels.
"""


def severity_to_level(severity: str) -> str:
    """
    Convert a severity string from the stuttering detection model
    into the corresponding exercise difficulty level.

    Args:
        severity (str): The severity level detected by the model.
                        Expected values: "mild", "moderate", "severe"

    Returns:
        str: The corresponding exercise level.
             "beginner" for mild, "intermediate" for moderate,
             "advanced" for severe.

    Raises:
        ValueError: If the severity string is not recognized.
    """
    severity_map = {
        "mild": "beginner",
        "moderate": "intermediate",
        "severe": "advanced",
    }

    severity_lower = severity.strip().lower()

    if severity_lower not in severity_map:
        raise ValueError(
            f"Unknown severity '{severity}'. "
            f"Expected one of: {', '.join(severity_map.keys())}"
        )

    return severity_map[severity_lower]


if __name__ == "__main__":
    # Quick test
    test_cases = ["mild", "moderate", "severe"]
    for s in test_cases:
        print(f"Severity '{s}' -> Level '{severity_to_level(s)}'")
