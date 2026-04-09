def fmt_bytes(b: int, *, unit_ref: int | None = None) -> str:
    """Format a byte count as a human-readable string.

    If unit_ref is given, the unit is chosen based on unit_ref's magnitude
    so that multiple values can be displayed in the same unit.
    """
    ref = unit_ref if unit_ref is not None else b
    if ref >= 1024 * 1024:
        return f"{b / (1024 * 1024):.2f} MiB"
    if ref >= 1024:
        return f"{b / 1024:.2f} KiB"
    return f"{b} B"
