"""Isaac Lab-Arena execution for passive resistance mechanisms."""


def embodiment(**kwargs):
    """Create the registered Inspect Robots embodiment without importing Kit early."""
    from .runtime import ResistanceEmbodiment

    return ResistanceEmbodiment(**kwargs)
