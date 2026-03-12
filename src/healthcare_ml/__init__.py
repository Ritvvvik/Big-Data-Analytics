"""Healthcare readmission prediction package."""


def run_training_pipeline(*args, **kwargs):
    """Lazy import wrapper to avoid hard import failures before dependencies are installed."""
    from .pipeline import run_training_pipeline as _run

    return _run(*args, **kwargs)


__all__ = ["run_training_pipeline"]
