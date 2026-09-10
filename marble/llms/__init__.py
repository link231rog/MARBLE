from .usage import ApiUsageMeter, UsageSnapshot

__all__ = [
    "ApiUsageMeter",
    "UsageSnapshot",
    "model_prompting",
]


def model_prompting(*args, **kwargs):
    """Lazily import the LiteLLM-backed prompting implementation."""

    from .model_prompting import model_prompting as _model_prompting

    return _model_prompting(*args, **kwargs)
