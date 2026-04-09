"""Input adapters for loading models into the TiGrIS IR."""

from pathlib import Path

from tigris.graph.ir import AnalyzedGraph

__all__ = ["load_model"]

_EXTENSION_MAP = {
    ".onnx": "tigris.loaders.onnx",
}


def load_model(path: str | Path) -> AnalyzedGraph:
    """Load a model file and return an AnalyzedGraph.

    Dispatches to the appropriate loader based on file extension.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    module_name = _EXTENSION_MAP.get(suffix)
    if module_name is None:
        supported = ", ".join(sorted(_EXTENSION_MAP.keys()))
        raise ValueError(
            f"Unsupported model format '{suffix}'. Supported: {supported}"
        )

    import importlib
    module = importlib.import_module(module_name)
    return module.load_model(path)
