from .pipeline import pipeline, HierarchicalLLM
from .inference import TextGenerator, TextEditor
from .qa import QuestionAnswerer
from .indexing import DocumentIndexer

try:
    from .. import _rust as _native
except Exception:
    _native = None

poincare = getattr(_native, "poincare", None) if _native else None
lorentz = getattr(_native, "lorentz", None) if _native else None
klein = getattr(_native, "klein", None) if _native else None
metrikey = getattr(_native, "metrikey", None) if _native else None
spline = getattr(_native, "spline", None) if _native else None
geodesic = getattr(_native, "geodesic", None) if _native else None

__all__ = [
    "pipeline",
    "HierarchicalLLM",
    "TextGenerator",
    "TextEditor",
    "QuestionAnswerer",
    "DocumentIndexer",
    "poincare",
    "lorentz",
    "klein",
    "metrikey",
    "spline",
    "geodesic",
]
