"""Local aesthetic preview proof of concept.

Layering (dependencies point inward):

    app.py / src.cli        presentation
        -> src.pipeline     orchestration
        -> src.treatments   registry of edits
        -> src.qc, src.identity, src.narrative
        -> src.detection, src.masks, src.imaging, src.models, src.config
"""

__all__ = ["__version__"]
__version__ = "0.2.0"
