"""Import optional OCR dependencies only for the requested document format."""

from importlib import import_module

__all__ = ["OCRDOCLoader", "OCRIMGLoader", "OCRPDFLoader", "OCRPPTLoader"]


def __getattr__(name):
    modules = {
        "OCRDOCLoader": ".edu_docloader",
        "OCRIMGLoader": ".edu_imgloader",
        "OCRPDFLoader": ".edu_pdfloader",
        "OCRPPTLoader": ".edu_pptloader",
    }
    if name not in modules:
        raise AttributeError(name)
    value = getattr(import_module(modules[name], __name__), name)
    globals()[name] = value
    return value
