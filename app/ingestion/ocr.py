"""OCR adapters used only for PDF pages without a usable text layer."""

from io import BytesIO
from typing import Any, Protocol

from PIL import Image


class OCREngine(Protocol):
    def recognize(self, image_png: bytes) -> str:
        """Recognize text in one PNG image."""
        ...


class OCRProcessingError(RuntimeError):
    """Raised when OCR fails for a specific document page."""


class PaddleOCREngine:
    """CPU PaddleOCR engine whose model and dependencies load on first use."""

    def __init__(self) -> None:
        self._engine: Any | None = None

    def _get_engine(self) -> Any:
        if self._engine is None:
            from paddleocr import PaddleOCR

            self._engine = PaddleOCR(
                lang="ch",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                device="cpu",
                enable_mkldnn=False,
            )
        return self._engine

    def recognize(self, image_png: bytes) -> str:
        import numpy as np

        with Image.open(BytesIO(image_png)) as image:
            pixels = np.asarray(image.convert("RGB"))
        results = self._get_engine().predict(pixels)
        lines: list[str] = []
        for result in results:
            texts = [str(text).strip() for text in result["rec_texts"]]
            polygons = result["rec_polys"]
            ordered = sorted(
                zip(polygons, texts, strict=True),
                key=lambda item: (
                    min(float(point[1]) for point in item[0]),
                    min(float(point[0]) for point in item[0]),
                ),
            )
            lines.extend(text for _, text in ordered if text)
        return "\n".join(lines)
