"""Optional FastAPI routes for image detection."""
import logging
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile

from .detector import YoloDetector
from .uploads import UploadValidationError, validate_upload

logger = logging.getLogger(__name__)


def create_vision_router(
    detector: YoloDetector | None = None,
    upload_dir: str | Path = "data/uploads",
    max_upload_bytes: int = 10 * 1024 * 1024,
):
    router = APIRouter(prefix="/vision", tags=["vision"])
    detector = detector or YoloDetector()
    destination = Path(upload_dir)
    destination.mkdir(parents=True, exist_ok=True)

    @router.get("/health")
    def health():
        return {"detector_available": detector.available, "error": detector.error}

    @router.post("/detect")
    async def detect(file: UploadFile = File(...)):  # noqa: B008
        content = await file.read()
        try:
            validate_upload(content, file.content_type)
        except UploadValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"detections": [d.__dict__ for d in detector.detect(content)], "available": detector.available}

    @router.post("/upload")
    async def upload(file: UploadFile = File(...)):  # noqa: B008
        content = await file.read()
        try:
            validate_upload(content, file.content_type)
        except UploadValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if len(content) > max_upload_bytes:
            raise HTTPException(status_code=422, detail="upload exceeds configured limit")
        image_id = f"{uuid4().hex}{Path(file.filename or 'image.png').suffix.lower() or '.png'}"
        (destination / image_id).write_bytes(content)
        logger.info(
            "image_uploaded image_id=%s filename=%s content_type=%s size_bytes=%d",
            image_id,
            file.filename or "image",
            file.content_type or "unknown",
            len(content),
        )
        return {"image_id": image_id, "content_type": file.content_type, "size": len(content)}

    return router
