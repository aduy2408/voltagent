"""Optional FastAPI routes for image detection."""
from .detector import YoloDetector
from .uploads import UploadValidationError, validate_upload


def create_vision_router(detector: YoloDetector | None = None):
    try:
        from fastapi import APIRouter, File, HTTPException, UploadFile
    except ImportError as exc:
        raise RuntimeError("FastAPI is required to create vision routes") from exc

    router = APIRouter(prefix="/vision", tags=["vision"])
    detector = detector or YoloDetector()

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

    return router
