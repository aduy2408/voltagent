from dataclasses import dataclass
from io import BytesIO

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}


@dataclass(frozen=True)
class UploadPolicy:
    max_bytes: int = 10 * 1024 * 1024
    allowed_types: frozenset[str] = frozenset(ALLOWED_IMAGE_TYPES)


class UploadValidationError(ValueError):
    pass


def validate_upload(content: bytes, content_type: str | None, policy: UploadPolicy | None = None) -> None:
    policy = policy or UploadPolicy()
    if not content:
        raise UploadValidationError("upload is empty")
    if len(content) > policy.max_bytes:
        raise UploadValidationError(f"upload exceeds {policy.max_bytes} bytes")
    if content_type not in policy.allowed_types:
        raise UploadValidationError("unsupported image type")
    try:
        from PIL import Image
        with Image.open(BytesIO(content)) as image:
            image.verify()
    except Exception as exc:
        raise UploadValidationError("invalid image data") from exc
