"""Image references for the multimodal readout.

An image cannot travel through the text readout: the client renders that prompt
itself with the checkpoint's chat template, and only the server can expand an
image placeholder into the image tokens the checkpoint expects. The image
readout therefore posts to ``/v1/chat/completions`` and lets the server render
the prompt. This module builds the one value that request needs, the
``image_url`` content part: an ``http(s)`` URL, a ``data:`` URL, or a base64
data URL made from a local PNG, JPEG, or WebP file.

Nothing here decodes, resizes, or re-encodes an image, and no image library is
imported. The bytes of a local file are passed to the server unchanged.

The two entry points answer two different questions:

- `prepare_image` turns whatever the caller has (a path, a URL) into a value the
  request can carry.
- `validate_image_url` rejects anything that is not already a URL, so a bare
  local file name is never sent to the server as the image.
"""

from __future__ import annotations

import base64
import pathlib

#: Media types the readout sends for local files, by lowercased file extension.
SUPPORTED_MEDIA_TYPES: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}

#: The extensions named in error messages, in the order a caller should read them.
SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")

HTTP_PREFIXES = ("http://", "https://")
DATA_PREFIX = "data:"


class ImageError(ValueError):
    """The image reference cannot be turned into a request the server accepts."""


def _sniff(data: bytes) -> str | None:
    """The media type of `data` from its leading bytes, or None if unsupported.

    A file extension is a claim; the signature is the evidence. A PNG named
    `.jpg` is sent as what it is rather than as what it is called.
    """
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if len(data) >= 12 and data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def validate_image_url(value: str) -> str:
    """Return an ``http(s)`` or ``data:`` image URL unchanged, or raise.

    `Readout.read_image` calls this before it builds a request. A bare local
    path is rejected rather than forwarded: the request carries the URL itself,
    so a file name that reached this point would be sent to the server as the
    image. Use `prepare_image` to turn a local file into a data URL first.
    """
    if not isinstance(value, str):
        raise ImageError(f"image_url must be a string, got {type(value).__name__}")
    url = value.strip()
    if not url:
        raise ImageError("image_url is empty")
    if url.startswith(HTTP_PREFIXES):
        if len(url) <= url.index("://") + 3:
            raise ImageError(f"image_url has no host: {url!r}")
        return url
    if url.startswith(DATA_PREFIX):
        if not url.startswith(f"{DATA_PREFIX}image/"):
            raise ImageError(f"image_url is a data URL that is not an image: {url[:32]!r}")
        if "," not in url or not url.split(",", 1)[1]:
            raise ImageError("image_url is a data URL with no payload")
        return url
    if "://" in url:
        scheme = url.split("://", 1)[0]
        raise ImageError(
            f"image_url scheme {scheme!r} is not supported; use http://, https://, or a data URL"
        )
    raise ImageError(
        f"image_url {url!r} is a local path, not a URL; call prepare_image() first so the "
        "server receives image bytes rather than a file name"
    )


def prepare_image(image: str | pathlib.Path) -> str:
    """Turn an image reference into the ``image_url`` a request can carry.

    - A local PNG, JPEG, or WebP file becomes a base64 ``data:`` URL.
    - An ``http(s)`` or ``data:`` URL passes through unchanged.
    - Anything else raises `ImageError` before any request is made: an empty
      value, an unsupported extension or URL scheme, a missing file, an empty
      file, or a file whose bytes are not a supported image.

    The file name is never part of the returned value, so a caller cannot leak a
    local path into a request by attaching an image.
    """
    if isinstance(image, pathlib.Path):
        text = str(image)
    elif isinstance(image, str):
        text = image.strip()
    else:
        raise ImageError(
            f"image must be a file path or a URL string, got {type(image).__name__}"
        )

    if not text:
        raise ImageError("image is empty")

    if text.startswith(HTTP_PREFIXES) or text.startswith(DATA_PREFIX):
        return validate_image_url(text)

    if "://" in text:
        scheme = text.split("://", 1)[0]
        raise ImageError(
            f"image scheme {scheme!r} is not supported; pass a local file, an http(s) URL, "
            "or a data URL"
        )

    path = pathlib.Path(text).expanduser()
    media_type = SUPPORTED_MEDIA_TYPES.get(path.suffix.lower())
    if media_type is None:
        shown = path.suffix or "(none)"
        raise ImageError(
            f"unsupported image extension {shown!r}; supported extensions are "
            f"{', '.join(SUPPORTED_EXTENSIONS)}"
        )
    if not path.is_file():
        raise ImageError(f"image file not found: {path}")

    data = path.read_bytes()
    if not data:
        raise ImageError(f"image file is empty: {path}")
    sniffed = _sniff(data)
    if sniffed is None:
        raise ImageError(
            f"image file is not a PNG, JPEG, or WebP image: {path}"
        )

    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{sniffed};base64,{encoded}"


__all__ = [
    "DATA_PREFIX",
    "HTTP_PREFIXES",
    "ImageError",
    "SUPPORTED_EXTENSIONS",
    "SUPPORTED_MEDIA_TYPES",
    "prepare_image",
    "validate_image_url",
]
