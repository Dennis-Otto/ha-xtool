"""A camera's live video, decoded to JPEG frames for its live viewers.

The X1 streams H.264 (see ``X1WSV2Protocol.live_video``); Home
Assistant's camera live view here is a multipart MJPEG stream, so the
video is decoded with PyAV (which Home Assistant ships for its stream
integration) and the newest picture is encoded as JPEG, at most
``fps`` times a second. One video runs per camera, however many viewers
watch it; it starts with the first viewer and stops after the last.
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)

# Pictures a second sent to the viewers (the X1 films about 20)
LIVE_VIDEO_FPS = 10
LIVE_VIDEO_JPEG_QUALITY = 80


def _new_decoder() -> Any:
    import av  # Home Assistant ships PyAV for its stream integration

    return av.CodecContext.create("h264", "r")


def _decode(decoder: Any, chunk: bytes) -> Any:
    """Feed H.264 to the decoder; the newest picture it completes, if any."""
    picture = None
    for packet in decoder.parse(chunk):
        for frame in decoder.decode(packet):
            picture = frame
    return picture


def _to_jpeg(picture: Any) -> bytes:
    out = io.BytesIO()
    picture.to_image().save(out, format="JPEG", quality=LIVE_VIDEO_JPEG_QUALITY)
    return out.getvalue()


class LiveVideo:
    """One camera's live video, shared by its viewers."""

    def __init__(
        self,
        hass: HomeAssistant,
        source: Callable[[], AsyncIterator[bytes]],
        name: str,
        fps: float = LIVE_VIDEO_FPS,
    ) -> None:
        self._hass = hass
        self._source = source
        self._name = name
        self._interval = 1 / fps
        self._viewers = 0
        self._task: asyncio.Task[None] | None = None
        # False once the running video has ended
        self._running = False
        self._jpeg: bytes | None = None
        self._count = 0
        self._changed = asyncio.Condition()

    async def frames(self) -> AsyncIterator[bytes]:
        """The video's pictures as JPEG while the caller reads them.

        Ends when the video does (another camera is live, the laser
        stopped it, the connection dropped); the caller then falls back
        to photos.
        """
        self._viewers += 1
        if self._task is None or self._task.done():
            self._jpeg, self._count, self._running = None, 0, True
            self._task = self._hass.async_create_background_task(
                self._run(), f"xTool {self._name} live video",
            )
        task = self._task
        seen = self._count
        try:
            while True:
                async with self._changed:
                    await self._changed.wait_for(lambda: self._count != seen or not self._running)
                if self._count == seen:
                    return
                seen = self._count
                if self._jpeg is not None:
                    yield self._jpeg
        finally:
            self._viewers -= 1
            if self._viewers == 0 and not task.done():
                task.cancel()

    async def _run(self) -> None:
        try:
            decoder = await self._hass.async_add_executor_job(_new_decoder)
            last = float("-inf")
            async for chunk in self._source():
                picture = await self._hass.async_add_executor_job(_decode, decoder, chunk)
                if picture is None or time.monotonic() - last < self._interval:
                    continue
                last = time.monotonic()
                jpeg = await self._hass.async_add_executor_job(_to_jpeg, picture)
                async with self._changed:
                    self._jpeg = jpeg
                    self._count += 1
                    self._changed.notify_all()
        except asyncio.CancelledError:
            raise
        except Exception as err:  # noqa: BLE001
            _LOGGER.debug("%s live video ended: %r", self._name, err)
        finally:
            async with self._changed:
                self._running = False
                self._changed.notify_all()
