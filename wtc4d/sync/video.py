"""Media access helpers built on ffmpeg/ffprobe.

Everything the sync workstream needs from a media file goes through here so
that the rest of the package never has to care whether the input is a video
file, a URL, or a directory of extracted frames.

Why ffmpeg rather than ``cv2.VideoCapture``: we need *exact* frame->second
mapping on 2001-era MPEG-2 with dropped frames and non-integer rates, plus
accurate seeking into remote files.  ``ffmpeg -ss <t> -i <input>`` performs an
accurate seek (keyframe seek followed by decode-and-discard), so frame ``k`` of
the output is at ``t + k / fps`` of the input.  ``-c copy`` does *not* -- it
snaps to the previous keyframe -- so we never use it here.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FFMPEG = os.environ.get("WTC4D_FFMPEG", "ffmpeg")
FFPROBE = os.environ.get("WTC4D_FFPROBE", "ffprobe")

# Frame-file extensions recognised when a "frames dir" is given instead of a video.
FRAME_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


class FfmpegMissingError(RuntimeError):
    """Raised when ffmpeg/ffprobe are needed but not installed."""


def have_ffmpeg() -> bool:
    return shutil.which(FFMPEG) is not None and shutil.which(FFPROBE) is not None


def require_ffmpeg() -> None:
    if not have_ffmpeg():
        raise FfmpegMissingError(
            f"{FFMPEG!r}/{FFPROBE!r} not found on PATH; install ffmpeg "
            "(apt-get install ffmpeg) or set WTC4D_FFMPEG/WTC4D_FFPROBE"
        )


@dataclass(frozen=True)
class MediaInfo:
    """What we need to know about a media file to time it."""

    path: str
    duration_s: float | None
    fps: float | None
    width: int | None
    height: int | None
    nb_frames: int | None
    has_audio: bool
    audio_sample_rate: int | None
    format_tags: dict[str, str]
    stream_tags: dict[str, str]

    @property
    def frame_count(self) -> int | None:
        if self.nb_frames:
            return self.nb_frames
        if self.duration_s and self.fps:
            return int(round(self.duration_s * self.fps))
        return None


def _run(cmd: list[str], *, stdout: int | None = subprocess.PIPE) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=stdout, stderr=subprocess.PIPE, check=False)


def _parse_rate(value: str | None) -> float | None:
    """``'30000/1001'`` -> 29.97002997..., ``'0/0'`` -> None."""
    if not value:
        return None
    if "/" in value:
        num, _, den = value.partition("/")
        try:
            n, d = float(num), float(den)
        except ValueError:
            return None
        return n / d if d else None
    try:
        return float(value)
    except ValueError:
        return None


def probe(path: str | Path) -> MediaInfo:
    """ffprobe a file (or URL) into a :class:`MediaInfo`."""
    require_ffmpeg()
    cmd = [
        FFPROBE,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    proc = _run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {proc.stderr.decode(errors='replace')}")
    data = json.loads(proc.stdout.decode())
    fmt = data.get("format", {})
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    duration = fmt.get("duration")
    fps = None
    width = height = nb_frames = None
    stream_tags: dict[str, str] = {}
    if video is not None:
        fps = _parse_rate(video.get("avg_frame_rate")) or _parse_rate(video.get("r_frame_rate"))
        width = int(video["width"]) if video.get("width") else None
        height = int(video["height"]) if video.get("height") else None
        if video.get("nb_frames", "N/A") not in ("N/A", None):
            try:
                nb_frames = int(video["nb_frames"])
            except ValueError:
                nb_frames = None
        if duration is None:
            duration = video.get("duration")
        stream_tags = {str(k): str(v) for k, v in (video.get("tags") or {}).items()}

    return MediaInfo(
        path=str(path),
        duration_s=float(duration) if duration not in (None, "N/A") else None,
        fps=fps,
        width=width,
        height=height,
        nb_frames=nb_frames,
        has_audio=audio is not None,
        audio_sample_rate=int(audio["sample_rate"]) if audio and audio.get("sample_rate") else None,
        format_tags={str(k): str(v) for k, v in (fmt.get("tags") or {}).items()},
        stream_tags=stream_tags,
    )


# --- frame access ------------------------------------------------------------


@dataclass(frozen=True)
class DecodedFrame:
    """One decoded frame.

    ``index`` counts from the first frame of the *file* (not of the iteration
    window), so it can be used directly as a shot-relative frame index when
    the file is the shot.  ``t_rel`` is seconds from the start of the file.
    ``image`` is HxWx3 uint8 BGR (OpenCV order) or HxW uint8 when grayscale.
    """

    index: int
    t_rel: float
    image: np.ndarray


def iter_video_frames(
    path: str | Path,
    *,
    start_s: float = 0.0,
    duration_s: float | None = None,
    fps: float | None = None,
    width: int | None = None,
    height: int | None = None,
    grayscale: bool = False,
    src_fps: float | None = None,
) -> Iterator[DecodedFrame]:
    """Decode frames through an ffmpeg rawvideo pipe.

    ``fps`` resamples the output (use it to subsample cheaply, e.g. 4 fps for
    a first pass); when ``None`` every frame is emitted.  ``width``/``height``
    downscale.  ``src_fps`` overrides the probed rate when computing
    ``index``/``t_rel`` (needed for variable-frame-rate sources).
    """
    require_ffmpeg()
    info = probe(path)
    in_fps = src_fps or info.fps
    if in_fps is None:
        raise RuntimeError(f"cannot determine frame rate of {path}")
    out_fps = fps or in_fps
    out_w = width or info.width
    out_h = height or info.height
    if out_w is None or out_h is None:
        raise RuntimeError(f"cannot determine frame size of {path}")

    filters = []
    if fps is not None:
        filters.append(f"fps={out_fps}")
    if width is not None or height is not None:
        filters.append(f"scale={out_w}:{out_h}")
    pix_fmt = "gray" if grayscale else "bgr24"
    channels = 1 if grayscale else 3
    frame_bytes = out_w * out_h * channels

    cmd = [FFMPEG, "-nostdin", "-loglevel", "error"]
    if start_s:
        cmd += ["-ss", f"{start_s:.6f}"]
    cmd += ["-i", str(path)]
    if duration_s is not None:
        cmd += ["-t", f"{duration_s:.6f}"]
    if filters:
        cmd += ["-vf", ",".join(filters)]
    cmd += ["-f", "rawvideo", "-pix_fmt", pix_fmt, "-vsync", "0", "-"]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None
    try:
        k = 0
        while True:
            buf = proc.stdout.read(frame_bytes)
            if not buf or len(buf) < frame_bytes:
                break
            arr = np.frombuffer(buf, dtype=np.uint8)
            img = arr.reshape((out_h, out_w) if grayscale else (out_h, out_w, 3))
            t_rel = start_s + k / out_fps
            yield DecodedFrame(index=int(round(t_rel * in_fps)), t_rel=t_rel, image=img)
            k += 1
    finally:
        if proc.stdout is not None:
            proc.stdout.close()
        proc.wait()


def list_frame_files(frames_dir: str | Path) -> list[Path]:
    """Sorted image files in a frames directory (``%06d.jpg`` style)."""
    d = Path(frames_dir)
    return sorted(p for p in d.iterdir() if p.suffix.lower() in FRAME_SUFFIXES)


def iter_frame_files(
    frames_dir: str | Path,
    *,
    fps: float,
    start_index: int = 0,
    step: int = 1,
    limit: int | None = None,
    grayscale: bool = False,
) -> Iterator[DecodedFrame]:
    """Iterate a directory of extracted frames as if it were a video.

    The frame index is taken from the *position in the sorted listing* plus
    ``start_index``; corpus extracts frames as ``%06d.jpg`` from the shot
    start, so position and shot-relative frame index agree.
    """
    import cv2  # local import: keeps `import wtc4d.sync` free of heavy deps

    files = list_frame_files(frames_dir)
    n = 0
    for pos in range(0, len(files), step):
        if limit is not None and n >= limit:
            return
        flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
        img = cv2.imread(str(files[pos]), flag)
        if img is None:
            continue
        idx = start_index + pos
        yield DecodedFrame(index=idx, t_rel=idx / fps, image=img)
        n += 1


class FrameSource:
    """Uniform frame access over a video file or a directory of frames."""

    def __init__(
        self,
        path: str | Path,
        *,
        fps: float | None = None,
        start_index: int = 0,
    ) -> None:
        self.path = Path(path)
        self.is_dir = self.path.is_dir()
        self.start_index = start_index
        if self.is_dir:
            if fps is None:
                raise ValueError("fps is required when reading a frames directory")
            self.fps = fps
            self.info = None
            self._count: int | None = len(list_frame_files(self.path))
        else:
            self.info = probe(self.path)
            self.fps = fps or self.info.fps or 0.0
            if not self.fps:
                raise RuntimeError(f"cannot determine fps of {self.path}")
            self._count = self.info.frame_count

    @property
    def frame_count(self) -> int | None:
        return self._count

    def iter(
        self,
        *,
        start_s: float = 0.0,
        duration_s: float | None = None,
        sample_fps: float | None = None,
        grayscale: bool = False,
        max_width: int | None = None,
    ) -> Iterator[DecodedFrame]:
        """Yield frames, optionally subsampled in time and downscaled."""
        if self.is_dir:
            step = 1
            if sample_fps:
                step = max(1, int(round(self.fps / sample_fps)))
            first = int(round(start_s * self.fps))
            limit = None
            if duration_s is not None:
                limit = max(1, int(round(duration_s * self.fps / step)))
            it = iter_frame_files(
                self.path,
                fps=self.fps,
                start_index=self.start_index,
                step=step,
                limit=limit,
                grayscale=grayscale,
            )
            for fr in it:
                if fr.index < first:
                    continue
                yield _maybe_downscale(fr, max_width)
            return

        width = height = None
        if max_width and self.info and self.info.width and self.info.width > max_width:
            scale = max_width / self.info.width
            width = max_width
            height = int(round((self.info.height or 0) * scale)) // 2 * 2
        for fr in iter_video_frames(
            self.path,
            start_s=start_s,
            duration_s=duration_s,
            fps=sample_fps,
            width=width,
            height=height,
            grayscale=grayscale,
            src_fps=self.fps,
        ):
            yield fr


def _maybe_downscale(fr: DecodedFrame, max_width: int | None) -> DecodedFrame:
    if not max_width or fr.image.shape[1] <= max_width:
        return fr
    import cv2

    scale = max_width / fr.image.shape[1]
    img = cv2.resize(
        fr.image,
        (max_width, max(1, int(round(fr.image.shape[0] * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return DecodedFrame(index=fr.index, t_rel=fr.t_rel, image=img)


# --- audio access ------------------------------------------------------------


def load_audio(
    path: str | Path,
    *,
    sample_rate: int = 16000,
    start_s: float = 0.0,
    duration_s: float | None = None,
) -> np.ndarray:
    """Decode to mono float32 in [-1, 1] at ``sample_rate``.

    8-16 kHz mono is plenty: the cues we align on (speech envelope, the
    low-frequency collapse rumble) live well below 4 kHz, and the sources are
    heavily re-encoded anyway.
    """
    require_ffmpeg()
    cmd = [FFMPEG, "-nostdin", "-loglevel", "error"]
    if start_s:
        cmd += ["-ss", f"{start_s:.6f}"]
    cmd += ["-i", str(path)]
    if duration_s is not None:
        cmd += ["-t", f"{duration_s:.6f}"]
    cmd += [
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-",
    ]
    proc = _run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg audio decode failed for {path}: {proc.stderr.decode()[-2000:]}")
    return np.frombuffer(proc.stdout, dtype="<f4").astype(np.float32, copy=True)


def write_wav(path: str | Path, x: np.ndarray, sample_rate: int) -> None:
    """Write mono float32 samples as 16-bit PCM WAV (stdlib only)."""
    import wave

    pcm = np.clip(np.asarray(x, dtype=np.float32), -1.0, 1.0)
    data = (pcm * 32767.0).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(data)


__all__ = [
    "DecodedFrame",
    "FfmpegMissingError",
    "FrameSource",
    "MediaInfo",
    "have_ffmpeg",
    "iter_frame_files",
    "iter_video_frames",
    "list_frame_files",
    "load_audio",
    "probe",
    "require_ffmpeg",
    "write_wav",
]
