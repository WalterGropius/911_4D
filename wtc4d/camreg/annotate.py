"""A minimal local tool for clicking landmarks on a frame image.

Investigators use this to build the 2D-3D correspondences
:mod:`wtc4d.camreg.pnp` needs.  Two front ends share the same save format
(:mod:`wtc4d.camreg.annotations`):

* a **matplotlib** click UI (default, zero extra dependencies beyond the
  camreg extra) -- click a point on the image, type the landmark id (with
  autocompletion suggestions printed to the console) in the terminal prompt
  that pops up, repeat, close the window to save;
* an optional **FastAPI** page (``pip install fastapi uvicorn``, not part of
  the camreg dependency group) for a nicer clickable landmark picker in the
  browser -- used the same way from the CLI with ``--backend web``.

Neither ever touches the network: the frame image is read from local disk
(the caller extracts it from the frame store on the compute volume) and only
the resulting JSON is written into the repo.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from wtc4d.camreg.annotations import (
    FrameAnnotation,
    LandmarkObservation,
    annotation_path,
    save_annotation,
)
from wtc4d.camreg.landmarks import landmark_registry

__all__ = ["annotate_matplotlib", "annotate_web", "load_image_gray", "suggest_landmarks"]


def load_image_gray(path: str | Path) -> np.ndarray:
    """Load an image file as float32 grayscale in [0, 1] (for headless use/tests)."""
    from PIL import Image

    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.float32) / 255.0


def suggest_landmarks(prefix: str = "", limit: int = 15) -> list[str]:
    """Landmark ids starting with ``prefix``, for autocompletion / a picker list."""
    ids = sorted(landmark_registry())
    if prefix:
        ids = [i for i in ids if i.startswith(prefix)]
    return ids[:limit]


def annotate_matplotlib(
    image_path: str | Path,
    shot_id: str,
    frame_idx: int = 0,
    camera_prior_id: str | None = None,
    out_dir: str | Path | None = None,
    annotator: str = "",
) -> FrameAnnotation:
    """Interactive matplotlib click UI.  Click a point, then type its landmark
    id at the terminal prompt.  Close the window (or press 'q' after an empty
    prompt) to finish and save.

    Not exercised by CI (it blocks on user input and needs a display); see
    ``tests/test_camreg_pnp.py`` for the machine-checkable parts of the
    annotation pipeline instead.
    """
    import matplotlib.pyplot as plt

    from wtc4d.camreg.annotations import image_sha256

    img = load_image_gray(image_path)
    h, w = img.shape
    registry = landmark_registry()
    observations: list[LandmarkObservation] = []

    fig, ax = plt.subplots()
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"{shot_id} #{frame_idx} -- click a landmark, then Enter")
    scatter = ax.scatter([], [], c="red", s=30, marker="+")

    print(f"Known landmark ids ({len(registry)}): {', '.join(sorted(registry))}")
    print("Click a point in the image window, then answer the prompts here.")
    print("Empty landmark id ends the session.")

    def on_click(event) -> None:
        if event.xdata is None or event.ydata is None:
            return
        u, v = float(event.xdata), float(event.ydata)
        lid = input(f"  clicked ({u:.1f}, {v:.1f}) -- landmark id: ").strip()
        if not lid:
            plt.close(fig)
            return
        if lid not in registry:
            print(f"  ! unknown landmark id {lid!r}; skipped. Known ids listed above.")
            return
        sigma_str = input("  sigma_px [2.0]: ").strip()
        sigma = float(sigma_str) if sigma_str else 2.0
        observations.append(LandmarkObservation(landmark_id=lid, u=u, v=v, sigma_px=sigma))
        pts = np.array([[o.u, o.v] for o in observations])
        scatter.set_offsets(pts)
        ax.set_title(f"{shot_id} #{frame_idx} -- {len(observations)} points")
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()

    ann = FrameAnnotation(
        shot_id=shot_id,
        frame_idx=frame_idx,
        image_width=w,
        image_height=h,
        image_sha256=image_sha256(image_path),
        image_path=str(image_path),
        camera_prior_id=camera_prior_id,
        observations=observations,
        annotator=annotator,
    )
    path = save_annotation(ann, annotation_path(shot_id, frame_idx, out_dir) if out_dir else None)
    print(f"saved {len(observations)} observations to {path}")
    return ann


def annotate_web(
    image_path: str | Path,
    shot_id: str,
    frame_idx: int = 0,
    camera_prior_id: str | None = None,
    out_dir: str | Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> FrameAnnotation:
    """A single-page FastAPI annotator: click the image in a browser tab.

    Needs ``fastapi`` and ``uvicorn``, which are **not** in the camreg
    dependency group (they would be dead weight for the CLI-only workflow and
    for CI); install them yourself to use this backend. Serves on localhost
    only.
    """
    try:
        import uvicorn
        from fastapi import FastAPI, Request
        from fastapi.responses import HTMLResponse, JSONResponse
    except ImportError as exc:  # noqa: BLE001
        raise ImportError(
            "the web annotator needs fastapi and uvicorn: pip install fastapi uvicorn"
        ) from exc
    import base64
    import io

    from PIL import Image

    from wtc4d.camreg.annotations import image_sha256

    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data_uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    registry = landmark_registry()
    state: dict = {"observations": []}

    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        options = "".join(f'<option value="{i}">{i}</option>' for i in sorted(registry))
        return f"""
        <html><body style="margin:0;background:#111;color:#eee;font-family:sans-serif">
        <div style="display:flex">
          <img id="img" src="{data_uri}" style="max-width:75vw;cursor:crosshair">
          <div style="padding:1em">
            <h3>{shot_id} #{frame_idx}</h3>
            <select id="lid">{options}</select><br><br>
            sigma_px <input id="sigma" value="2.0" size="4"><br><br>
            <ul id="log"></ul>
            <button onclick="finish()">Save &amp; quit</button>
          </div>
        </div>
        <script>
        const img = document.getElementById('img');
        img.onclick = (e) => {{
          const r = img.getBoundingClientRect();
          const u = (e.clientX - r.left) * {w} / r.width;
          const v = (e.clientY - r.top) * {h} / r.height;
          const lid = document.getElementById('lid').value;
          const sigma = parseFloat(document.getElementById('sigma').value || '2.0');
          fetch('/click', {{method:'POST', headers:{{'Content-Type':'application/json'}},
            body: JSON.stringify({{landmark_id: lid, u, v, sigma_px: sigma}})}})
            .then(() => {{
              const li = document.createElement('li');
              li.textContent = lid + ': (' + u.toFixed(1) + ', ' + v.toFixed(1) + ')';
              document.getElementById('log').appendChild(li);
            }});
        }};
        function finish() {{ fetch('/done', {{method:'POST'}}).then(() => window.close()); }}
        </script>
        </body></html>
        """

    @app.post("/click")
    async def click(request: Request):
        body = await request.json()
        state["observations"].append(LandmarkObservation(**body))
        return JSONResponse({"ok": True})

    @app.post("/done")
    async def done():
        raise SystemExit(0)

    print(f"open http://{host}:{port} in a browser, click landmarks, then 'Save & quit'")
    try:
        uvicorn.run(app, host=host, port=port, log_level="warning")
    except SystemExit:
        pass

    ann = FrameAnnotation(
        shot_id=shot_id,
        frame_idx=frame_idx,
        image_width=w,
        image_height=h,
        image_sha256=image_sha256(image_path),
        image_path=str(image_path),
        camera_prior_id=camera_prior_id,
        observations=state["observations"],
    )
    from wtc4d.camreg.annotations import annotation_path

    path = save_annotation(ann, annotation_path(shot_id, frame_idx, out_dir) if out_dir else None)
    print(f"saved {len(state['observations'])} observations to {path}")
    return ann
