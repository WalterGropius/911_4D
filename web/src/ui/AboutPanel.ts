/** About / methodology modal. Short by design; links out to `docs/` for depth. */
const REPO_DOCS_BASE = "https://github.com/WalterGropius/911_4D/blob/main/docs";

export function createAboutModal(): { el: HTMLElement; open: () => void; close: () => void } {
  const backdrop = document.createElement("div");
  backdrop.className = "modal-backdrop";
  backdrop.hidden = true;
  backdrop.setAttribute("role", "presentation");

  const modal = document.createElement("div");
  modal.className = "modal";
  modal.setAttribute("role", "dialog");
  modal.setAttribute("aria-modal", "true");
  modal.setAttribute("aria-labelledby", "about-title");
  modal.tabIndex = -1;
  backdrop.appendChild(modal);

  modal.innerHTML = `
    <h2 id="about-title">About 911_4D</h2>
    <p>
      911_4D is an open-source, navigable 4D (3D + time) reconstruction of the
      World Trade Center attacks, built from publicly available footage. Every
      element is meant to be traceable back to its source frames, with
      uncertainty treated as a first-class property rather than hidden away.
    </p>
    <p>
      This viewer renders a <code>SceneManifest</code>: time-tagged Gaussian
      splat assets (towers, smoke, debris, ground), registered camera poses,
      and named events. Cameras registered against real footage can be
      "flown to" and compared against their source frame in the evidence
      view.
    </p>
    <dl>
      <dt>Time convention</dt>
      <dd>Seconds since 2001-09-11 00:00 EDT. 08:46:30 = first impact.</dd>
      <dt>World frame</dt>
      <dd>Local East-North-Up metres, origin between the towers.</dd>
      <dt>Status</dt>
      <dd>
        The bundled sample scene is a synthetic placeholder (simple boxes and
        mounds at the towers' real positions/heights) so the timeline and
        camera systems can be exercised before real reconstructions are
        published.
      </dd>
    </dl>
    <p>
      <a href="${REPO_DOCS_BASE}/methodology.md" target="_blank" rel="noopener noreferrer">Methodology</a>
      &middot;
      <a href="${REPO_DOCS_BASE}/ethics.md" target="_blank" rel="noopener noreferrer">Ethics &amp; guardrails</a>
      &middot;
      <a href="${REPO_DOCS_BASE}/licensing.md" target="_blank" rel="noopener noreferrer">Licensing</a>
    </p>
  `;

  const closeBtn = document.createElement("button");
  closeBtn.type = "button";
  closeBtn.textContent = "Close";
  closeBtn.style.marginTop = "1rem";
  modal.appendChild(closeBtn);

  function close() {
    backdrop.hidden = true;
  }
  function open() {
    backdrop.hidden = false;
    modal.focus();
  }

  closeBtn.addEventListener("click", close);
  backdrop.addEventListener("click", (e) => {
    if (e.target === backdrop) close();
  });
  backdrop.addEventListener("keydown", (e) => {
    if (e.key === "Escape") close();
  });

  return { el: backdrop, open, close };
}
