/**
 * Content warning splash. Blocks interaction with the viewer until
 * dismissed; remembers dismissal for the session (not permanently — this is
 * sensitive historical material and the warning should resurface across
 * visits) via `sessionStorage`.
 */
const SESSION_KEY = "wtc4d.splash.dismissed";

export function hasSplashBeenDismissedThisSession(): boolean {
  try {
    return sessionStorage.getItem(SESSION_KEY) === "1";
  } catch {
    return false;
  }
}

export function createSplash(onContinue: () => void): HTMLElement {
  const el = document.createElement("div");
  el.className = "splash";
  el.setAttribute("role", "dialog");
  el.setAttribute("aria-modal", "true");
  el.setAttribute("aria-labelledby", "splash-title");

  const card = document.createElement("div");
  card.className = "splash__card";
  el.appendChild(card);

  const h1 = document.createElement("h1");
  h1.id = "splash-title";
  h1.textContent = "Before you continue";
  card.appendChild(h1);

  const p1 = document.createElement("p");
  p1.textContent =
    "This is a navigable 4D reconstruction of the World Trade Center attacks of " +
    "September 11, 2001, built from public footage for research, education, and " +
    "remembrance. It includes the towers' collapse and the aftermath.";
  card.appendChild(p1);

  const p2 = document.createElement("p");
  p2.textContent =
    "The reconstruction focuses on structures and event geometry, not on individual " +
    "people. Geometry shown here that is not yet backed by registered footage is " +
    "clearly marked as illustrative/mock in this build.";
  card.appendChild(p2);

  const actions = document.createElement("div");
  actions.className = "splash__actions";
  card.appendChild(actions);

  const continueBtn = document.createElement("button");
  continueBtn.type = "button";
  continueBtn.className = "splash__continue";
  continueBtn.textContent = "I understand, continue";
  actions.appendChild(continueBtn);

  const dontShow = document.createElement("label");
  const checkbox = document.createElement("input");
  checkbox.type = "checkbox";
  dontShow.appendChild(checkbox);
  dontShow.appendChild(document.createTextNode("Don't show again this session"));
  actions.appendChild(dontShow);

  continueBtn.addEventListener("click", () => {
    if (checkbox.checked) {
      try {
        sessionStorage.setItem(SESSION_KEY, "1");
      } catch {
        // ignore (private browsing etc.)
      }
    }
    onContinue();
  });

  queueMicrotask(() => continueBtn.focus());
  return el;
}
