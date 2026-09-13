/** Unobtrusive attribution footer. */
export function createFooter(onAboutClick: () => void): HTMLElement {
  const el = document.createElement("div");
  el.className = "attribution";
  const aboutBtn = document.createElement("button");
  aboutBtn.type = "button";
  aboutBtn.textContent = "About";
  aboutBtn.style.background = "none";
  aboutBtn.style.border = "none";
  aboutBtn.style.padding = "0";
  aboutBtn.style.font = "inherit";
  aboutBtn.style.textDecoration = "underline";
  aboutBtn.style.color = "var(--fg-dim)";
  aboutBtn.addEventListener("click", onAboutClick);
  el.appendChild(aboutBtn);
  el.appendChild(document.createTextNode(" · 911_4D, open source · "));
  const link = document.createElement("a");
  link.href = "https://github.com/WalterGropius/911_4D";
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  link.textContent = "source & data provenance";
  el.appendChild(link);
  return el;
}
