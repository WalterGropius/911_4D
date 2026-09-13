# Security Policy

## Reporting a vulnerability

If you find a security vulnerability in this repository's code
(`wtc4d/`, `infra/`, `web/`) — for example, something that could leak the
Modal compute credentials described in `infra/README.md`, execute
arbitrary code via a malicious data file, or expose data that shouldn't be
public — please report it privately rather than opening a public issue.

Use GitHub's private vulnerability reporting for this repository
(**Security → Report a vulnerability** on the repository's GitHub page)
if it is enabled. If it is not yet enabled, open an issue titled
`[security] <short description>` with only enough detail to confirm the
issue is real and let a maintainer follow up for details privately — do
not post exploit details, credentials, or sensitive data in a public
issue.

We aim to acknowledge a report within 5 business days and to have a fix
or mitigation plan within 30 days for a confirmed issue, consistent with
the response-time commitments in `docs/ethics.md` and `docs/licensing.md`.

## Secrets and credentials

This project's compute runs on Modal, dispatched from GitHub Actions using
the repository secrets `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` (see
`infra/README.md`). Never commit credentials, tokens, or `.env` files
(see `.gitignore`); if you believe a secret has been committed or leaked,
report it the same way as a vulnerability above so it can be rotated
promptly, and treat that report as more urgent than the 30-day window
above — flag it as a live credential leak, not a routine bug.

## Content and takedown requests

This policy covers *security* vulnerabilities in code and infrastructure.
For a takedown request, a rights-holder dispute, or a request from a
victim's family regarding how footage is used or depicted, see
`docs/ethics.md` ("Requests from victims' families and rights-holders")
and `docs/licensing.md` ("DMCA / takedown process") instead — those follow
the same response-time commitments but a different review process, since
they are not code vulnerabilities.

## Supported versions

This project is in early, active development (see `README.md`: "phase 1
scaffolding") with no tagged releases yet; security fixes are applied to
`main` only.
