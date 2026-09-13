# infra/

Compute lives on [Modal](https://modal.com).  This directory is owned by the
`infra` workstream: Modal apps (`infra/modal/`), the GitHub Actions dispatch
workflow that runs them, and volume layout documentation.

Important constraint: Claude Code cloud sessions cannot reach Modal's gRPC
API directly (the egress proxy does not support gRPC).  Modal jobs are
therefore launched from GitHub Actions (`workflow_dispatch`) using the
repository secrets `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`, or from a
contributor's own machine with `modal run`.
