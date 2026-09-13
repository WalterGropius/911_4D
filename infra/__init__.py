"""infra: Modal compute apps, job dispatch and GitHub Actions plumbing.

Owned by the ``infra`` workstream (see ``CONTRIBUTING.md``).  Importing any
module under this package must never contact Modal, GitHub or any other
network service — it only builds Python objects (App, Image, Volume
definitions) that are resolved lazily when actually run or deployed.
"""
