"""Root CLI. Workstream sub-CLIs are discovered lazily.

Each workstream package may expose ``wtc4d.<pkg>.cli:app`` (a ``typer.Typer``).
It is mounted as ``wtc4d <pkg> ...`` when importable; missing optional
dependencies simply hide the subcommand instead of breaking the root CLI.
"""

from __future__ import annotations

import importlib

import typer

WORKSTREAMS = ("corpus", "sync", "geo", "camreg", "procedural", "recon")

app = typer.Typer(no_args_is_help=True, help="911_4D toolkit")


@app.callback()
def _root() -> None:
    """911_4D: 4D reconstruction of the WTC attacks."""


@app.command()
def info() -> None:
    """Print world frame, epochs and which workstreams are installed."""
    from rich import print as rprint

    from wtc4d import __version__, timeline, world

    rprint(f"[bold]wtc4d[/bold] {__version__}")
    o = world.WORLD_ORIGIN
    rprint(f"world origin (ENU): lat={o.lat:.6f} lon={o.lon:.6f} alt={o.alt_m:.1f} m")
    for ep in timeline.EPOCHS:
        rprint(
            f"  {ep.id:<3} {ep.name:<34} {timeline.fmt_local(ep.t_start)} - {timeline.fmt_local(ep.t_end)}"
        )
    for ws in WORKSTREAMS:
        try:
            importlib.import_module(f"wtc4d.{ws}")
            status = "[green]installed[/green]"
        except Exception as exc:  # noqa: BLE001
            status = f"[yellow]missing[/yellow] ({type(exc).__name__})"
        rprint(f"  {ws:<12} {status}")


def _mount_workstreams() -> None:
    for ws in WORKSTREAMS:
        try:
            mod = importlib.import_module(f"wtc4d.{ws}.cli")
        except Exception:  # noqa: BLE001  (missing package or optional deps)
            continue
        sub = getattr(mod, "app", None)
        if isinstance(sub, typer.Typer):
            app.add_typer(sub, name=ws)


_mount_workstreams()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
