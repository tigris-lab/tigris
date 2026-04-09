"""``tigris codegen`` command."""

from pathlib import Path

import click

from tigris.cli import cli, console


@cli.command()
@click.argument("plan", type=click.Path(exists=True))
@click.option("--backend", "-b", type=click.Choice(["reference", "esp-nn", "cmsis-nn"]),
              default="reference", help="Kernel backend (default: reference)")
@click.option("--output", "-o", default=None, help="Output C file path")
def codegen(plan: str, backend: str, output: str | None):
    """Generate a C inference harness from a compiled .tgrs plan."""
    from tigris.emitters.codegen import generate_c

    plan_path = Path(plan)
    plan_data = plan_path.read_bytes()

    with console.status("Generating C source..."):
        c_source = generate_c(plan_data, backend)

    out = Path(output) if output else plan_path.with_suffix(".c")
    out.write_text(c_source)

    console.print(f"[bold green]C source written to {out}[/]")
    console.print(f"  backend: {backend}", style="dim")
    console.print(f"  size: {len(c_source)} bytes", style="dim")
