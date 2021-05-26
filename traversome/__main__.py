#! /usr/bin/env python

"""
Command-line Interface to traversome
"""

import os
from pathlib import Path
from enum import Enum
import typer
from traversome.traversome import Traversome
from traversome import __version__

# add the -h option for showing help
CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


class LogLevel(str, Enum):
    """categorical options for loglevel to CLI"""
    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


# creates the top-level traversome app
app = typer.Typer(add_completion=False, context_settings=CONTEXT_SETTINGS)


def version_callback(value: bool):
    """Adding a --version option to the CLI"""
    if value:
        typer.echo(f"traversome {__version__}")
        raise typer.Exit()


def docs_callback(value: bool):
    """function to open docs"""
    if value:
        typer.echo("Opening https://eaton-lab.org/traversome in default browser")
        typer.launch("https://eaton-lab.org/traversome")
        raise typer.Exit()


@app.callback()
def main(
        version: bool = typer.Option(None, "--version", "-v", callback=version_callback, is_eager=True,
                                     help="print version and exit."),
        docs: bool = typer.Option(None, "--docs", callback=docs_callback, is_eager=True,
                                  help="Open documentation in browser."),
):
    """
    Call traversome commands to access tools in the traversome toolkit,
    and traversome COMMAND -h to see help options for each tool
    (e.g., traversome ml -h)
    """
    typer.secho(
        f"traversome (v.{__version__}): genomic isomer frequency estimator",
        fg=typer.colors.MAGENTA, bold=True,
    )


class PathGen(str, Enum):
    Heuristic = "H"
    Provided = "U"


@app.command()
def ml(
    graph_file: Path = typer.Option(
        ..., "-g", "--graph",
        help="GFA/FASTG format Graph file",
        exists=True, resolve_path=True),
    alignment_file: Path = typer.Option(
        ..., "-a", "--alignment",
        help="GAF format alignment file",
        exists=True, resolve_path=True,
    ),
    output_dir: Path = typer.Option(
        './', "-o", "--output",
        help="Output directory",
        exists=False, resolve_path=True),
    path_generator: PathGen = typer.Option(
        PathGen.Heuristic, "-P",
        help="Path generator: H (Heuristic)/U (User-provided)"),
    random_seed: int = typer.Option(12345, "--rs", "--random-seed", help="Random seed. "),
    linear_chr: bool = typer.Option(False, "-L", help="Chromosome topology NOT forced to be circular. "),
        out_seq_threshold: float = typer.Option(
            0.001, "-S",
            help="Threshold for sequence output",
            min=0, max=1),
    keep_temp: float = typer.Option(False, "--keep-temp", help="Keep temporary files for debug. "),
    log_level: LogLevel = typer.Option(
        LogLevel.INFO, "--loglevel", "--log-level", help="Logging level. Use DEBUG for more, ERROR for less."),
    ):
    """
    Conduct Maximum Likelihood analysis for solving assembly graph
    Examples:
    traversome ml -g graph.gfa -a align.gaf -o .
    """
    os.makedirs(str(output_dir), exist_ok=True)
    traverser = Traversome(
        graph=str(graph_file),
        alignment=str(alignment_file),
        outdir=str(output_dir),
        out_prob_threshold=out_seq_threshold,
        force_circular=not linear_chr,
        random_seed=random_seed,
        keep_temp=keep_temp,
        loglevel=log_level
    )
    traverser.run(
        path_generator=path_generator,
        multi_chromosomes=True  # opts.is_multi_chromosomes,
        )


@app.command()
def mc(
    graph_file: Path = typer.Option(
        ..., "-g", "--graph",
        help="GFA/FASTG format Graph file",
        exists=True, resolve_path=True),
    alignment_file: Path = typer.Option(
        ..., "-a", "--alignment",
        help="GAF format alignment file",
        exists=True, resolve_path=True,
        ),
    output_dir: Path = typer.Option(
        './', "-o", "--output",
        help="Output directory",
        exists=False, resolve_path=True),
    path_generator: PathGen = typer.Option(
        PathGen.Heuristic, "-P",
        help="Path generator: H (Heuristic)/U (User-provided)"),
    random_seed: int = typer.Option(12345, "--rs", "--random-seed", help="Random seed"),
    linear_chr: bool = typer.Option(False, "-L", help="Chromosome topology NOT forced to be circular. "),
    out_seq_threshold: float = typer.Option(
        0.001, "-S",
        help="Threshold for sequence output",
        min=0, max=1),
    n_generations: int = typer.Option(10000, "--mcmc", help="MCMC generations"),
    n_burn: int = typer.Option(1000, "--burn", help="MCMC Burn-in"),
    keep_temp: bool = typer.Option(False, help="Keep temporary files"),
    log_level: LogLevel = typer.Option(
        LogLevel.INFO, "--loglevel", help="Logging level. Use DEBUG for more, ERROR for less."),
    ):
    """
    Conduct Bayesian MCMC analysis for solving assembly graph
    Examples:
    traversome mc -g graph.gfa -a align.gaf -o .
    """
    os.makedirs(str(output_dir), exist_ok=True)
    traverser = Traversome(
        graph=str(graph_file),
        alignment=str(alignment_file),
        outdir=str(output_dir),
        out_prob_threshold=out_seq_threshold,
        do_bayesian=True,
        force_circular=not linear_chr,
        n_generations=n_generations,
        n_burn=n_burn,
        random_seed=random_seed,
        keep_temp=keep_temp,
        loglevel=log_level
    )
    traverser.run(
        path_generator=path_generator,
        multi_chromosomes=True  # opts.is_multi_chromosomes,
    )
