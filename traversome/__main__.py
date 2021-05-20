#! /usr/bin/env python

"""
Command-line Interface to traversome
"""

import os
import typer
from enum import Enum
from .traversome import Traversome
from traversome import __version__

# add the -h option for showing help
CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


class LogLevel(str, Enum):
    """categorical options for loglevel to CLI"""
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


@app.command()
def ml(
    graph_file: str = typer.Option(None, "-g", "--graph", help="GFA/FASTG format Graph file. "),
    alignment_file: str = typer.Option(None, "-a", "--alignment", help="GAF format alignment file. "),
    output_dir: str = typer.Option(None, "-o", "--output", help="Output directory. "),
    path_generator: str = typer.Option("H", "-P", help="Path generator: H (Heuristic)/U (User-provided)."),
    linear_chr: bool = typer.Option(False, "-L", help="Chromosome topology NOT forced to be circular. "),
    out_seq_threshold: float = typer.Option(0.001, "-S", help="Output sequences with proportion >= %(default)s"),
    keep_temp: float = typer.Option(False, "--keep-temp", help="Keep temporary files for debug. Default: %(default)s"),
    log_level: LogLevel = typer.Option(LogLevel.INFO, help="Logging level. Use DEBUG for more, ERROR for less."),
    ):
    """
    Conduct Maximum Likelihood analysis for solving assembly graph
    Examples:
    traversome ml -g graph.gfa -a align.gaf -o .
    """
    if not os.path.isfile(graph_file):
        raise IOError(graph_file + " not found/valid!")
    if not os.path.isfile(alignment_file):
        raise IOError(alignment_file + " not found/valid!")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    assert path_generator in ("H", "U")
    assert 0 <= out_seq_threshold <= 1
    try:
        traverser = Traversome(
            graph=graph_file,
            alignment=alignment_file,
            outdir=output_dir,
            out_prob_threshold=out_seq_threshold,
            keep_temp=keep_temp,
            loglevel=log_level
        )
        traverser.run(
            path_generator=path_generator,
            multi_chromosomes=True,  # opts.is_multi_chromosomes,
            force_circular=not linear_chr)
    except:
        typer.Exit()


@app.command()
def mc(
    graph_file: str = typer.Option(None, "-g", "--graph", help="GFA/FASTG format Graph file. "),
    alignment_file: str = typer.Option(None, "-a", "--alignment", help="GAF format alignment file. "),
    output_dir: str = typer.Option(None, "-o", "--output", help="Output directory. "),
    path_generator: str = typer.Option("H", "-P", help="Path generator: H (Heuristic)/U (User-provided)."),
    linear_chr: bool = typer.Option(False, "-L", help="Chromosome topology NOT forced to be circular. "),
    out_seq_threshold: float = typer.Option(0.001, "-S", help="Output sequences with proportion >= %(default)s"),
    n_generations: int = typer.Option(10000, "--mcmc", help="Number of MCMC generations. "),
    n_burn: int = typer.Option(1000, "--burn", help="Number of MCMC Burn-in. "),
    keep_temp: float = typer.Option(False, "--keep-temp", help="Keep temporary files for debug. Default: %(default)s"),
    log_level: LogLevel = typer.Option(LogLevel.INFO, help="Logging level. Use DEBUG for more, ERROR for less."),
    ):
    """
    Conduct Bayesian MCMC analysis for solving assembly graph
    Examples:
    traversome mc -g graph.gfa -a align.gaf -o .
    """
    if not os.path.isfile(graph_file):
        raise IOError(graph_file + " not found/valid!")
    if not os.path.isfile(alignment_file):
        raise IOError(alignment_file + " not found/valid!")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    assert path_generator in ("H", "U")
    assert 0 <= out_seq_threshold <= 1
    try:
        traverser = Traversome(
            graph=graph_file,
            alignment=alignment_file,
            outdir=output_dir,
            out_prob_threshold=out_seq_threshold,
            do_bayesian=True,
            n_generations=n_generations,
            n_burn=n_burn,
            keep_temp=keep_temp,
            loglevel=log_level
        )
        traverser.run(
            path_generator=path_generator,
            multi_chromosomes=True,  # opts.is_multi_chromosomes,
            force_circular=not linear_chr)
    except:
        typer.Exit()