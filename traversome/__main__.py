#! /usr/bin/env python

"""
Command-line Interface to traversome
"""
import time
time_zero = time.time()
import os, sys, platform
from pathlib import Path
from enum import Enum
import typer
from traversome import __version__

# add the -h option for showing help
CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
RUNNING_HEAD = f"traversome (v.{__version__}): genomic isomer frequency estimator"
RUNNING_ENV_INFO = \
    "\n" \
    "Python " + str(sys.version).replace("\n", " ") + "\n" + \
    "PLATFORM:  " + " ".join(platform.uname()) + "\n" + \
    "WORKDIR:   " + os.getcwd() + "\n" + \
    "COMMAND:   " + " ".join(["\"" + arg + "\"" if " " in arg else arg for arg in sys.argv]) + \
    "\n"


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
    typer.secho(RUNNING_HEAD, fg=typer.colors.MAGENTA, bold=True)


class PathGen(str, Enum):
    Heuristic = "H"
    Provided = "U"


class ModelSelectionMode(str, Enum):
    AIC = "aic"
    BIC = "bic"
    Single = "single"


class ChTopology(str, Enum):
    circular = "c"
    unconstrained = "u"


class ChComposition(str, Enum):
    single = "s"
    unconstrained = "u"


# deprecated for now
# @app.command()
# def ml(
#     graph_file: Path = typer.Option(
#         ..., "-g", "--graph",
#         help="GFA/FASTG format Graph file",
#         exists=True, resolve_path=True),
#     alignment_file: Path = typer.Option(
#         ..., "-a", "--alignment",
#         help="GAF format alignment file",
#         exists=True, resolve_path=True,
#     ),
#     output_dir: Path = typer.Option(
#         './', "-o", "--output",
#         help="Output directory",
#         exists=False, resolve_path=True),
#     path_generator: PathGen = typer.Option(
#         PathGen.Heuristic, "-P",
#         help="Path generator: H (Heuristic)/U (User-provided)"),
#     num_search: int = typer.Option(
#         1000, "-N", "--num-search",
#         help="Num of valid traversals for heuristic searching."),
#     num_processes: int = typer.Option(
#         1, "-p", "--processes",
#         help="Num of processes. "),
#     function: ModelSelectionMode = typer.Option(
#         ModelSelectionMode.AIC, "-F", "--func",
#         help="Function: aic (reverse model selection using stepwise AIC)\n"
#              "bic (reverse model selection using stepwise BIC)\n"
#              "single (conduct maximum likelihood estimation on the component-richest model without model selection)"),
#     random_seed: int = typer.Option(
#         12345, "--rs", "--random-seed", help="Random seed. "),
#     topology: ChTopology = typer.Option(
#         ChTopology.circular, "--topology",
#         help="Chromosomes topology: c (constrained to be circular)/ u (unconstrained). "),
#     composition: ChComposition = typer.Option(
#         ChComposition.unconstrained, "--composition",
#         help="Chromosomes composition: "
#              "s (single, each single form covers all compositions) / "
#              "u (unconstrained, single or multi-chromosomes. recommended)"),
#     out_seq_threshold: float = typer.Option(
#         0.0, "-S",
#         help="Threshold for sequence output",
#         min=0, max=1),
#     overwrite: bool = typer.Option(False, help="Remove previous result if exists."),
#     keep_temp: float = typer.Option(False, "--keep-temp", help="Keep temporary files for debug. "),
#     log_level: LogLevel = typer.Option(
#         LogLevel.INFO, "--loglevel", "--log-level", help="Logging level. Use DEBUG for more, ERROR for less."),
#     ):
#     """
#     Conduct Maximum Likelihood analysis for solving assembly graph
#     Examples:
#     traversome ml -g graph.gfa -a align.gaf -o .
#     """
#     from loguru import logger
#     initialize(
#         output_dir=output_dir,
#         loglevel=log_level,
#         overwrite=overwrite)
#     try:
#         assert path_generator != "U", "User-provided is under developing, please use heuristic instead!"
#         from traversome.traversome import Traversome
#         traverser = Traversome(
#             graph=str(graph_file),
#             alignment=str(alignment_file),
#             outdir=str(output_dir),
#             function=function,
#             out_prob_threshold=out_seq_threshold,
#             force_circular=topology == "c",
#             num_search=num_search,
#             num_processes=num_processes,
#             random_seed=random_seed,
#             keep_temp=keep_temp,
#             loglevel=log_level
#         )
#         traverser.run(
#             path_generator=path_generator,
#             hetero_chromosomes=composition == "u"  # opts.is_multi_chromosomes,
#             )
#     except:
#         logger.exception("")
#     logger.info("Total cost %.4f" % (time.time() - time_zero))


@app.command()
def thorough(
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
    num_search: int = typer.Option(1000, "-N", "--num-search", help="Num of valid traversals for heuristic searching."),
    criterion: ModelSelectionMode = typer.Option(
        ModelSelectionMode.AIC, "-F", "--func",
        help="aic (reverse model selection using stepwise AIC, default)\n"
             "bic (reverse model selection using stepwise BIC)"),
    random_seed: int = typer.Option(12345, "--rs", "--random-seed", help="Random seed"),
    topology: ChTopology = typer.Option(
        ChTopology.circular, "--topology",
        help="Chromosomes topology: c (constrained to be circular)/ u (unconstrained). "),
    composition: ChComposition = typer.Option(
        ChComposition.single, "--composition",
        help="Chromosomes composition: "
             "s (single, each single form covers all contigs, default) / "
             "u (unconstrained, single or multi-chromosomes)"),
    out_seq_threshold: float = typer.Option(
        0.0, "-S",
        help="Threshold for sequence output",
        min=0, max=1),
    num_processes: int = typer.Option(
        1, "-p", "--processes",
        help="Num of processes. "),
    n_generations: int = typer.Option(10000, "--mcmc", help="MCMC generations"),
    n_burn: int = typer.Option(1000, "--burn", help="MCMC Burn-in"),
    overwrite: bool = typer.Option(False, help="Remove previous result if exists."),
    keep_temp: bool = typer.Option(False, help="Keep temporary files"),
    log_level: LogLevel = typer.Option(
        LogLevel.INFO, "--loglevel", help="Logging level. Use DEBUG for more, ERROR for less."),
    ):
    """
    Conduct Bayesian MCMC analysis for solving assembly graph
    Examples:
    traversome mc -g graph.gfa -a align.gaf -o .
    """
    from loguru import logger
    initialize(
        output_dir=output_dir,
        loglevel=log_level,
        overwrite=overwrite)
    try:
        assert path_generator != "U", "User-provided is under developing, please use heuristic instead!"
        from traversome.traversome import Traversome
        traverser = Traversome(
            graph=str(graph_file),
            alignment=str(alignment_file),
            outdir=str(output_dir),
            model_criterion=criterion,
            out_prob_threshold=out_seq_threshold,
            num_search=num_search,
            num_processes=num_processes,
            force_circular=topology == "c",
            n_generations=n_generations,
            n_burn=n_burn,
            random_seed=random_seed,
            keep_temp=keep_temp,
            loglevel=log_level
        )
        traverser.run(
            path_generator=path_generator,
            hetero_chromosomes=composition == "u"
        )
    except:
        logger.exception("")
    logger.info("Total cost %.4f" % (time.time() - time_zero))


def initialize(output_dir, loglevel, overwrite):
    """
    clear files if overwrite
    log head and running environment
    """
    os.makedirs(str(output_dir), exist_ok=overwrite)
    if overwrite and os.path.isdir(output_dir):
        for exist_f in output_dir.glob("*.*"):
            os.remove(exist_f)
    logfile = os.path.join(output_dir, "traversome.log.txt")
    from loguru import logger
    setup_simple_logger(sink_list=[logfile], loglevel=loglevel)
    logger.info(RUNNING_HEAD)
    setup_simple_logger(sink_list=[logfile, sys.stdout], loglevel=loglevel)
    logger.info(RUNNING_ENV_INFO)


def setup_simple_logger(sink_list, loglevel="INFO"):
    """
    Configure Loguru to log to stdout and logfile.
    param: sink_list e.g. [sys.stdout, logfile]
    """
    # add stdout logger
    from loguru import logger
    simple_config = {
        "handlers": [
            {
                "sink": sink_obj,
                "format": "<level>{message}</level>",
                "level": loglevel,
                }
            for sink_obj in sink_list]
    }
    logger.configure(**simple_config)
    logger.enable("traversome")
