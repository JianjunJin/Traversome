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
from typing import Union
from math import inf
from traversome import __version__

# add the -h option for showing help
CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
RUNNING_HEAD = f"traversome (v.{__version__}): genomic variant frequency estimator"
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


class VarGen(str, Enum):
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
#     var_gen_scheme: VarGen = typer.Option(
#         VarGen.Heuristic, "-P",
#         help="Path generator: H (Heuristic)/U (User-provided)"),
#     min_valid_search: int = typer.Option(
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
#         assert var_gen_scheme != "U", "User-provided is under developing, please use heuristic instead!"
#         from traversome.traversome import Traversome
#         traverser = Traversome(
#             graph=str(graph_file),
#             alignment=str(alignment_file),
#             outdir=str(output_dir),
#             function=function,
#             out_prob_threshold=out_seq_threshold,
#             force_circular=topology == "c",
#             min_valid_search=min_valid_search,
#             num_processes=num_processes,
#             random_seed=random_seed,
#             keep_temp=keep_temp,
#             loglevel=log_level
#         )
#         traverser.run(
#             var_gen_scheme=var_gen_scheme,
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
    var_gen_scheme: VarGen = typer.Option(
        VarGen.Heuristic, "-G",
        help="Variant generating scheme: H (Heuristic)/U (User-provided)"),
    min_valid_search: int = typer.Option(
        1000, "-n", "--min-val-search",
        help="Minimum num of valid traversals for heuristic searching."),
    max_valid_search: int = typer.Option(
        100000, "-N", "--max-val-search",
        help="Maximum num of valid traversals for heuristic searching."),
    criterion: ModelSelectionMode = typer.Option(
        ModelSelectionMode.AIC, "-F", "--func",
        help="aic (reverse model selection using stepwise AIC, default)\n"
             "bic (reverse model selection using stepwise BIC)"),
    random_seed: int = typer.Option(12345, "--rs", "--random-seed", help="Random seed"),
    topology: ChTopology = typer.Option(
        ChTopology.circular, "--topology",
        help="Chromosomes topology: "
             "c (constrained to be circular);"
             "u (unconstrained). "),
    composition: ChComposition = typer.Option(
        ChComposition.unconstrained, "--composition",
        help="Chromosomes composition: "
             "s (single, each single form covers all contigs, default) / "
             "u (unconstrained, single or multi-chromosomes)"),
    out_seq_threshold: float = typer.Option(
        0.0, "-S",
        help="Threshold for sequence output",
        min=0, max=1),
    min_alignment_identity_cutoff: float = typer.Option(
        0.8, "--min-align-id",
        help="Threshold for alignment identity, below which the alignment with be discarded. ",
        min=0, max=1),
    min_alignment_len_cutoff: int = typer.Option(
        100, "--min-align-len",
        help="Threshold for alignment length, below which the alignment with be discarded. ",
        min=100),
    graph_component_selection: str = typer.Option(
        "0", "--graph-selection",
        help="Use this if your graph is not manually curated into the target complete graph. "
             "First, he weight of each connected component will be calculated as \sum_{i=1}^{N}length_i*depth_i, "
             "where N is the contigs in that component. Then, the components will be sorted in a decreasing order. "
             "1) If the input is an integer or a slice, this will trigger the selection of specific component "
             "by the decreasing order, e.g. 0 will keep the first component; 0,4 will keep the first four components; "
             "2) If the input is a float in the range of (0, 1), this will trigger the selection using "
             "the accumulated weight ratio cutoff, "
             "above which, the remaining components will be discarded. "
             "A cutoff of 1 means keeping all components in the graph, "
             "while a value close to 0 means only keep the connected "
             "component with the largest weight. "),
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
    traversome thorough -g graph.gfa -a align.gaf -o .
    """
    from loguru import logger
    initialize(
        output_dir=output_dir,
        loglevel=log_level,
        overwrite=overwrite)
    try:
        assert var_gen_scheme != "U", "User-provided is under developing, please use heuristic instead!"
        assert max_valid_search >= min_valid_search, ""
        from traversome.traversome import Traversome
        if graph_component_selection.isdigit():
            graph_component_selection = int(graph_component_selection)
        elif "." in graph_component_selection:
            graph_component_selection = float(graph_component_selection)
        else:
            try:
                graph_component_selection = slice(*eval(graph_component_selection))
            except (SyntaxError, TypeError):
                raise TypeError(str(graph_component_selection) + " is invalid for --graph-selection!")
        traverser = Traversome(
            graph=str(graph_file),
            alignment=str(alignment_file),
            outdir=str(output_dir),
            model_criterion=criterion,
            out_prob_threshold=out_seq_threshold,
            min_valid_search=min_valid_search,
            max_valid_search=max_valid_search,
            num_processes=num_processes,
            force_circular=topology == "c",
            n_generations=n_generations,
            n_burn=n_burn,
            random_seed=random_seed,
            keep_temp=keep_temp,
            loglevel=log_level,
            min_alignment_identity_cutoff=min_alignment_identity_cutoff,
            min_alignment_len_cutoff=min_alignment_len_cutoff,
            graph_component_selection=graph_component_selection,
        )
        traverser.run(
            path_gen_scheme=var_gen_scheme,
            hetero_chromosomes=composition == "u"
        )
    except SystemExit:
        pass
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
