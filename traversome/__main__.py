#! /usr/bin/env python

"""
Command-line Interface to traversome
"""
import shutil
import time
time_zero = time.time()
import os, sys, platform
from pathlib import Path
from enum import Enum
import typer
from typing import Union, Optional
from math import inf
from shutil import rmtree as rmdir
from traversome import __version__
from traversome.utils import setup_logger, parse_g_select
import yaml


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


# ================== logging options to yaml ==================
def serialize_option_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, slice):
        return {
            "start": value.start,
            "stop": value.stop,
            "step": value.step,
        }
    return value


def write_options_to_yaml(options: dict, yaml_file: str):
    """
    Write the actual options for the current command invocation.
    :param options: The current command options
    :param yaml_file: The path to the output YAML file
    """
    options = {
        key: serialize_option_value(value)
        for key, value in options.items()
    }
    with open(yaml_file, 'w') as f:
        yaml.dump(options, f, 
                  default_flow_style=False)  # add to produce human-readable YAML
    # typer.echo(f"Wrote options to {yaml_file}")
# ================== logging options to yaml ends ==================


class StartStrategy(str, Enum):
    random = "random"
    numerate = "numerate"
    # weighting ?


class ModelSelectionMode(str, Enum):
    AIC = "AIC"
    BIC = "BIC"
    # Single = "single"


class ChTopology(str, Enum):
    circular = "circular"
    unconstrained = "all"
    none = None  # not specified


class ChComposition(str, Enum):
    single = "single"
    unconstrained = "all"
    none = None  # not specified


# typer does not support mutually exclusive options yet, use Enum to work around
# ref: https://github.com/tiangolo/typer/issues/140
class Previous(str, Enum):
    terminate = "terminate"
    resume = "resume"
    overwrite = "overwrite"



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
def trim(
    graph_file: Path = typer.Option(
        ..., "-g", "--graph",
        help="GFA/FASTG format Graph file",
        exists=True, resolve_path=True),
    output_graph: Path = typer.Option(
        ..., "-o", "--output",
        help="Output graph file",
        exists=False, resolve_path=True),
    trim_overlaps: bool = typer.Option(
        False, "--trim-overlaps",  "--trim-o",
        help="Trim redundant overlap information in the graph. "
             "If disabled, the overlaps will be kept in the output graph. "),
    graph_component_selection: str = typer.Option(
        "0", "--graph-selection", "--graph-s",
        help="Use this to select certain connected components in the graph for assembly. "
             "First, the weight of each connected component will be calculated as \\sum_{i=1}^{N}length_i*depth_i, "
             "where N is the contigs in that component. Then, the components will be sorted in a decreasing order. "
             "1) If the input is an integer or a slice, this will trigger the selection of specific component "
             "by the decreasing order, e.g. 0 will keep the first component; 0,4 will keep the first four components; "
             "2) If the input is a float in the range of (0, 1), this will trigger the selection using "
             "the accumulated weight ratio cutoff, "
             "above which, the remaining components will be discarded. "
             "A cutoff of 1.0 means keeping all components in the graph, "
             "while a value close to 0 means only keep the connected "
             "component with the largest weight. "),
    log_level: LogLevel = typer.Option(
        LogLevel.INFO, "--loglevel", help="Logging level. Use DEBUG for more, ERROR for less."),
    ):
    """trim the graph by either redundant overlaps or specific connected components."""
    graph_component_selection = parse_g_select(graph_component_selection)
    from loguru import logger
    initialize(
        output_dir=None,  # no log file
        loglevel=log_level,
        previous=None)
    setup_logger(loglevel=log_level, timed=True)
    debug = log_level in (LogLevel.DEBUG, LogLevel.TRACE)
    from traversome.Assembly import Assembly
    assembly_obj = Assembly(str(graph_file))
    modified1 = False
    if isinstance(graph_component_selection, int) or isinstance(graph_component_selection, slice):
        modified1 = assembly_obj.reduce_graph_by_weight(component_ids=graph_component_selection)
    elif isinstance(graph_component_selection, float):
        modified1 = assembly_obj.reduce_graph_by_weight(cutoff_to_total=graph_component_selection)
    if not modified1:
        logger.info("No component to be trimmed.")
    modified2 = False
    if trim_overlaps:
        logger.info("Trimming overlaps in the graph..")
        modified2 = assembly_obj.trim_overlaps(debug=debug)
        if not modified2:
            logger.info("No overlaps to be trimmed.")
    if not modified1 and not modified2:
        logger.info("No graph generated due to no trim operation.")
    else:
        logger.info(f"Writing trimmed graph to {str(output_graph)}..")
        assembly_obj.write_to_gfa(str(output_graph))
        

@app.command()
def thorough(
    graph_file: Path = typer.Option(
        ..., "-g", "--graph",
        help="GFA/FASTG format Graph file",
        exists=True, resolve_path=True),
    # TODO gzip
    reads_file: Path = typer.Option(
        None, "-f", "--fastq",
        help="FASTQ format long read file. "
             "Conflict with flag `-a`.",
        resolve_path=True,
        ),
    alignment_file: Path = typer.Option(
        None, "-a", "--alignment",
        help="GAF format alignment file. "
             "Gzip-compressed file will be recognized by the postfix. "
             "Conflict with flag `-f`.",
        resolve_path=True,
        ),
    output_dir: Path = typer.Option(
        './', "-o", "--output",
        help="Output directory",
        exists=False, resolve_path=True),
    # identifiable_by_unique_rp: bool = typer.Option(
    #     False, "--strict",
    #     help="Read path and their occurrence counts will be used to identify the composition of variants. "
    #          "Choose --strict to use only unique read paths to identify the variants; " \
    #          "variants with the same read path but different counts will be treated as unidentifiable. "
    #          "This can be helpful by merging ambiguous variant combinations and achieve better bootstrap support, "
    #          "but with lower resolution. "),
    var_fixed: Path = typer.Option(
        None, "--vf", "--variant-fixed",
        help=r"A file containing user assigned variant paths "
             r"that will remain after model selection process. "
             r"Each line contains a variant path in Bandage path format, "
             r"e.g. '53+,45-,33-(circular)' or in GAF alignment format, e.g. '>53<45<33'"
        ),
    var_candidate: Path = typer.Option(
        None, "--vc", "--variant-candidate",
        help=r"A file containing user proposed candidate variant paths "
             r"that will be compared during model selection process. "
             r"Each line contains a variant path in Bandage path format, "
             r"e.g. '53+,45-,33-(circular)' or in GAF alignment format, e.g. '>53<45<33'"
    ),
    # Currently there is no other searching scheme implemented
    # var_gen_scheme: VarGen = typer.Option(
    #     VarGen.Heuristic, "-G",
    #     help="Variant generating scheme: H (Heuristic)"),
    graph_based_multiplicity_wiggle: float = typer.Option(
        0.25, "--g-wiggle",
        help="Use connection information to allow more possible multiplicity results, "
             "therefore more candidate variants. "
             "Input a float value between 0 and 0.5; a larger number means more wiggle. ",
        min=0, max=0.5, # 0.0 means no wiggle
    ),
    penalty_for_length: float = typer.Option(
        1e-5, "--len-penalty",
        help="penalty for the size of the variant, "
             "used to avoid overlooping the same contig too many times over depth-heterogeneous graph."
    ),
    fix_shallowest_contig: int = typer.Option(
        1, "--fix-shallow",
        help="In a connected component, "
             "1. fix all contigs that has a similar (<1.15 fold) coverage to the shallowest one to be single-copy (default); "
             "2. fix the shallowest contig to be single-copy in the proposed variants; "
             "3. fix all single-connected contigs that has a similar (<1.15 fold) coverage "
             "   to the shallowest single-connected one to be single-copy; "
             "4. fix the shallowest single-connected contig (one in and one out) to be single-copy; "
             "0. Disable this fixation; ",
        min=0, max=4,
        ),
    # search_start_scheme: StartStrategy = typer.Option(
    #     StartStrategy.random, "--sss",  #  "--search-start-strategy",
    #     help="random (randomly chosen from read aligned paths, default)\n"
    #          "numerate (sequentially numerate from read aligned paths)"),
    use_alignment_cov: bool = typer.Option(
        False, "--use-align-cov",
        help="Use alignment as the contig coverage. "  # for variant heuristic proposal step
             "Testing mode. "
        ),
    decompose_circular_unit: bool = typer.Option(
        True, "--no-decompose",
        help="By default, any proposed circular candidate path that exceed the longest read path will enter into "
             "the decomposition trial process, which is trying to find if there are identical/near-identical units "
             "in the circular candidate path and decompose it into smaller circular candidate paths if possible. "
             "This is a design for most biological "
             "Choose to disable this process."
    ),
    search_decay_factor: float = typer.Option(
        20., "--sdf",  # "--search-decay-factor",
        help="[1, INF] Search decay factor. "
             "The chance reduces by which, "
             "a read with less overlap with current path will be used to extend current path. "
             "A large value leads to strictly following read paths with the largest overlap. "),
    min_valid_search: int = typer.Option(
        500, "-n", "--min-val-search",
        help="Minimum num of valid traversals for heuristic search."),
    max_valid_search: int = typer.Option(
        10000, "-N", "--max-val-search",
        help="Maximum num of valid traversals for heuristic search."),
    max_num_traversals: int = typer.Option(
        30000, "-M", "--max-traversals",
        help="Hard bound for total number of searches (traversals). "),
    max_uniq_traversal: int = typer.Option(
        200, "--max-unique",  # "--max-unique-search",
        help="Hard bound for number of unique valid traversals for heuristic search. "
             "Too many unique candidates will cost computational resources and usually indicate bad dataset. "),
    max_uncover_ratio: float = typer.Option(
        0.001, "--max-uncover",
        help="Tolerance of uncovered read paths weighted by alignment counts during variants assessment. ",
        min=0, max=0.5),
    criterion: ModelSelectionMode = typer.Option(
        ModelSelectionMode.BIC, "-F", "--func",
        help="AIC (reverse model selection using stepwise AIC)\n"
             "BIC (reverse model selection using stepwise BIC, default)"),
    # disabled for now
    augmented_bootstrap: bool = typer.Option(
        False, "--augmented-bootstrap", "--abs",
        help="Use augmented bootstrap (keep the original records for each replicate) to estimate the support of the variants. "),
    bootstrap: int = typer.Option(
        100, "--bs", "--bootstrap",
        help="The number of repeats used to perform bootstrap analysis. "),
    bs_threshold: float = typer.Option(
        0.95, "--bs-threshold",  # "--bootstrap-threshold",
        help="Support below which will be treated as unsupported. "
             "Early interruption of the bootstrap will be triggered if solutions are not consistent across bootstraps. "
    ),
    # jackknife: int = typer.Option(
    #     0, "--jk", "--jackknife",
    #     help="The number of repeats used to perform jackknife analysis. "),
    # deprecated
    # fast_bootstrap: int = typer.Option(
    #     0, "--fbs", "--fast-bootstrap",
    #     help="Perform fast pseudo bootstrap analysis. "),
    random_seed: int = typer.Option(
        12345, "--rs", "--random-seed", help="Random seed"),
    topology: ChTopology = typer.Option(
        ..., "--topo", "--topology",
        help="Chromosome variant topology: "
             "circular (constrained to be circular);"
             "all (unconstrained). ",
        prompt_required=True),
    composition: ChComposition = typer.Option(
        ChComposition.unconstrained, "--v-comp", "--composition",
        help="Chromosome variant composition: "
             "single (each single form covers all contigs; only isomeric variants) / "
             "all (unconstrained, single or multi-chromosomes, RECOMMENDED for most cases)",
        prompt_required=True),
    size_ratio: float = typer.Option(
        0.0, "--v-len", "--length-cutoff",
        help="Chromosome variant with less than the ratio * MAX(lengths_of_all_variants) will be discarded. ",
        min=0, max=1.),
    out_seq_threshold: float = typer.Option(
        0.00, "--seq-out",
        help="Threshold for sequence output. Use 2 to disable.",
        min=0, max=2),
    graph_aligner_params: str = typer.Option(
        "--precise-clipping 0.95", "--graph-aligner-params",
        help="Extra parameters passed to GraphAligner quoted with '' "),
    # quality_control_alignment_cov: float = typer.Option(
    #     300., "--qc-depth",
    #     help="The alignment depth as the goal to search for top-quality alignment records. "
    #     ),
    min_read_identity_cutoff: float = typer.Option(
        0.992, "--min-read-id",
        help="Threshold for alignment identity, read with below which the alignment will be discarded. "
             "The default value is for hifi reads. Try 0.95~0.99 for other types of reads and graph combinations.",
        max=1),
    min_record_identity_cutoff: float = typer.Option(
        0.99, "--min-record-id",
        help="Threshold for alignment identity, a record of a read with below which the alignment will be discarded. "
             "The default value is for hifi reads. Try 0.95~0.99 for other types of reads and graph combinations.",
        max=1),
    min_alignment_len_cutoff: int = typer.Option(
        5000, "--min-align-len",
        help="Threshold for the continuous alignment length of a read, below which the alignment will be discarded. "),
    min_alignment_counts: int = typer.Option(
        -1, "--min-align-counts",
        help="Threshold for counts per path, below which the alignment(s) of that path will be discarded. "
             "Automatic selection (-1) does not guarantee the best performance - good bootstrap support. "
             "This is a very important parameter to set, to filter out the low-quality alignments. "
             "However, using this parameter may lead to the loss of resolution. "
             "Default: auto(-1)"
    ),
    graph_component_selection: str = typer.Option(
        "0", "--graph-selection",
        help="Use this if your graph is not manually curated into the target complete graph. "
             "First, the weight of each connected component will be calculated as \\sum_{i=1}^{N}length_i*depth_i, "
             "where N is the contigs in that component. Then, the components will be sorted in a decreasing order. "
             "1) If the input is an integer or a slice, this will trigger the selection of specific component "
             "by the decreasing order, e.g. 0 will keep the first component; 0,4 will keep the first four components; "
             "2) If the input is a float in the range of (0, 1), this will trigger the selection using "
             "the accumulated weight ratio cutoff, "
             "above which, the remaining components will be discarded. "
             "A cutoff of 1.0 means keeping all components in the graph, "
             "while a value close to 0 means only keep the connected "
             "component with the largest weight. "),
    purge_shallow_contigs: float = typer.Option(
        0.001, "--graph-purge",
        help="Use this if your graph is not manually curated into the target complete graph. "
             "After applying '--graph-selection', an average_depth will be estimated and "
             "any contigs that has shallower depth than FLOAT * average_depth will be discarded. "
             "The higher the value FLOAT is, the more contigs will be discarded. "),
    prune_terminal_contigs: bool = typer.Option(
        False, "--prune-terminal-contigs",
        help="Choose to prune terminal contigs from the assembly graph. "
             "This will remove contigs that have no or only one neighbor. "
             "This is useful when the graph is not manually curated into the target complete graph. "
             "This will be automatically turned on if '--topo circular' is set. "
             "Raw reads may be required if graph is about to change. "),
    keep_graph_redundancy: bool = typer.Option(
        True, "--merge-possible-contigs",
        help="Choose to merge neighboring contigs when possible "
             "and may potentially decrease the computational burden. "
             "Raw reads may be required if graph is about to change. "
        ),
    keep_unaligned_contigs: bool = typer.Option(
        False, "--keep-unaligned",
        help="Choose to keep unaligned contigs without been pruning from the assembly graph. "
    ),
    check_conflicts: bool = typer.Option(
        False, "--check-conflicts",
        help="[BETA] By default, ignore conflicts between the graph and the alignment file. "
             "Choose to detect conflict and potentially (see '--min-consensus-reads') to add edges to the graph. "
    ),
    add_conflict_edges: int = typer.Option(
        5, "--min-consensus-reads",
        help="[BETA] Followed by an integer indicating the minimum number of consensus reads required to add an edge. "
             "This will modify the graph by add edges representing the significantly conflicting reads against the graph, "
             "as detected by Traversome. "
             "Number of conflicts per window below this value will be then ignored. "
             "Disabled if set to 0."),
    gmm_max_std: float = typer.Option(
        50., "--gmm-max-std",
        help="[BETA] Maximum standard deviation value for each modal in the Gaussian Mixture Model. "
             "The larger the value is, the less break points will be generated by the same conflict signals. "
             "."),
    num_processes: int = typer.Option(
        1, "-p", "--processes",
        help="Num of processes. Multiprocessing will lead to non-identical result with the same random seed."),
    # genetic algorithm options
    # disabled for now due to the inefficient performance
    # ga_pop_size: int = typer.Option(
    #     100, "--ga-pop-size",
    #     help="Population size for the genetic algorithm. "
    #          "The larger the value is, the more likely the best/equally-best result will be achieved, "
    #          "but also the longer the computation time will be."),
    # ga_max_gen: int = typer.Option(
    #     200, "--ga-max-gen",
    #     help="Maximum number of generations for the genetic algorithm. "),
    # ga_mut_prob: float = typer.Option(
    #     0.1, "--ga-mut-prob",
    #     help="Mutation probability for the genetic algorithm. "),
    # ga_patience: int = typer.Option(
    #     10, "--ga-patience",
    #     help="Patience for the genetic algorithm. "
    #             "If the best solution does not change for this many generations, "
    #             "the algorithm will stop early. "),
    # reverse model selection options
    rms_max_queue_size: int = typer.Option(None, "--rms-max-queue-size",
        help="Maximum queue size for reverse model selection. "),
    rms_max_end_hits: int = typer.Option(None, "--rms-max-end-hits",
        help="Maximum end hits (one of the stop criteria) for reverse model selection."
             "Stop the searching if it hit the end of a search branch for this many times. "),
    rms_max_unchanged: int = typer.Option(None, "--rms-max-unchanged",
        help="The patience level of reverse model selection (one of the stop criteria)."
             "Stop the searching if the best model has not changed for this many iterations. "),
    # disable the mcmc part for now
    # n_generations: int = typer.Option(
    #     0, "--mcmc",
    #     help="MCMC generations. Set '--mcmc 0' to disable mcmc process. NOT RECOMMENDED as of now. "),
    # n_burn: int = typer.Option(1000, "--burn", help="MCMC Burn-in"),
    # mc_bracket_depth: Optional[int] = typer.Option(
    #     10000, "--mc-bd",  # "--mc-bracket-depth",
    #     help="Bracket depth for gcc during MCMC sampling. "),
    # TODO: to be fully implemented
    prev_run: Previous = typer.Option(
        Previous.terminate, "--previous",
        help="terminate: exit with error if previous result exists\n"
             "resume: continue on latest checkpoint if previous result exists\n"
             "overwrite: remove previous result if exists."),
    keep_temp: bool = typer.Option(
        False, "--keep-temp",
        help="Keep intermediate graph, alignment, and candidate variant files. Deleted by default. "),
    log_level: LogLevel = typer.Option(
        LogLevel.INFO, "--loglevel", help="Logging level. Use DEBUG for more, ERROR for less."),
    ):
    """
    Conduct thorough analysis from assembly graph and reads/read-graph alignment.
    Examples:
    traversome thorough -g graph.gfa -a align.gaf -o output_dir --topo circular
    """
    assert str(graph_file).endswith(".gfa") or str(graph_file).endswith(".fastg"), "Invalid graph format provided!"
    # use_gfa_annotation_lines = False
    if reads_file and alignment_file:
        sys.stderr.write("Flags `-f` and `-a` are mutually exclusive!")
        sys.exit()
    elif not reads_file and not alignment_file:
        # use_gfa_annotation_lines = True
        sys.stderr.write("Either the sequence file (via `-f`) or the graph alignment (via `-a`) should be provided!")
        sys.exit()
    elif reads_file and not reads_file.exists():
        raise FileNotFoundError(reads_file)
    elif alignment_file and not alignment_file.exists():
        raise FileNotFoundError(alignment_file)
    
    quality_control_alignment_cov = 300.  # disable target-depth-based quality control for now
    jackknife = 0  # disable jackknife for now
    if bootstrap and jackknife:
        sys.stderr.write("Flags `--bootstrap` and `--jackknife` are mutually exclusive!")
    # data filtering
    if min_read_identity_cutoff == -1:
        min_read_identity_cutoff = "auto"
    else:
        assert min_read_identity_cutoff > 0.8
    assert min_record_identity_cutoff > 0.8
    if min_alignment_len_cutoff == -1:
        min_alignment_len_cutoff = "auto"
    else:
        assert min_alignment_len_cutoff >= 1000
    if min_alignment_counts == -1:
        min_alignment_counts = "auto"
    else:
        assert min_alignment_counts > 0
    if topology == ChTopology.circular:
        prune_terminal_contigs = True
    graph_component_selection = parse_g_select(graph_component_selection)
    run_options = locals().copy()

    from loguru import logger
    initialize(
        output_dir=output_dir,
        loglevel=log_level,
        previous=prev_run)
    # write options to yaml file
    write_options_to_yaml(run_options, output_dir.joinpath("options.yaml"))
    # set theano cache directory
    theano_cache_dir = output_dir.joinpath("theano.cache")
    symengine_cache_dir = output_dir.joinpath("symengine.cache")
    try:
        # disable caching can solve the temp file issue, but largely reduce performance
        # import theano
        # theano.config.cxx = ""
        # set the caching directory to be inside each output directory
        os.environ["THEANO_FLAGS"] = "base_compiledir={}".format(str(theano_cache_dir))
        os.environ["AESARA_FLAGS"] = "base_compiledir={}".format(str(theano_cache_dir))
        # TODO to fix pytensor issue
        os.environ["PYTENSOR_FLAGS"] = "base_compiledir={}".format(str(theano_cache_dir))
        # os.environ["TMPDIR"] = str(output_dir.joinpath("tmp.mp"))
        os.environ["SYMENGINE_CACHE_DIR"] = str(symengine_cache_dir)

        # assert var_gen_scheme != "U", "User-provided is under developing, please use heuristic instead!"
        setup_logger(loglevel=log_level, timed=True, log_file=os.path.join(output_dir, "traversome.log.txt"))
        if max_valid_search < 1:
            logger.info("Heuristic search disabled. ")
            if bool(var_candidate) or bool(var_fixed):
                if bool(var_fixed) and not var_fixed.is_file():
                    raise FileNotFoundError(f"Invalid input file: {var_fixed}")
                if bool(var_candidate) and not var_candidate.is_file():
                    raise FileNotFoundError(f"Invalid input file: {var_candidate}")
                logger.info("Only user input variants will be included in downstream analysis. ")
            else:
                raise ValueError("User input variant(s) (--vf/--vc) is required when heuristic search was disabled! ")
        else:
            if max_valid_search < min_valid_search:
                logger.warning("Input minimum num of valid traversals is larger than the maximum! Resetting..")
                min_valid_search = max_valid_search
            logger.info(f"Minimum num of valid traversals: {min_valid_search}")
            logger.info(f"Maximum num of valid traversals: {max_valid_search}")
        force_circular = topology == ChTopology.circular
        logger.info(f"constraint chromosome topology to be circular: {force_circular}")
        single_chr = composition == ChComposition.single
        logger.info(f"constraint chromosome composition to be single: {single_chr} \n")

        # if use_gfa_annotation_lines:
        #     logger.info("Graph alignment source: GFA file")
        # elif reads_file:
        #     logger.info("Graph alignment source: To be made")
        # elif alignment_file:
        #     logger.info("Graph alignment source: GAF alignment file")

        # assert max_valid_search >= min_valid_search, ""
            
        # TODO: use json file to record parameters
        from traversome.traversome import Traversome
        traverser = Traversome(
            graph=str(graph_file),
            alignment=str(alignment_file) if alignment_file else alignment_file,
            reads_file=str(reads_file) if reads_file else reads_file,  # TODO allow multiple files
            var_fixed=str(var_fixed) if var_fixed else var_fixed,
            var_candidate=str(var_candidate) if var_candidate else var_candidate,
            outdir=str(output_dir),
            # identifiable_by_unique_rp=identifiable_by_unique_rp,
            model_criterion=criterion,
            augmented_bootstrap=augmented_bootstrap,
            bootstrap=bootstrap,
            bs_threshold=bs_threshold,
            jackknife=jackknife,
            # fast_bootstrap=fast_bootstrap,  # deprecated
            out_prob_threshold=out_seq_threshold,
            graph_based_multiplicity_wiggle=graph_based_multiplicity_wiggle,
            penalty_for_size=penalty_for_length,
            fix_shallowest_contig=fix_shallowest_contig,
            # search_start_scheme=search_start_scheme,
            use_alignment_cov=use_alignment_cov,
            decompose_circular_unit=decompose_circular_unit,
            search_decay_factor=search_decay_factor,
            min_valid_search=min_valid_search,
            max_valid_search=max_valid_search,
            max_num_traversals=max_num_traversals,
            max_uniq_traversal=max_uniq_traversal,
            max_uncover_ratio=max_uncover_ratio,
            # ga_pop_size=ga_pop_size,
            # ga_max_gen=ga_max_gen,
            # ga_mut_prob=ga_mut_prob,
            # ga_patience=ga_patience,
            rms_max_queue_size=rms_max_queue_size,
            rms_max_end_hits=rms_max_end_hits,
            rms_max_unchanged=rms_max_unchanged,
            num_processes=num_processes,
            uni_chromosome=single_chr,
            force_circular=force_circular,
            size_ratio=size_ratio,
            n_generations=0,  # n_generations,
            n_burn=1000,      # n_burn,
            random_seed=random_seed,
            # keep_temp=False,
            loglevel=log_level,
            graph_aligner_params=graph_aligner_params,
            min_read_identity_cutoff=min_read_identity_cutoff,
            min_record_identity_cutoff=min_record_identity_cutoff,
            min_alignment_len_cutoff=min_alignment_len_cutoff,
            quality_control_alignment_cov=quality_control_alignment_cov,
            min_alignment_counts=min_alignment_counts,
            graph_component_selection=graph_component_selection,
            purge_shallow_contigs=purge_shallow_contigs,
            prune_terminal_contigs=prune_terminal_contigs,
            keep_graph_redundancy=keep_graph_redundancy,
            keep_unaligned_contigs=keep_unaligned_contigs,
            check_conflicts=check_conflicts,
            add_conflict_edges=add_conflict_edges,
            gmm_max_std=gmm_max_std,
            resume=prev_run == "resume",
            mc_bracket_depth=10000,  # mc_bracket_depth,
            # use_gfa_alignment=use_gfa_annotation_lines,
            keep_temp=keep_temp,
        )
        traverser.run()
        del traverser
    # except KeyboardInterrupt:
    #     logger.error("Interrupted by user.")
    #     logger.info("Total cost %.4f" % (time.time() - time_zero))
    #     raise SystemExit(1)
    except SystemExit as e:
        logger.info("Total cost %.4f" % (time.time() - time_zero))
        raise e
    except:
        logger.exception("")
        logger.info("Total cost %.4f" % (time.time() - time_zero))
        raise SystemExit(1)
    finally:
        logger.info("Total cost %.4f" % (time.time() - time_zero))
        #
    
    # remove theano cached files if not keeping temporary files
    # TODO WARNING (theano.link.c.cmodule) and FileNotFoundError to be fixed
    # if not keep_temp:
    #     try:
    #         import theano.gof.compiledir
    #         theano.gof.compiledir.compiledir_purge()
    #     except ModuleNotFoundError:
    #         import aesara.compile.compiledir
    #         from aesara.link.c.basic import get_module_cache
    #         # aesara.compile.compiledir.cleanup()
    #         # cache = get_module_cache(init_args=dict(do_refresh=False))
    #         # cache.clear_old()
    #         aesara.compile.compiledir.basecompiledir_purge()
    #     theano.gof.cc.cmodule.clear_compiledir()
    #     for lock_f in theano_cache_dir.glob("*/.lock"):
    #         os.remove(lock_f)
    #     shutil.rmtree(theano_cache_dir, ignore_errors=True)
    #     os.system("rm -rf " + str(theano_cache_dir))


def initialize(output_dir, loglevel, previous):
    """
    clear files if overwrite
    log head and running environment
    """
    if output_dir is None:  # skip generating log file
        from loguru import logger
        # avoid repeating RUNNING_HEAD in the screen output by typer.secho
        setup_logger(loglevel=loglevel, timed=False, log_file=None, screen_out=None)
        logger.info(RUNNING_HEAD)
        setup_logger(loglevel=loglevel, timed=False, log_file=None)
        logger.info(RUNNING_ENV_INFO)
    else:
        os.makedirs(str(output_dir), exist_ok=previous in ("overwrite", "resume"))
        if previous == "overwrite" and os.path.isdir(output_dir):
            # rmdir(output_dir.joinpath("tmp.mp"), ignore_errors=True)
            rmdir(output_dir.joinpath("tmp.candidates"), ignore_errors=True)
            rmdir(output_dir.joinpath("theano.cache"), ignore_errors=True)
            for exist_f in output_dir.glob("*.*"):
                os.remove(exist_f)
        # os.makedirs(str(output_dir.joinpath("tmp.mp")), exist_ok=previous in ("overwrite", "resume"))
        logfile = os.path.join(output_dir, "traversome.log.txt")
        from loguru import logger
        # avoid repeating RUNNING_HEAD in the screen output by typer.secho
        setup_logger(loglevel=loglevel, timed=False, log_file=logfile, screen_out=None)
        logger.info(RUNNING_HEAD)
        setup_logger(loglevel=loglevel, timed=False, log_file=logfile)
        logger.info(RUNNING_ENV_INFO)


@app.command()
def simulate(
    graph_f: Path = typer.Option(
        ..., "-g", "--graph",
        help="GFA/FASTG format Graph file",
        exists=True, resolve_path=True),
    variant_file: Path = typer.Option(
        ..., "--vp", "--variant-path-file",
        help="A file containing variant paths "
             "that will be compared during model selection process. "
             "Each line contains a variant path in Bandage path format, "
             "e.g. 53+,45-,33+(circular) or in GAF alignment format, e.g. >53<45>33"),
    var_proportions: str = typer.Option(
        ..., "-p", "--prop",
        help="Proportions of the input variants split by comma and sum up to 1, e.g. -p 0.7,0.3"),
    len_distribution: str = typer.Option(
        ..., "-l", "--len",
        help="Followed by ont, pb, hifi or mean,std_dev (e.g. -l 15000,13000). "
             "The read lengths will be draw from a gamma distribution."),
    # TODO add sequencing depth
    # TODO add Mb, Gb, etc
    data_size: int = typer.Option(
        ..., "-s", "--data-size",
        help="Expected total number of bases. "),
    output_gaf: Path = typer.Option(
        None, "-a", "--output-ali",
        help="Output alignment file (*.gaf or *.gaf.gz)",
        exists=False, resolve_path=True),
    output_fasta: Path = typer.Option(
        None, "-f", "--output-fas",
        help="Output sequence file (*.fasta or *.fasta.gz)",
        exists=False, resolve_path=True),
    random_seed: int = typer.Option(
        12345, "--rs", "--random-seed",
        help="Random seed"),
    ):
    """
    Currently only simulate true alignment without errors
    """
    from traversome.Simulator import SimpleSimulator
    from traversome.Assembly import Assembly
    from traversome.utils import user_paths_reader
    var_props = [float(vp) for vp in var_proportions.split(",")]
    assert sum(var_props) == 1.
    length_distribution = len_distribution if len_distribution in {"ont", "pb", "hifi"} \
        else [int(float(l_)) for l_ in len_distribution.split(",")]
    assert output_gaf or output_fasta, "Select at least one type of output!"
    if output_gaf:
        logfile = str(output_gaf).replace(".gaf", "") + ".log"
    else:
        logfile = str(output_fasta).replace(".fasta", "") + ".log"
    from loguru import logger
    # avoid repeating RUNNING_HEAD in the screen output by typer.secho
    setup_logger(loglevel="INFO", timed=False, log_file=logfile, screen_out=None)
    logger.info(RUNNING_HEAD)
    setup_logger(loglevel="INFO", timed=False, log_file=logfile)
    logger.info(RUNNING_ENV_INFO)
    setup_logger(loglevel="INFO", timed=True, log_file=logfile)
    simulator = SimpleSimulator(
        graph_obj=Assembly(graph_file=str(graph_f)),
        variants=user_paths_reader(str(variant_file)),
        variant_proportions=var_props,
        length_distribution=length_distribution,
        data_size=data_size,
        out_gaf=str(output_gaf),
        out_fasta=str(output_fasta),
        random_seed=random_seed
        )
    simulator.run()
