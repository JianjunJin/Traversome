#!/usr/bin/env python

"""
Top-level class for running the CLI
"""

import os
import sys
import random
import time

from loguru import logger
from copy import deepcopy
from pathlib import Path as fpath
from collections import OrderedDict
from traversome.Assembly import Assembly
from traversome.PanGenome import PanGenome
from traversome.GraphAlignRecords import GraphAlignRecords
from traversome.GraphAlignConflicts import GraphAlignConflicts
import dill
from traversome.utils import \
    SubPathInfo, Criterion, VariantSubPathsGenerator, executable, run_graph_aligner, user_paths_reader, setup_logger, \
    path_to_gaf_str, Bins, BinInfo, optimize_min_adj, run_dill_encoded, summarize_read_lengths
from traversome.ModelFitMaxLike import ModelFitMaxLike
from traversome.VariantGenerator import VariantGenerator
from traversome.ModelGenerator import PathMultinomialModel
from typing import OrderedDict as typingODict
from typing import Set, Union
from multiprocessing import Manager, Pool, Process
import traceback
import gc
import math
import numpy as np

# import time


class Traversome(object):
    """
    """

    def __init__(
            self,
            graph,
            alignment,
            reads_file,
            outdir,
            identifiable_by_unique_rp=False,
            var_fixed=None,
            var_candidate=None,
            num_processes=1,
            force_circular=True,
            uni_chromosome=False,
            out_prob_threshold=0.001,
            keep_temp=False,
            random_seed=12345,
            loglevel="INFO",
            resume=False,
            min_alignment_len_cutoff=5000,
            min_read_identity_cutoff=0.992,
            min_record_identity_cutoff=0.99,
            min_alignment_counts="auto",
            **kwargs):
        # store input files and params
        self.graph_file = graph
        self.graph_format = graph.split(".")[-1]  # already limited the types
        # if kwargs.get("use_gfa_alignment", False):
        #     self.alignment_file = graph
        # else:
        #     self.alignment_file = alignment
        self.alignment_file = alignment
        self.reads_file = reads_file
        self.outdir = outdir
        self.identifiable_by_unique_rp = identifiable_by_unique_rp
        self.var_fixed_f = var_fixed
        self.var_candidate_f = var_candidate
        self.num_processes = num_processes
        self.force_circular = force_circular
        self.uni_chromosome = uni_chromosome
        self.out_prob_threshold = out_prob_threshold
        self.keep_temp = keep_temp
        self.resume = resume
        self.min_alignment_len_cutoff = min_alignment_len_cutoff
        self.min_record_identity_cutoff = min_record_identity_cutoff
        self.min_read_identity_cutoff = min_read_identity_cutoff
        self.min_alignment_counts = min_alignment_counts
        self.kwargs = kwargs

        # init logger
        self.logfile = os.path.join(self.outdir, "traversome.log.txt")
        self.loglevel = loglevel.upper()
        setup_logger(loglevel=self.loglevel, timed=True, log_file=self.logfile)

        # alignment and graph values
        self.graph = None
        # self.alignment = None
        self.align_len_at_path_map = {}
        self.num_valid_records = 0
        # self.align_len_at_path_sorted = None
        self.max_alignment_length = None  # only used in the variant proposal, not in latter model fitting
        self.min_alignment_length = None
        self.read_paths = OrderedDict()
        self.read_paths_masked = set() # used to model selection and fitting, disabled for now or maybe permanently
        # used to generate an addition table for unused read paths for finetuning parameters
        self._raw_read_paths = OrderedDict()
        self._raw_max_alignment_length = None
        self._raw_min_alignment_length = None
        self.max_read_path_size = None
        self.subpath_generator = None

        # variant model to be generated
        self.user_variant_paths = []
        self.user_variant_fixed_ids = set()
        self.variant_paths = []  # each element is a tuple(path)
        self.variant_sizes = []
        self.variant_topos = []
        self.num_put_variants = None
        self.variant_readpath_counters = OrderedDict()  # variant -> dict(sub_path->sub_path_counts)
        self.all_sub_paths = OrderedDict()
        self.sbp_to_sbp_id = {}
        self.observed_sbp_id_set = set()
        self.bins_list = []
        self._cache_num_ali_starts = {}

        # for bootstrapping records
        self.records_pool_in_sbp_ids = []
        self.records_pool_to_sbp_ids = {}
        self.records_pool_sorted = []
        self.be_unidentifiable_to = {}
        # use the merged variants to represent each set of variants.
        # within each set the variants are unidentifiable to each other
        self.repr_to_merged_variants = {}
        # The result of model fitting using ml/mc, the base to update above model information for a second fitting run
        # {variant_id: percent}

        self._cid_sorter = {}
        self._cid_to_fid = OrderedDict()
        self._repr_cid_to_fid_column = OrderedDict()  # {repr_variant_id: fid_column_index}
        self._outcome_rep_cid_to_fids = OrderedDict()  # {repr_variant_id: list of unidentifiable fids}
        self.vp_unique_results = {}
        self.vp_unique_results_sorted = []

        # for single-best model selection
        # self.variant_proportions = OrderedDict()
        # self.res_loglike = None
        # self.res_criterion = None
        # self.variant_proportions_reps = []
        # self.variant_proportions_best = OrderedDict()
        
        # for multi-best search
        self._variant_proportions_cmb = OrderedDict()  # {tuple of variant ids -> {variant_id: percent}}
        self._variant_proportions_cmb_tuple = None, None
        # self._res_loglike_cmb = []
        # self._res_criterion_cmb = []
        self._variant_proportions_cmb_reps = []
        self._variant_proportions_cmb_best = OrderedDict()

        # self.pal_len_sbp_Xs = OrderedDict()
        # self.sbp_Xs = []

        self.bs_eligible = None
        self.max_like_fit = None
        self.bayesian_fit = None
        self.model = None
        self.random = random
        self.random.seed(random_seed)

    def run(self):
        """
        """
        logger.info("======== DIGESTING DATA STARTS ========")
        # TODO use json to store env parameters later
        self.graph = Assembly(self.graph_file)
        self.graph.update_vertex_clusters()
        raw_n_comp = len(self.graph.vertex_clusters)
        raw_n_vt = len(self.graph.vertex_info)
        # output new graph
        output_tmp_graph = False
        # choose the component
        graph_component_selection = self.kwargs.get("graph_component_selection", 0)
        if isinstance(graph_component_selection, int) or isinstance(graph_component_selection, slice):
            self.graph.reduce_graph_by_weight(component_ids=graph_component_selection)
            logger.trace("  graph reduced by component selection")
            output_tmp_graph = True
        elif isinstance(graph_component_selection, float):
            self.graph.reduce_graph_by_weight(cutoff_to_total=graph_component_selection)
            logger.trace("  graph reduced by component selection")
            output_tmp_graph = True

        # merge graph if possible; not that necessary
        if self.kwargs.get("keep_graph_redundancy", False):
            pass
        else:
            if self.graph.merge_all_possible_vertices():
                if self.alignment_file:
                    # the graph was changed and cannot be traced back to the names in the alignment
                    # this can also be moved down considering structure, however more computational efficient to be here
                    raise Exception("Graph be simplified by merging all possible nodes! "
                                    "Please provide the raw reads, or simplify the graph and redo the alignment, "
                                    "or remove '--merge-possible-contigs' from the options!")
                else:
                    logger.info("  graph merged")
                    output_tmp_graph = True
            self.graph.update_vertex_clusters()
        
        try:
            if self.graph.trim_overlaps(self.loglevel in ("DEBUG", "TRACE")):
                logger.trace("  graph trimmed by overlaps")
                output_tmp_graph = True
        except Exception as e:
            if "Solution not found for current graph" in str(e):
                if self.kwargs.get("check_conflicts", False):
                    logger.warning("Overlap-trimming solution not found for current graph!")
                    logger.warning("Graph-alignment conflict checking will be impaired!")
                else:
                    # I believe there's no major influence for downstream analysis
                    logger.info("Overlap-trimming solution not found for current graph.")
            else:
                raise e
        if self.graph_format == "fastg":
            logger.trace("  graph converted to gfa")
            output_tmp_graph = True

        if output_tmp_graph:
            self.graph_file = os.path.join(self.outdir, "tmp.processed.1.gfa")
            self.graph.write_to_gfa(self.graph_file)

        # logger.info("  #contigs in total: {}".format(len(self.graph.vertex_info)))
        # logger.info(
        #     "  #contigs in each component: {}".format(sorted([len(cls) for cls in self.graph.vertex_clusters])))

        self.load_user_paths()

        # if the alignment file is not provided, then align the reads to the graph
        if not self.alignment_file:
            if self.resume and os.path.exists(os.path.join(self.outdir, "tmp.alignment.gaf")):
                self.alignment_file = os.path.join(self.outdir, "tmp.alignment.gaf")
            elif executable("GraphAligner"):  # TODO: add path option
                self.alignment_file = os.path.join(self.outdir, "tmp.alignment.gaf")
                run_graph_aligner(
                    graph_file=self.graph_file,
                    seq_file=self.reads_file,
                    alignment_file=self.alignment_file,
                    num_processes=self.num_processes,
                    other_params=self.kwargs.get("graph_aligner_params", ""))
            else:
                raise Exception("GraphAligner not available or damaged!")

        # parse the alignment and detect abnormal alignments, if the alignment is redo after detection, then reparse the alignment
        for parse_attempt in range(2):
            min_alignment_len_cutoff = self.min_alignment_len_cutoff
            min_record_identity_cutoff = self.min_record_identity_cutoff
            min_read_identity_cutoff = self.min_read_identity_cutoff
            if min_read_identity_cutoff != "auto" and min_alignment_len_cutoff != "auto":
                alignment = GraphAlignRecords(
                    self.alignment_file,
                    min_record_identity=min_record_identity_cutoff,
                    min_align_len=min_alignment_len_cutoff,
                    min_identity=min_read_identity_cutoff,
                    assembly_graph_obj=self.graph,
                )
            else:
                # initial read
                alignment = GraphAlignRecords(
                    self.alignment_file,
                    min_record_identity=min_record_identity_cutoff,
                    min_align_len=100 if min_alignment_len_cutoff == "auto" else min_alignment_len_cutoff,
                    min_identity=0.8 if min_read_identity_cutoff == "auto" else min_read_identity_cutoff,
                    assembly_graph_obj=self.graph,
                    )
                self.auto_filter_alignment(alignment, min_alignment_len_cutoff, min_read_identity_cutoff)
            if not alignment.raw_records:
                logger.error("Insufficient alignment records remains after filtering!")
                raise SystemExit(0)
            if self.min_alignment_counts == "auto":
                min_alignment_counts = self.estimate_min_align_counts(alignment=alignment)
                logger.info(f"Setting minimum alignment counts to {min_alignment_counts}")
            else:
                min_alignment_counts = self.min_alignment_counts

            if parse_attempt == 0:
                # no need to check conflicts for the second round
                if self.kwargs.get("check_conflicts", False):
                    reparse_alignment = self.detect_alignment_abnormals(alignment)
                    if not reparse_alignment:
                        break
                else:
                    break

        if self.kwargs.get("use_alignment_cov", False):
            self.generate_read_paths(
                graph_alignment=alignment,
                filter_by_graph=False,
                min_alignment_counts=min_alignment_counts)
            self.estimate_contig_coverages_from_read_paths()
            # reset read_paths to empty
            self.read_paths = OrderedDict()
            self.read_paths_masked = set()
            self._raw_read_paths = OrderedDict()

        if self.purge_graph_by_depth(depth_threshold=self.kwargs.get("purge_shallow_contigs", 0.001)):
            output_tmp_graph = True

        if self.kwargs.get("prune_terminal_contigs", False):
            if self.prune_terminal_contigs():
                output_tmp_graph = True

        self.generate_read_paths(
            graph_alignment=alignment,
            filter_by_graph=True,
            min_alignment_counts=min_alignment_counts)
        if self.kwargs.get("use_alignment_cov", False):
            self.estimate_contig_coverages_from_read_paths()

        logger.info("Align stat - #raw noisy records aligned: %i " % alignment.n_file_records)
        logger.info("Align stat = #raw read paths: %i" % len(self._raw_read_paths))
        logger.info("Align stat - #filtered records aligned: %i" % len(alignment.raw_records))
        logger.info("Align stat - #filtered reads aligned: %i" % len(alignment.read_records))
        logger.info("Align stat - #filtered read paths: %i" % len(self.read_paths))
        # self.get_align_len_dist(graph_alignment=alignment)
        geometric_mean, geometric_std, lower_95, upper_95 =\
            summarize_read_lengths(list(self.align_len_at_path_map.values()))
        logger.info(f"Align stat - filtered length range at path: "
                    f"Range=[{self.min_alignment_length}-{self.max_alignment_length}], "
                    f"95%=[{lower_95:.0f}-{upper_95:.0f}], "
                    f"GM={geometric_mean:.0f}, GSD={geometric_std:.2f}")
        logger.info("Align stat - filtered max size at path: {}".format(self.max_read_path_size))
        # logger.info("  #read paths masked: %i" % len(self.read_paths_masked))
        # free memory to reduce the burden for potential downstream parallelization
        del alignment
        gc.collect()
        aligned_path_bases = sum(self.align_len_at_path_map.values())
        graph_len = sum(self.graph.vertex_info[v_].len for v_ in self.graph.vertex_info)
        filtered_ave_depth = float(aligned_path_bases / graph_len)
        logger.info(f"Align stat - filtered average depth: {filtered_ave_depth: .2f}")
        if not self.kwargs.get("keep_unaligned_contigs", False):
            self.prune_unaligned_contigs()
            output_tmp_graph = True
        logger.info("Graph stat - #components: {}".format(raw_n_comp))
        logger.info("Graph stat - #raw contigs: {}".format(raw_n_vt))
        logger.info("Graph stat - #filtered contigs: {}".format(len(self.graph.vertex_info)))
        logger.info("Graph stat - #filtered contigs in each component: {}".format(
            sorted([len(cls) for cls in self.graph.vertex_clusters])))
        total_counts = sum(self.graph.vertex_info[v_].cov * self.graph.vertex_info[v_].len
                           for v_ in self.graph.vertex_info)
        logger.info(f"Graph stat - average depth: {float(total_counts) / graph_len: .2f}")

        if output_tmp_graph:
            self.graph.write_to_gfa(os.path.join(self.outdir, "tmp.processed.2.gfa"))

        if filtered_ave_depth < 1.:
            logger.error("Insufficient alignment records remains after filtering!")
            raise SystemExit(1)

        self.subpath_generator = VariantSubPathsGenerator(
            graph=self.graph,
            # force_circular=self.force_circular,
            min_alignment_len=self.min_alignment_length,
            max_alignment_len=self.max_alignment_length,
            read_paths_hashed=set(self.read_paths))
        logger.info("======== DIGESTING DATA ENDS ========\n")

        logger.info("======== VARIANTS SEARCHING STARTS ========")
        # logger.debug("Cleaning graph ...")
        # self.clean_graph()
        if self.kwargs.get("max_valid_search", 100000) == 0:
            self.variant_paths = list(self.user_variant_paths)
            self.variant_sizes = [self.graph.get_path_length(variant_p, check_valid=False, adjust_for_cyclic=True)
                                  for variant_p in self.variant_paths]
            self.variant_topos = [self.graph.is_circular_path(variant_p)
                                  for variant_p in self.variant_paths]
            # self._update_params_for_variants()
        else:
            logger.debug("Generating candidate variant paths ...")
            self.gen_candidate_variants(
                # path_generator=path_gen_scheme,
                graph_based_multiplicity_wiggle=self.kwargs.get("graph_based_multiplicity_wiggle", 0.),
                penalty_for_size=self.kwargs.get("penalty_for_size", 1e-5),
                fix_shallowest_contig=self.kwargs.get("fix_shallowest_contig", 0),
                # start_strategy=getattr(self.kwargs.get("search_start_scheme", None), "value", "random"),
                search_decay_factor=self.kwargs.get("search_decay_factor", 20.),
                min_num_search=self.kwargs.get("min_valid_search", 500),
                max_num_search=self.kwargs.get("max_valid_search", 10000),
                max_num_traversals=self.kwargs.get("max_num_traversals", 30000),
                max_uniq_traversal=self.kwargs.get("max_uniq_traversal", 200),
                max_uncover_ratio=self.kwargs.get("max_uncover_ratio", 0.001),
                num_processes=self.num_processes,
                uni_chromosome=self.uni_chromosome,
                size_ratio=self.kwargs.get("size_ratio", 0.0)
            )

        self.num_put_variants = len(self.variant_paths)
        if self.num_put_variants == 0:
            logger.error("No candidate variants found!")
            logger.info("======== VARIANTS SEARCHING ENDS ========\n")
            raise SystemExit(1)
        elif self.num_put_variants == 1 or len(self.repr_to_merged_variants) == 1:
            # self.variant_proportions_best[0] = self.variant_proportions[0] = 1.   # for reverse model selection
            # TODO
            self._variant_proportions_cmb_best[(0,)][0] = self._variant_proportions_cmb[(0,)][0] = 1.   # for genetic_algorithm_search
            logger.info("======== VARIANTS SEARCHING ENDS ========\n")
            self.gen_all_informative_sub_paths(silent=True)
        else:
            for go_p, path in enumerate(self.variant_paths):
                logger.info("cid_{} PATH: {}".format(go_p, self.graph.repr_path(path)))
            logger.info("======== VARIANTS SEARCHING ENDS ========\n")
            logger.info("======== MODEL SELECTION & FITTING STARTS ========")

            logger.info("Generating sub-paths ..")
            self.gen_all_informative_sub_paths()
            # ONLY apply self.read_paths_masked to model selection and fitting
            # main_sub_paths, align_len_at_path_sorted = \
            #     self.sample_sub_paths(masking=self.read_paths_masked)
            # logger.info("Indexing {} valid informative sub-paths after masking ".format(len(filtered_sub_paths)))
            # self.all_sub_paths = filtered_sub_paths  # assign the post-filter sub_paths
            main_sub_paths = self.all_sub_paths
            logger.info("Indexing {} valid informative sub-paths ".format(len(main_sub_paths)))
            rec_id_sorted_by_len, align_len_at_path_sorted = \
                zip(*sorted(self.align_len_at_path_map.items(), key=lambda x: (x[1], x[0])))  # by length then record id
            # align_len_at_path_sorted = sorted(self.align_len_at_path_map.values())
            max_len, min_len = max(align_len_at_path_sorted), min(align_len_at_path_sorted)
            logger.info("Alignment length range at path: [{}, {}]".format(min_len, max_len))
            logger.info("Alignment max size at path: {}".format(self.get_max_read_path_size(main_sub_paths)))
            self.bins_list = self.generate_multinomial_bin_stats(
                main_sub_paths,
                rec_id_sorted_by_len=rec_id_sorted_by_len,
                align_len_at_path_sorted=align_len_at_path_sorted,
                quiet=False)
            # self.generate_sub_path_stats(main_sub_paths, align_len_at_path_sorted)

            # build an index, used to access all read paths are covered in later model selection
            sbp_to_sbp_id = self.update_sp_to_sp_id_dict(main_sub_paths)
            # difference between this number and total number of sub-paths
            #            will happen when the current variants cannot cover all read paths
            #                     or when an alignable path is not informative
            logger.debug("Estimating candidate variant frequencies using Maximum Likelihood...")

            self.model = PathMultinomialModel(
                variant_sizes=self.variant_sizes,
                variant_topos=self.variant_topos,
                bins_list=self.bins_list,
                all_sub_paths=main_sub_paths)
            # self.variant_proportions, self.res_loglike, self.res_criterion = \
            #     self.fit_model_using_reverse_model_selection(
            #         model=self.model,
            #         sbp_to_sbp_id=sbp_to_sbp_id,
            #         criterion=self.kwargs.get("model_criterion", Criterion.BIC))
            
            # genetic_algorithm_search
            # self._variant_proportions_cmb = OrderedDict()
            # for sorted_var_ids, (res_prop, echo_prop, this_like, this_criteria) in \
            #         self.fit_model_using_genetic_algorithm(
            #             model=self.model,
            #             sbp_to_sbp_id=sbp_to_sbp_id,
            #             criterion=self.kwargs.get("model_criterion", Criterion.AIC),
            #             init_self_max_like=True
            #         ).items():
            #     self._variant_proportions_cmb[sorted_var_ids] = (res_prop, echo_prop, this_like, this_criteria)
            
            # switch back to the reverse model selection
            self._variant_proportions_cmb = OrderedDict()
            for sorted_var_ids, (res_prop, echo_prop, this_like, this_criteria) in \
                    self.fit_model_using_reverse_model_selection(
                        model=self.model,
                        sbp_to_sbp_id=sbp_to_sbp_id,
                        criterion=self.kwargs.get("model_criterion", Criterion.BIC),
                        init_self_max_like=True,
                        num_processes=self.num_processes
                    ).items():
                self._variant_proportions_cmb[sorted_var_ids] = (res_prop, echo_prop, this_like, this_criteria)
            
            logger.info("======== MODEL SELECTION & FITTING ENDS ========\n")

            if self.kwargs.get("bootstrap", 0):  # or self.kwargs.get("jackknife", 0):
                logger.info("======== BOOTSTRAPPING STARTS ========")
                self.do_bootstrap(n_processes=self.num_processes)
                self.summarize_bootstrap_replicates()
                logger.info("======== BOOTSTRAPPING ENDS ========\n")

            if not self._variant_proportions_cmb_best:  # if it is not modified during subsampling
                for sorted_var_ids, res_tuple in self._variant_proportions_cmb.items():
                    tuple_var_ids = tuple(sorted(sorted_var_ids, key=lambda x: self._cid_sorter[x]))
                    self._variant_proportions_cmb_best[tuple_var_ids] = res_tuple
            # MCMC
            if self.kwargs.get("n_generations", 0) > 0 and \
                    len([repr_v
                         for repr_v in self._variant_proportions_cmb_best
                         if repr_v in self.repr_to_merged_variants]) > 1:  # if there are more than 1 variants
                # TODO add mcmc result to the summary table
                logger.info("======== BAYESIAN ESTIMATION STARTS ========")
                logger.debug("Estimating candidate variant frequencies using Bayesian MCMC ...")
                for sorted_var_ids in self._variant_proportions_cmb.keys():
                    res_prop = self.fit_model_using_bayesian_mcmc(chosen_ids=set(sorted_var_ids))
                    # add mcmc posterior score?
                    tuple_var_ids = tuple(sorted(sorted_var_ids, key=lambda x: self._cid_sorter[x]))
                    self._variant_proportions_cmb_best[tuple_var_ids] = (res_prop, "-", "-", "-")
                logger.info("======== BAYESIAN ESTIMATION ENDS ========\n")

        logger.info("======== OUTPUT FILES STARTS ========")
        self.generate_fid()
        self.output_variant_info()
        self.output_readpath_info()
        self.output_sampling_info()
        self.output_result_info()
        if self.kwargs.get("bootstrap", 0) == 0 or self.bs_eligible or self.num_put_variants == 1:
            self.output_pangenome_graph()
            self.output_seqs()
        # remove temporary files
        if not self.keep_temp:
            for f_ in os.listdir(self.outdir):
                # remove tmp.*.gfa and tmp.*.gaf
                if f_.startswith("tmp.") and os.path.isfile(os.path.join(self.outdir, f_)):
                    os.remove(os.path.join(self.outdir, f_))
                # remove tmp.candidates
                elif f_.startswith("tmp.") and os.path.isdir(os.path.join(self.outdir, f_)):
                    for f__ in os.listdir(os.path.join(self.outdir, f_)):
                        os.remove(os.path.join(self.outdir, f_, f__))
                    os.rmdir(os.path.join(self.outdir, f_))
            # # also remove the non empty directory tmp.candidates
            # if os.path.exists(os.path.join(self.outdir, "tmp.candidates")):
            #     for f_ in os.listdir(os.path.join(self.outdir, "tmp.candidates")):
            #         os.remove(os.path.join(self.outdir, "tmp.candidates", f_))
            #     os.rmdir(os.path.join(self.outdir, "tmp.candidates"))
        logger.info("======== OUTPUT FILES ENDS ========\n")

    def auto_filter_alignment(self, alignment, min_alignment_len_cutoff, min_alignment_identity_cutoff):
        # lengths = [r.p_align_len for r in alignment.raw_records]
        # identities = [r.identity for r in alignment.raw_records]
        lengths = [rd.p_align_len for rd in alignment.read_records.values()]
        identities = [rd.p_identity for rd in alignment.read_records.values()]
        if min_alignment_len_cutoff != "auto":
            min_ln_adj_end = 1  # only generate one combination
            start_length = min_alignment_len_cutoff
        else:
            # min_ln_adj_end = 2000
            min_ln_adj_end = 1  # only generate one combination
            start_length = 5000  # TODO make to be set
        if min_alignment_identity_cutoff != "auto":
            min_id_adj_end = 1e-5   # only generate one combination
            start_identity = min_alignment_identity_cutoff
        else:
            min_id_adj_end = None
            start_identity = 0.95  # TODO make to be set
        graph_len = sum(self.graph.vertex_info[v_].len for v_ in self.graph.vertex_info)
        target_sum = graph_len * self.kwargs.get("quality_control_alignment_cov", 250.)
        optimal_min_id_adj, optimal_min_ln_adj, min_diff, res_sum = optimize_min_adj(
            lengths=lengths,
            identities=identities,
            target_sum=target_sum,
            start_length=start_length,
            start_identity=start_identity,
            min_id_adj_end=min_id_adj_end,
            min_ln_adj_end=min_ln_adj_end)
        if res_sum / graph_len < 5.:
            logger.warning("Potentially insufficient alignment of good quality!")
        new_min_alignment_len_cutoff = start_length + optimal_min_ln_adj
        if min_alignment_len_cutoff == "auto":
            logger.info(f"Setting minimum alignment length to {new_min_alignment_len_cutoff}")
        new_min_alignment_identity_cutoff = start_identity + optimal_min_id_adj
        if min_alignment_identity_cutoff == "auto":
            logger.info(f"Setting minimum alignment identity to {new_min_alignment_identity_cutoff}")
        alignment.min_align_len = new_min_alignment_len_cutoff
        alignment.min_identity = new_min_alignment_identity_cutoff
        alignment.filter_read_records(min_align_len=new_min_alignment_len_cutoff, min_identity=new_min_alignment_identity_cutoff)

    def estimate_min_align_counts(self, alignment):
        """empirical function"""
        lengths = [r.p_align_len for r in alignment.raw_records]
        identities = [r.identity for r in alignment.raw_records]
        mean_len = np.average(lengths)
        mean_identity = np.average(identities, weights=lengths)
        graph_len = sum(self.graph.vertex_info[v_].len for v_ in self.graph.vertex_info)
        valid_bases = sum([r.p_align_len for r in alignment.raw_records])
        # use read statistics instead of raw alignment records
        # lengths = [r.p_align_len for r in alignment.read_records.values()]
        # identities = [r.p_identity for r in alignment.read_records.values()]
        # mean_len = np.average(lengths)
        # mean_identity = np.average(identities, weights=lengths)
        # graph_len = sum(self.graph.vertex_info[v_].len for v_ in self.graph.vertex_info)
        # valid_bases = sum(lengths)

        # logger.info(f"DEBUG - valid_bases: {valid_bases}")
        # logger.info(f"DEBUG - graph_len: {graph_len}")
        # logger.info(f"DEBUG - mean id: {mean_identity}")
        # logger.info(f"DEBUG - mean len: {mean_len}")
        # depth_factor = (np.log2(1 - mean_identity) * np.log2(mean_len) / 25.) ** 2
        # length negatively correlates to the total number of reads, therefore,
        #     negatively correlates to expected correct same-path reads
        # -np.log10(1- mean_identity)*10 is the phred quality score
        # depth_factor = -np.log(1 - mean_identity) * mean_len**0.5 / 35
        # logger.info("DEBUG - format: -np.log(1 - mean_identity) * mean_len**0.5 / 35")
        depth_factor = -np.log(1 - mean_identity) * (3 + mean_len / 10000)
        # 2025-08-18
        # return max(3, math.ceil(valid_bases / (graph_len * depth_factor) + 1))
        return max(3, math.ceil(valid_bases / (2 * graph_len * depth_factor) + 1))

    def calculate_reps(self, n_replicate, threshold, n_digit):
        """
        Calculate the replicates for bootstrapping using single process.

        :param n_replicate: number of replicates to run
        :param threshold: threshold for the number of unique variants
        :param n_digit: number of digits for displaying the replicate number
        """
        count_unique = {}  # a dict to count the number of unique variants across replicates
        go_bs = 0
        self.bs_eligible = True
        while go_bs < n_replicate:
            logger.debug(f"Sampling {go_bs + 1} --------")
            logger.debug("Generating sub-paths ..")
            # if self.kwargs.get("bootstrap", 0):
            sampled_sub_paths, rec_id_sorted_by_len, align_len_at_path_sorted = \
                self.sample_sub_paths(bootstrap_size=self.num_valid_records, masking=self.read_paths_masked)
            # else:
            #     sampled_sub_paths, rec_id_sorted_by_len, align_len_at_path_sorted = \
            #         self.sample_sub_paths(
            #             jackknife_size=int(self.num_valid_records / float(n_replicate)),
            #             masking=self.read_paths_masked)
            logger.debug("Indexing {} valid informative sub-paths after masking ".format(len(sampled_sub_paths)))
            if not sampled_sub_paths:
                continue
            # self.generate_sub_path_stats(sampled_sub_paths, align_len_at_path_sorted=align_len_at_path_sorted)
            bins_list = self.generate_multinomial_bin_stats(
                all_sub_paths=sampled_sub_paths,
                rec_id_sorted_by_len=rec_id_sorted_by_len,
                align_len_at_path_sorted=align_len_at_path_sorted)
            sbp_to_sbp_id = self.update_sp_to_sp_id_dict(sampled_sub_paths)
            sampled_model = PathMultinomialModel(
                variant_sizes=self.variant_sizes,
                variant_topos=self.variant_topos,
                bins_list=bins_list,
                all_sub_paths=sampled_sub_paths)
            # using reverse model selection
            # v_prop, *foo = self.fit_model_using_reverse_model_selection(
            #     model=sampled_model,
            #     sbp_to_sbp_id=sbp_to_sbp_id,
            #     criterion=self.kwargs.get("model_criterion", Criterion.AIC),
            #     init_self_max_like=False,
            #     bootstrap_str=f"BS{go_bs + 1: 0{n_digit}d}")
            # self.variant_proportions_reps.append(v_prop)
            
            # using genetic algorithm search
            # best_models = self.fit_model_using_genetic_algorithm(
            #     model=sampled_model,
            #     sbp_to_sbp_id=sbp_to_sbp_id,
            #     criterion=self.kwargs.get("model_criterion", Criterion.AIC),
            #     init_self_max_like=False,
            #     bootstrap_str=f"BS{go_bs + 1: 0{n_digit}d}")
            
            # switch back to the reverse model selection
            best_models = self.fit_model_using_reverse_model_selection(
                model=sampled_model,
                sbp_to_sbp_id=sbp_to_sbp_id,
                criterion=self.kwargs.get("model_criterion", Criterion.BIC),
                init_self_max_like=False,
                bootstrap_str=f"BS{go_bs + 1: 0{n_digit}d}")

            self._variant_proportions_cmb_reps.append(best_models)
            
            last_v_tuple = tuple(sorted(best_models.keys()))
            # self.res_loglike_reps.append(loglike)
            # self.res_criteria_reps.append(criteria)

            if not self._check_bs_threshold(
                    count_unique=count_unique, n_reps=n_replicate, threshold=threshold, last_v_tuple=last_v_tuple):
                # if go_bs < n_replicate - 1:
                #     logger.info("Sampling terminates due to divergence in bootstraps (see '--bs-threshold'). "
                #                 "No convincing support can be found given the dataset and parameters. ")
                self.bs_eligible = False
                break
            go_bs += 1

    def calculate_reps_worker(
            self, n_replicate, threshold, n_digit, g_vars, event, error_queue, job_id_queue, num_processes_list):
        """
        Worker function for parallel processing of bootstrapping replicates.

        :param n_replicate: number of replicates to run
        :param threshold: threshold for the number of unique variants
        :param n_digit: number of digits for displaying the replicate number
        :param g_vars: global variables for multiprocessing
        """
        try:
            work_id = job_id_queue.get()
            n_process = num_processes_list[work_id] if num_processes_list else 1
            # don't know why the subprocesses will be generated with an additional process during reverse model selection
            # so we need to reduce the number of processes by 1
            n_process = 1 if n_process <= 2 else n_process
            with g_vars.go_rep_lock:
                go_bs = g_vars.go_rep.value
                g_vars.go_rep.value += 1
                # explicitly reset the loglevel in cases where macOS subprocesses do not obey
                setup_logger(loglevel=self.loglevel, log_file=self.logfile)
            while go_bs < n_replicate:
                logger.debug(f"Sampling {go_bs + 1} --------")
                logger.debug("Generating sub-paths ..")
                # if self.kwargs.get("bootstrap", 0):
                sampled_sub_paths, rec_id_sorted_by_len, align_len_at_path_sorted = \
                    self.sample_sub_paths(bootstrap_size=self.num_valid_records, masking=self.read_paths_masked)
                # else:
                #     sampled_sub_paths, rec_id_sorted_by_len, align_len_at_path_sorted = \
                #         self.sample_sub_paths(
                #             jackknife_size=int(self.num_valid_records / float(n_replicate)),
                #             masking=self.read_paths_masked)
                logger.debug("Indexing {} valid informative sub-paths after masking ".format(len(sampled_sub_paths)))
                if not sampled_sub_paths:
                    continue
                # self.generate_sub_path_stats(sampled_sub_paths, align_len_at_path_sorted=align_len_at_path_sorted)
                bins_list = self.generate_multinomial_bin_stats(
                    all_sub_paths=sampled_sub_paths,
                    rec_id_sorted_by_len=rec_id_sorted_by_len,
                    align_len_at_path_sorted=align_len_at_path_sorted)
                sbp_to_sbp_id = self.update_sp_to_sp_id_dict(sampled_sub_paths)
                sampled_model = PathMultinomialModel(
                    variant_sizes=self.variant_sizes,
                    variant_topos=self.variant_topos,
                    bins_list=bins_list,
                    all_sub_paths=sampled_sub_paths)
                # using reverse model selection
                # v_prop, *foo = self.fit_model_using_reverse_model_selection(
                #     model=sampled_model,
                #     sbp_to_sbp_id=sbp_to_sbp_id,
                #     criterion=self.kwargs.get("model_criterion", Criterion.AIC),
                #     init_self_max_like=False,
                #     bootstrap_str=f"BS{go_bs + 1: 0{n_digit}d}")
                # self.variant_proportions_reps.append(v_prop)

                # using genetic algorithm search
                # best_models = self.fit_model_using_genetic_algorithm(
                #     model=sampled_model,
                #     sbp_to_sbp_id=sbp_to_sbp_id,
                #     criterion=self.kwargs.get("model_criterion", Criterion.AIC),
                #     init_self_max_like=False,
                #     bootstrap_str=f"BS{go_bs + 1: 0{n_digit}d}",
                #     num_processes=n_process)

                # switch back to the reverse model selection
                best_models = self.fit_model_using_reverse_model_selection(
                    model=sampled_model,
                    sbp_to_sbp_id=sbp_to_sbp_id,
                    criterion=self.kwargs.get("model_criterion", Criterion.BIC),
                    init_self_max_like=False,
                    bootstrap_str=f"BS{go_bs + 1: 0{n_digit}d}",
                    num_processes=n_process,
                    event=event)

                g_vars.variant_proportions_cmb_reps[go_bs] = best_models

                last_v_tuple = tuple(sorted(best_models.keys()))

                with g_vars.count_unique_lock:
                    count_unique = dict(g_vars.count_unique)
                    g_vars.count_valid.value += 1
                    if not self._check_bs_threshold(count_unique=count_unique,
                                                    n_reps=n_replicate,
                                                    threshold=threshold,
                                                    last_v_tuple=last_v_tuple):
                        # g_vars.count_unique.update(count_unique)  # update the count_unique dict
                        g_vars.bs_eligible.value = False
                        event.set()  # signal that all jobs are done
                        return
                    g_vars.count_unique.update(count_unique)  # update the count_unique dict

                with g_vars.go_rep_lock:
                    go_bs = g_vars.go_rep.value
                    g_vars.go_rep.value += 1
                if go_bs >= n_replicate:
                    return
        except Exception as e:
            tb = traceback.format_exc()
            location = sys.exc_info()[-1]
            logger.error(f"Error in worker {os.getpid()}: {e}\n{tb}")
            error_queue.put((e, tb, location))
            event.set()

    def do_bootstrap(self, n_processes=1):
        """
        using bootstrap
        """
        self._prepare_for_sampling()
        n_replicate = self.kwargs.get("bootstrap", 0) # \
            # if self.kwargs.get("bootstrap", 0) else self.kwargs.get("jackknife", 0)
        threshold = self.kwargs.get("bs_threshold", 0.95)
        # self.variant_proportions_reps = []
        self._variant_proportions_cmb_reps = []
        n_digit = len(str(n_replicate))
        
        if n_processes <= 1:
            # single process
            self.calculate_reps(n_replicate=n_replicate, threshold=threshold, n_digit=n_digit)
        else:
            # multi-process
            logger.info(f"Running {n_processes} processes for bootstrapping ...")
            with Manager() as manager:
                event = manager.Event()
                error_queue = manager.Queue()
                job_id_queue = manager.Queue()
                global_vars = manager.Namespace()
                global_vars.variant_proportions_cmb_reps = manager.list([None] * n_replicate)
                global_vars.count_unique = manager.dict()
                global_vars.count_unique_lock = manager.Lock()
                global_vars.count_valid = manager.Value("i", 0)
                global_vars.bs_eligible = manager.Value("b", True)
                global_vars.go_rep = manager.Value("i", 0)
                global_vars.go_rep_lock = manager.Lock()
                # prioritize the multiprocessing for bootstrapping over each bootstrap replicate
                # distribute the number of processes evenly across replicates
                real_num_processes = min(n_processes, n_replicate)
                for i in range(real_num_processes):
                    job_id_queue.put(i)
                num_processes_list = np.array([n_processes // n_replicate] * real_num_processes)
                if n_replicate % n_processes != 0:
                    num_processes_list[:n_replicate % n_processes] += 1  # distribute the remaining processes evenly
                num_processes_list = num_processes_list.tolist()
                logger.info("Serializing traversome for multiprocessing ..")
                payload = dill.dumps(
                    (self.calculate_reps_worker,
                     (n_replicate, threshold, n_digit, global_vars, event, error_queue, job_id_queue, num_processes_list)
                     ))
                job_list = []
                for go_w in range(real_num_processes):
                    logger.info("assigned job to worker {}".format(go_w + 1))
                    p = Process(target=run_dill_encoded, args=(payload,))
                    job_list.append(p)
                    p.start()
                try:
                    # wait for all processes to finish or until the event is set
                    while not event.is_set():
                        if all(not p.is_alive() for p in job_list):
                            break
                        time.sleep(0.5)
                finally:
                    for p in job_list:
                        if p.is_alive():
                            # time.sleep(0.25) # give some time for the process to end its subprocesses
                            p.terminate()
                        p.join()  # wait for all processes to finish
                # pool_obj = Pool(processes=real_num_processes)
                # job_list = []
                # for go_w in range(real_num_processes):
                #     logger.info("assigning job to worker {}".format(go_w + 1))
                #     job_list.append(pool_obj.apply_async(run_dill_encoded, (payload,)))
                #     logger.info("assigned job to worker {}".format(go_w + 1))
                # pool_obj.close()
                # event.wait()
                # pool_obj.terminate()
                # while not error_queue.empty():
                #     e, tb, location = error_queue.get()
                #     logger.error("\n" + "".join(tb))  # + "\n" + str(location) + "\n" + str(e))
                #     sys.exit(0)
                self._variant_proportions_cmb_reps = [x for x in list(global_vars.variant_proportions_cmb_reps) if x]
                self.bs_eligible = global_vars.bs_eligible.value

        if not self.bs_eligible and len(self._variant_proportions_cmb_reps) < n_replicate:
            logger.info("Sampling terminates due to divergence in bootstraps (see '--bs-threshold'). "
                        "No convincing support can be found given the dataset and parameters. ")

        # if loglevel is reset, set it back
        setup_logger(loglevel=self.loglevel, timed=True, log_file=self.logfile)
        logger.info("Bootstrapping finished with {} replicates.".format(len(self._variant_proportions_cmb_reps)))

    def summarize_bootstrap_replicates(self):
        """ Summarize the bootstrap replicates. """
        # summarize the replicates
        # TODO: can we sort the result later after the total summarize/re-evaluation is done?
        # if self._variant_proportions_cmb_reps:
        #     num_reps = len(self._variant_proportions_cmb_reps)
        #     self.vp_unique_results = {}
        #     for go_r, best_models in enumerate(self._variant_proportions_cmb_reps):
        #         sorted_best_models_tuple = tuple(sorted(best_models.keys()))
        #         if sorted_best_models_tuple not in self.vp_unique_results:
        #             self.vp_unique_results[sorted_best_models_tuple] = {"rep_ids": [], "support": None, "values": []}
        #         self.vp_unique_results[sorted_best_models_tuple]["rep_ids"].append(go_r)
        #         v_props_comb = []
        #         for b_m in sorted_best_models_tuple:
        #             v_props_comb.append(best_models[b_m][0])  # res_prop
        #         self.vp_unique_results[sorted_best_models_tuple]["values"].append(tuple(v_props_comb))

        # if there are replicates, summarize the replicates
        logger.info("Summarizing the replicates ..")
        if self._variant_proportions_cmb_reps:
            num_reps = len(self._variant_proportions_cmb_reps)
            self.vp_unique_results = {}
            # TODO # sort the res by proportion decreasingly (-x[1]), then c_id (x[0])
            self.__sorting_cid()  # use a universal cid_sorter to keep them in order across bootstraps
            for go_r, best_models in enumerate(self._variant_proportions_cmb_reps):
                sorted_best_models = sorted(best_models.items(), key=lambda x: x[0])
                tuple_v_chosen_cmb = []
                tuple_v_props_cmb = []
                for b_m in sorted_best_models:
                    res_prop = b_m[1][0]
                    sorted_res = sorted(list(res_prop.items()), key=lambda x: self._cid_sorter[x[0]])
                    tuple_v_chosen, tuple_v_props = zip(*sorted_res)
                    tuple_v_chosen_cmb.append(tuple_v_chosen)
                    tuple_v_props_cmb.append(tuple_v_props)
                tuple_v_chosen_cmb = tuple(tuple_v_chosen_cmb)
                tuple_v_props_cmb = tuple(tuple_v_props_cmb)
                if tuple_v_chosen_cmb not in self.vp_unique_results:
                    self.vp_unique_results[tuple_v_chosen_cmb] = {"rep_ids": [], 
                                                                  "support": None, 
                                                                  "values": [],
                                                                  "raw_prop": None,
                                                                  "raw_like": None,
                                                                  "raw_criterion": None}
                self.vp_unique_results[tuple_v_chosen_cmb]["rep_ids"].append(go_r)
                self.vp_unique_results[tuple_v_chosen_cmb]["values"].append(tuple_v_props_cmb)

            # calculate supports
            for res_info in self.vp_unique_results.values():
                support = len(res_info["rep_ids"]) / float(num_reps)
                res_info["support"] = support

            # 5. sort by supports AND whether concordant with the estimate from raw dataset
            #    use vp_unique_results_sorted to determine the way to summarize the replicates
            # 5.1 get the sorted estimate from the raw dataset
            raw_best = sorted(self._variant_proportions_cmb.items(), key=lambda x: x[0])
            raw_tuple_v_chosen = []
            raw_tuple_v_props = []
            raw_tuple_like = []
            raw_tuple_crt = []
            for b_m in raw_best:
                res_prop, foo, res_like, res_criteria = b_m[1]
                # sort the res by proportion decreasingly (-x[1]), then c_id (x[0])
                sorted_res = sorted(list(res_prop.items()), key=lambda x: self._cid_sorter[x[0]])
                tuple_v_chosen, tuple_v_props = zip(*sorted_res)
                raw_tuple_v_chosen.append(tuple_v_chosen)
                raw_tuple_v_props.append(res_prop)
                raw_tuple_like.append(res_like)
                raw_tuple_crt.append(res_criteria)
            raw_res = tuple(raw_tuple_v_chosen)
            # self._variant_proportions_cmb_tuple = raw_res, raw_tuple_v_props, raw_tuple_like, raw_tuple_crt
            # 5.2 sort
            self.vp_unique_results_sorted = \
                sorted(self.vp_unique_results, key=lambda x: (-self.vp_unique_results[x]["support"], x != raw_res, x))

            # use the entire dataset to recalculate loglike and criterion
            # for these results with bootstrap larger than 0.1 (arbitrarily)
            if self.bs_eligible and \
                    (len(self.vp_unique_results_sorted) > 1 or self.vp_unique_results_sorted[0] != raw_res):
                logger.info("Re-evaluate the supported models using whole dataset ..")
            sbp_to_sbp_id = self.update_sp_to_sp_id_dict(self.all_sub_paths)
            cache_new_best = OrderedDict()
            for go_s, tuple_v_chosen_cmb in enumerate(self.vp_unique_results_sorted):
                # TODO: need to create class to record the results later
                chosen_dict = self.vp_unique_results[tuple_v_chosen_cmb]
                if tuple_v_chosen_cmb == raw_res:
                    raw_prop = OrderedDict()
                    for tuple_v_chosen, v_prop in zip(raw_res, raw_tuple_v_props):
                        raw_prop[tuple_v_chosen] = v_prop
                    raw_like = list(raw_tuple_like)
                    raw_criterion = list(raw_tuple_crt)
                else:
                    # re-evaluate only when bs is eligible
                    if self.bs_eligible:
                        if chosen_dict["support"] > 0.05 or go_s == 0:  # the first one
                            raw_prop = OrderedDict()
                            raw_like, raw_criterion = [], []
                            for tuple_v_chosen in tuple_v_chosen_cmb:
                                this_prop, foo, this_like, this_criterion = self.fit_model_using_point_maximum_likelihood(
                                    model=self.model,
                                    sbp_to_sbp_id=sbp_to_sbp_id,
                                    criterion=self.kwargs.get("model_criterion", Criterion.BIC),
                                    chosen_ids=set(tuple_v_chosen),
                                    init_self_max_like=False)
                                cache_new_best[tuple(tuple_v_chosen)] = (this_prop, foo, this_like, this_criterion)
                                raw_prop[tuple_v_chosen] = this_prop
                                raw_like.append(this_like)
                                raw_criterion.append(this_criterion)
                        else:
                            values = np.average(self.vp_unique_results[tuple_v_chosen_cmb]["values"], axis=0)  # code still work for multiple solution
                            raw_prop = OrderedDict([(tuple_v_chosen, {c_id: values[go_t][go_c] for go_c, c_id in enumerate(tuple_v_chosen)}) 
                                                    for go_t, tuple_v_chosen in enumerate(tuple_v_chosen_cmb)])
                            raw_like, raw_criterion = None, None   # TODO *->None, what influence the downstream
                    else:  # skip minor results
                        values = np.average(self.vp_unique_results[tuple_v_chosen_cmb]["values"], axis=0)  # code still work for multiple solution
                        raw_prop = OrderedDict([(tuple_v_chosen, {c_id: values[go_t][go_c] for go_c, c_id in enumerate(tuple_v_chosen)}) 
                                                for go_t, tuple_v_chosen in enumerate(tuple_v_chosen_cmb)])
                        raw_like, raw_criterion = None, None   # TODO *->None, what influence the downstream

                chosen_dict["raw_prop"] = raw_prop
                chosen_dict["raw_like"] = raw_like
                chosen_dict["raw_criterion"] = raw_criterion
                # reset the best result to be the best supported by replicates (go_s == 0) given other conditions
                # mcmc (if requested) will be based on the new best result
                if go_s == 0 and tuple_v_chosen_cmb != raw_res and chosen_dict["support"] > 0.05:
                    self._variant_proportions_cmb_best = cache_new_best
        logger.info("Num of unique replicates: %d" % len(self.vp_unique_results))

        # discarded: reverse model selection with single solution
        # # if there are replicates, summarize the replicates
        # logger.info("Summarizing the replicates ..")
        # if self.variant_proportions_reps:
        #     num_reps = len(self.variant_proportions_reps)
        #     self.vp_unique_results = {}
        #     self.__sorting_cid()  # use a universal cid_sorter to keep them in order across bootstraps
        #     for go_r, v_prop in enumerate(self.variant_proportions_reps):
        #         # sort the res by proportion decreasingly (-x[1]), then c_id (x[0])
        #         sorted_res = sorted(list(v_prop.items()), key=lambda x: self._cid_sorter[x[0]])
        #         tuple_v_chosen, tuple_v_props = zip(*sorted_res)
        #         if tuple_v_chosen not in self.vp_unique_results:
        #             self.vp_unique_results[tuple_v_chosen] = {"rep_ids": [], "support": None, "values": []}
        #         self.vp_unique_results[tuple_v_chosen]["rep_ids"].append(go_r)
        #         self.vp_unique_results[tuple_v_chosen]["values"].append(tuple_v_props)
        #     # calculate supports
        #     for res_info in self.vp_unique_results.values():
        #         support = len(res_info["rep_ids"]) / float(num_reps)
        #         res_info["support"] = support
        #     # sort by supports AND whether concordant with the estimate from raw dataset
        #     raw_res = tuple(sorted(self.variant_proportions.keys(), key=lambda x: (self._cid_sorter.get(x, 0), x)))
        #     self.vp_unique_results_sorted = \
        #         sorted(self.vp_unique_results, key=lambda x: (-self.vp_unique_results[x]["support"], x != raw_res, x))

        #     # use the entire dataset to recalculate loglike and criterion
        #     # for these results with bootstrap larger than 0.1 (arbitrarily)
        #     if self.bs_eligible and \
        #             (len(self.vp_unique_results_sorted) > 1 or self.vp_unique_results_sorted[0] != raw_res):
        #         logger.info("Re-evaluate the supported models using whole dataset ..")
        #     sbp_to_sbp_id = self.update_sp_to_sp_id_dict(self.all_sub_paths)
        #     for go_s, tuple_v_chosen in enumerate(self.vp_unique_results_sorted):
        #         chosen_dict = self.vp_unique_results[tuple_v_chosen]
        #         if tuple_v_chosen == raw_res:
        #             raw_prop, raw_like, raw_criterion = \
        #                 self.variant_proportions, self.res_loglike, self.res_criterion
        #         else:
        #             # re-evaluate only when bs is eligible
        #             if self.bs_eligible:
        #                 if chosen_dict["support"] > 0.05 or go_s == 0:  # the first one
        #                     raw_prop, raw_like, raw_criterion = self.fit_model_using_point_maximum_likelihood(
        #                         model=self.model,
        #                         sbp_to_sbp_id=sbp_to_sbp_id,
        #                         criterion=self.kwargs.get("model_criterion", Criterion.BIC),
        #                         chosen_ids=set(tuple_v_chosen),
        #                         init_self_max_like=False)
        #                 else:
        #                     values = np.average(self.vp_unique_results[tuple_v_chosen]["values"], axis=0)
        #                     raw_prop = {c_id: values[go_c] for go_c, c_id in enumerate(tuple_v_chosen)}
        #                     raw_like, raw_criterion = "*", "*"
        #             else:  # skip minor results
        #                 values = np.average(self.vp_unique_results[tuple_v_chosen]["values"], axis=0)
        #                 raw_prop = {c_id: values[go_c] for go_c, c_id in enumerate(tuple_v_chosen)}
        #                 raw_like, raw_criterion = "*", "*"
        #         chosen_dict["raw_prop"] = raw_prop
        #         chosen_dict["raw_like"] = raw_like
        #         chosen_dict["raw_criterion"] = raw_criterion
        #         # reset the best result to be the best supported by replicates (go_s == 0) given other conditions
        #         # mcmc (if requested) will be based on the new best result
        #         if go_s == 0 and tuple_v_chosen != raw_res and chosen_dict["support"] > 0.05:
        #             self.variant_proportions_best = raw_prop

    def __sorting_cid(self):
        self._cid_sorter = {}
        # compatible with multiple-solution res
        for go_r, best_model in enumerate(self._variant_proportions_cmb_reps):
            for sorted_var_ids, (v_prop, *foo) in sorted(best_model.items(), key=lambda x: x[0]):
                # sort the res by proportion decreasingly (-x[1]), then c_id (x[0])
                sorted_res = sorted(list(v_prop.items()), key=lambda x: (-x[1], x[0]))
                tuple_v_chosen, tuple_v_props = zip(*sorted_res)
                for cid_ in tuple_v_chosen:
                    if cid_ not in self._cid_sorter:
                        self._cid_sorter[cid_] = len(self._cid_sorter)
        for sorted_var_ids, (v_prop, *foo) in sorted(self._variant_proportions_cmb.items(), key=lambda x: x[0]):
            # sort the res by proportion decreasingly (-x[1]), then c_id (x[0])
            sorted_res = sorted(list(v_prop.items()), key=lambda x: (-x[1], x[0]))
            tuple_v_chosen, tuple_v_props = zip(*sorted_res)
            for cid_ in tuple_v_chosen:
                if cid_ not in self._cid_sorter:
                    self._cid_sorter[cid_] = len(self._cid_sorter)
        # compatible with single-solution res
        # for go_r, v_prop in enumerate(self.variant_proportions_reps):
        #     # sort the res by proportion decreasingly (-x[1]), then c_id (x[0])
        #     sorted_res = sorted(list(v_prop.items()), key=lambda x: (-x[1], x[0]))
        #     tuple_v_chosen, tuple_v_props = zip(*sorted_res)
        #     for cid_ in tuple_v_chosen:
        #         if cid_ not in self._cid_sorter:
        #             self._cid_sorter[cid_] = len(self._cid_sorter)

    def _check_bs_threshold(self, count_unique, n_reps, threshold, last_v_tuple):
        """
        check if bootstrap is possible to converge to a single solution with the threshold value
        """
        # code applicable to both 'reverse model selection' and 'genetic algorithm search'
        if last_v_tuple not in count_unique:
            count_unique[last_v_tuple] = 0
        count_unique[last_v_tuple] += 1
        counts = count_unique.values()
        current_max = max(counts)
        remaining = n_reps - sum(counts)
        if float(current_max + remaining) / n_reps < threshold:
            return False
        else:
            return True

    def output_pangenome_graph(self):
        """
        # TODO can be improved later
        # TODO, isolated as an independent module
        """
        # prepare sorted output
        out_paths = [self.variant_paths[cid]
                     for b_m in self._variant_proportions_cmb_best
                     for rpr_cid in b_m
                     for cid in self.repr_to_merged_variants[rpr_cid]]
        out_path_prop = [bm_values[0][rpr_cid] / float(len(self.repr_to_merged_variants[rpr_cid]))
                         for b_m, bm_values in self._variant_proportions_cmb_best.items()
                         for rpr_cid in b_m
                         for cid in self.repr_to_merged_variants[rpr_cid]]
        out_fids = [self._cid_to_fid[cid]
                    for b_m in self._variant_proportions_cmb_best
                    for rpr_cid in b_m
                    for cid in self.repr_to_merged_variants[rpr_cid]]
        sort_indices = np.argsort(out_fids)
        out_paths = [out_paths[idx] for idx in sort_indices]
        out_path_prop = OrderedDict([(new_idx, out_path_prop[old_idx]) for new_idx, old_idx in enumerate(sort_indices)])
        out_fids = [out_fids[idx] for idx in sort_indices]
        ##
        pangenome = PanGenome(
            original_graph=self.graph,
            variant_paths_sorted=out_paths,
            variant_props_ordered=out_path_prop,
            variant_labels=out_fids)
        logger.info("Constructing the pangenome ..")
        pangenome.gen_raw_pan_graph()
        out_gfa = os.path.join(self.outdir, "pangenome.gfa")
        pangenome.pan_graph.write_to_gfa(out_gfa)
        logger.info("Pangenome graph written to %s" % os.path.relpath(out_gfa))
        # logger.info("Pangenome graph written to %s" % os.path.join(self.outdir, "pangenome.gfa"))

    def output_result_info(self):
        final_res_file = os.path.join(self.outdir, "final.result.tab")
        with open(final_res_file, "w") as output_h:
            criterion_type = self.kwargs.get("model_criterion", Criterion.BIC).value
            met_str = "BOOTSTRAP"  # if self.kwargs.get("bootstrap", 0) else "JACKKNIFE"
            num_fids = len(self._outcome_rep_cid_to_fids)
            output_h.write(f"SOLUTIONS\t{met_str}_SUPPORT\tLOGLIKELIHOOD\t{criterion_type}\t" +
                           "\t".join(["&".join([f"FID_{_fid}" for _fid in fid_ls])
                                      for fid_ls in self._outcome_rep_cid_to_fids.values()]) +
                           f"\tFREQ_STD_OF_{met_str}\n")  # TODO add mcmc range column here and below
            # sort result to the same as the replicates  
            raw_res = tuple(self._variant_proportions_cmb_best.keys())  # no need to resort
            if raw_res in self.vp_unique_results:
                num_solutions = len(self.vp_unique_results)
            else:
                num_solutions = len(self.vp_unique_results) + 1
            s_digit = len(str(num_solutions))
            # sub_s_digit is the number of digits for maximum equivalent solutions
            max_equiv_sl = len(raw_res)
            for tuple_cmb_chosen in self.vp_unique_results:
                max_equiv_sl = max(max_equiv_sl, len(tuple_cmb_chosen))
            sub_s_digit = len(str(max_equiv_sl))
            # output the results
            for go_s, tuple_cmb_chosen in enumerate(self.vp_unique_results_sorted):
                chosen_cmb_dict = self.vp_unique_results[tuple_cmb_chosen]
                value_cmb_replicates = chosen_cmb_dict['values']
                this_support = chosen_cmb_dict['support']
                raw_like = chosen_cmb_dict.get('raw_like', None)
                raw_criterion = chosen_cmb_dict.get('raw_criterion', None)
                raw_prop = chosen_cmb_dict.get('raw_prop', None)
                for go_sub_s, tuple_v_chosen in enumerate(tuple_cmb_chosen):
                    if len(value_cmb_replicates) > 1:  # if there are replicates
                        replicate_std = np.std([val_reps[go_sub_s] for val_reps in value_cmb_replicates], 
                                               axis=0, ddof=1)  # ddof=1 for a sample taken full
                        replicate_std_str = ",".join([f"{std_:.4f}" for std_ in replicate_std])
                    else:
                        replicate_std_str = "*"   # ",".join(["*" for foo_ in value_cmb_replicates[0][go_sub_s]])
                    this_line = [f"{go_s + 1:0{s_digit}d}/{go_sub_s + 1:0{sub_s_digit}d}"
                                 f"\t{this_support}"
                                 f"\t{raw_like[go_sub_s] if raw_like else '*'}"
                                 f"\t{raw_criterion[go_sub_s] if raw_criterion else '*'}"] + \
                                ["-" for _id in range(num_fids)] + \
                                [replicate_std_str]
                    if raw_prop is None:  # low bootstrap support ones without re-assessment
                        for cid in tuple_v_chosen:
                            this_line[self._repr_cid_to_fid_column[cid]] = "*"
                    else:
                        for cid, prop_val in raw_prop[tuple_v_chosen].items():
                            this_line[self._repr_cid_to_fid_column[cid]] = f"{prop_val:.4f}"  # given that fid is 1-based
                    output_h.write("\t".join(this_line) + "\n")
            if not self.vp_unique_results_sorted:
                for go_sub_s, tuple_v_chosen in enumerate(raw_res):
                    this_line = [f"{num_solutions:0{s_digit}d}/{go_sub_s + 1:0{sub_s_digit}d}"
                                 f"\t-"
                                 f"\t-"
                                 f"\t-"] + \
                                ["-" for _id in range(num_fids)] + \
                                ["-"]
                    for cid, prop_val in self._variant_proportions_cmb_best[tuple_v_chosen][0].items():
                        this_line[self._repr_cid_to_fid_column[cid]] = f"{prop_val:.4f}"  # given that fid is 1-based
                    output_h.write("\t".join(this_line) + "\n")
            elif raw_res not in self.vp_unique_results:  # if there is no support for the raw-dataset-based best result
                for go_sub_s, tuple_v_chosen in enumerate(self._variant_proportions_cmb):
                    this_line = [f"{num_solutions:0{s_digit}d}/{go_sub_s + 1:0{sub_s_digit}d}"
                                 f"\t0"
                                 f"\t{self._variant_proportions_cmb[tuple_v_chosen][2]}"
                                 f"\t{self._variant_proportions_cmb[tuple_v_chosen][3]}"] + \
                                ["-" for _id in range(num_fids)] + \
                                ["-"]
                    for cid, prop_val in self._variant_proportions_cmb[tuple_v_chosen][0].items():
                        this_line[self._repr_cid_to_fid_column[cid]] = f"{prop_val:.4f}"  # given that fid is 1-based
                    output_h.write("\t".join(this_line) + "\n")
        logger.info(f"The summary of final result written to {os.path.relpath(final_res_file)}")

    def output_sampling_info(self):
        """
        output bootstrap results
        """
        # TODO fid_4 is always skipped

        if self._variant_proportions_cmb_reps:
            met_str = "bootstraps"   # if self.kwargs.get("bootstrap", 0) else "jackknife"
            replicates_file = os.path.join(self.outdir, f"{met_str}.replicates.tab")
            with open(replicates_file, "w") as output_h:
                num_fids = len(self._outcome_rep_cid_to_fids)
                output_h.write(f"{met_str.upper()}_ID/SID\t" +
                               "\t".join(["&".join([f"FID_{_fid}" for _fid in fid_ls])
                                          for fid_ls in self._outcome_rep_cid_to_fids.values()]) +
                               "\n")
                s_digit = len(str(len(self._variant_proportions_cmb_reps)))
                for go_bs, best_models in enumerate(self._variant_proportions_cmb_reps):
                    for go_solution, best_m_tuple in enumerate(best_models):
                        v_prop, *foo = best_models[best_m_tuple]
                        this_line = [f"{met_str}_{go_bs + 1:0{s_digit}d}/{go_solution + 1}"] + \
                                    ["-" for _id in range(num_fids)]
                        for cid, prop_val in v_prop.items():
                            this_line[self._repr_cid_to_fid_column[cid]] = f"{prop_val:.4f}"  # fid is 1-based
                        output_h.write("\t".join(this_line) + "\n")
            logger.info(f"Bootstrap replicates written to {os.path.relpath(replicates_file)}")

    def generate_fid(self):
        self._cid_to_fid = OrderedDict()  # cid to fid
        self._outcome_rep_cid_to_fids = OrderedDict()  # repr_cid to fid list
        self._repr_cid_to_fid_column = OrderedDict()  # repr_cid to fid column index
        count_fid = 0
        count_fid_column = 0
        # find candidate path ids chosen in all replicates and export them as final path ids
        # TODO: this is currently not sorting the fid by proportion but by cid
        for tuple_v_prop_cmb in self.vp_unique_results_sorted:
            for tuple_v_prop in tuple_v_prop_cmb:
                for cid in tuple_v_prop:
                    count_fid, count_fid_column = self.__index_cid_to_fid(cid, count_fid, count_fid_column)
        # if the result from raw dataset does not show in any of the replicates, which is bad but possible
        for cid_tuple in self._variant_proportions_cmb:
            for cid in cid_tuple:
                count_fid, count_fid_column = self.__index_cid_to_fid(cid, count_fid, count_fid_column)

    def __index_cid_to_fid(self, cid, count_fid, count_fid_column):
        if cid not in self._repr_cid_to_fid_column:
            count_fid_column += 1
            self._repr_cid_to_fid_column[cid] = count_fid_column  # 1-based
        if cid in self.repr_to_merged_variants:
            cid_list = self.repr_to_merged_variants[cid]
        else:
            cid_list = [cid]
        fid_ls = []
        for cid_ in cid_list:
            if cid_ not in self._cid_to_fid:
                count_fid += 1  # fid is therefore 1-based
                self._cid_to_fid[cid_] = count_fid
                fid_ls.append(count_fid)
        if cid not in self._outcome_rep_cid_to_fids:
            self._outcome_rep_cid_to_fids[cid] = fid_ls
        return count_fid, count_fid_column

    def output_variant_info(self):
        variants_info_file = os.path.join(self.outdir, "variants.info.tab")
        with open(variants_info_file, "w") as output_v_h:
            output_v_h.write("FID\tCID\tUnidentifiable_to_cid\tCONTIGS\tBASES\tPATH\n")
            for cid, fid in self._cid_to_fid.items():
                uid = self.be_unidentifiable_to.get(cid, cid)
                uid = "-" if uid == cid else uid
                this_vp = self.variant_paths[cid]
                this_size = self.variant_sizes[cid]
                output_v_h.write(f"{fid}\t{cid}\t{uid}\t{len(this_vp)}\t{this_size}\t{path_to_gaf_str(this_vp)}\n")
        logger.info(f"Variants info written to {os.path.relpath(variants_info_file)}")

    def output_readpath_info(self):
        # output readpath information
        read_path_info_file = os.path.join(self.outdir, "readpath.info.tab")
        with open(read_path_info_file, "w") as output_r_h:
            fid_part = "\t".join(["&".join([f"fid_{_fid}" for _fid in fid_ls])
                                  for fid_ls in self._outcome_rep_cid_to_fids.values()])
            output_r_h.write(f"rp_id\t{fid_part}\tpath\tnum_reads\n")
            for go_rp, (this_path, record_ids) in enumerate(self.read_paths.items()):
                output_r_h.write(f"{go_rp}\t")
                # occurence per variant (OPV)
                for cid in self._outcome_rep_cid_to_fids:
                    this_vp = self.variant_paths[cid]
                    output_r_h.write(f"{self.variant_readpath_counters[this_vp].get(this_path, 0)}\t")
                # other information
                output_r_h.write(f"{path_to_gaf_str(this_path)}\t{len(record_ids)}\n")
        logger.info(f"Readpath info written to {os.path.relpath(read_path_info_file)}")
        #
        readpath_record_ids_file = os.path.join(self.outdir, "readpath.record_ids.tab")
        with open(readpath_record_ids_file, "w") as output_rr_h:
            output_rr_h.write("rp_id\trecord_id\n")
            for go_rp, record_ids in enumerate(self.read_paths.values()):
                output_rr_h.write(f"{go_rp}\t{','.join([str(_r_id) for _r_id in record_ids])}\n")
        logger.info(f"Readpath record ids written to {os.path.relpath(readpath_record_ids_file)}")
        #
        num_used = len(self.read_paths)
        readpath_unused_info_file = os.path.join(self.outdir, "readpath.unused.info.tab")
        with open(readpath_unused_info_file, "w") as output_ru_h:
            fid_part = "\t".join([f"fid_{_fid}"
                                  for _cid, _fid in self._cid_to_fid.items()])
            output_ru_h.write(f"rp_id\t{fid_part}\tpath\tnum_reads\n")
            unused_read_paths = OrderedDict([(this_rp, count_r)
                                             for this_rp, count_r in self._raw_read_paths.items()
                                             if this_rp not in self.read_paths])
            all_subpath_generator = VariantSubPathsGenerator(
                graph=self.graph,
                min_alignment_len=self._raw_min_alignment_length,
                max_alignment_len=self._raw_max_alignment_length,
                read_paths_hashed=unused_read_paths)
            f_variant_rp_counter = {}
            for cid, fid in self._cid_to_fid.items():
                this_vp = self.variant_paths[cid]
                f_variant_rp_counter[fid] = all_subpath_generator.gen_subpaths(this_vp)
            for go_rp, (this_rp, count_r) in enumerate(unused_read_paths.items()):
                output_ru_h.write(f"{go_rp + num_used}\t")
                for cid, fid in self._cid_to_fid.items():
                    output_ru_h.write(f"{f_variant_rp_counter[fid].get(this_rp, 0)}\t")
                output_ru_h.write(f"{path_to_gaf_str(this_rp)}\t{count_r}\n")
        logger.info(f"Readpath-unused info written to {os.path.relpath(readpath_unused_info_file)}")

        # # output variant-readpath information
        # # rows by variants, columns by readpaths
        # with open(os.path.join(self.outdir, "readpaths.per.variant.tab"), "w") as output_vrc_h:
        #     output_vrc_h.write("FID\t" + "\t".join([f"rp_{_id}" for _id in range(len(self.read_paths))]) + "\n") # header
        #     for cid, fid in self.cid_to_fid.items():
        #         this_vp = self.variant_paths[cid]
        #         this_rp_counts = []
        #         for this_rp in self.read_paths:
        #             this_rp_counts.append(self.variant_readpath_counters[this_vp].get(this_rp, 0))
        #         output_vrc_h.write(f"{fid}\t" + "\t".join([str(count) for count in this_rp_counts]) + "\n")
        #     output_vrc_h.write("(Num_reads)\t" + "\t".join([str(len(self.read_paths[this_rp])) for this_rp in self.read_paths]) + "\n")


    def purge_graph_by_depth(
            self,
            depth_threshold: float = 0.001):
        to_remove = []
        threshold = depth_threshold * self.estimate_graph_average_depth()
        for v_name in self.graph.vertex_info:
            # TODO here the coverage is from the graph
            if self.graph.vertex_info[v_name].cov < threshold:  
                to_remove.append(v_name)
        if to_remove:
            self.graph.remove_vertex(to_remove)
            return True
        else:
            return False

    def prune_terminal_contigs(self):
        """
        Remove terminal contigs with that have no or only one neighbor"""
        modified = False
        while True:
            to_remove = []
            for v_name in self.graph.vertex_info:
                if self.graph.vertex_info[v_name].is_terminal():
                    to_remove.append(v_name)
            if to_remove:
                self.graph.remove_vertex(to_remove)
                modified = True
            else:
                break
        return modified


    def estimate_graph_average_depth(self):
        # TODO can be improved
        graph_len = sum(self.graph.vertex_info[v_].len for v_ in self.graph.vertex_info)
        total_counts = sum(self.graph.vertex_info[v_].cov * self.graph.vertex_info[v_].len
                           for v_ in self.graph.vertex_info)
        return float(total_counts) / graph_len

    def load_user_paths(self):
        added_paths = set()
        count_fixed = 0
        count_total = 0
        if self.var_fixed_f:
            for var_path in user_paths_reader(self.var_fixed_f):
                if self.graph.contain_path(var_path):
                    # TODO: update the code if circular become an attribute later
                    # var_path = self.graph.get_standardized_path(var_path)
                    var_path = self.graph.get_standardized_path_circ(self.graph.roll_path(var_path))
                    if var_path not in added_paths:
                        self.user_variant_paths.append(var_path)
                        self.user_variant_fixed_ids.add(count_fixed)
                        count_fixed += 1
                else:
                    logger.warning(f"{count_total + 1}th variant in {self.var_fixed_f} is incompatible with the graph!")
                count_total += 1
        if self.var_candidate_f:
            count_2 = 0
            for var_path in user_paths_reader(self.var_candidate_f):
                count_2 += 1
                if self.graph.contain_path(var_path):
                    # TODO: update the code if circular become an attribute later
                    var_path = self.graph.get_standardized_path_circ(self.graph.roll_path(var_path))
                    if var_path not in added_paths:
                        self.user_variant_paths.append(var_path)
                else:
                    logger.warning(f"{count_2}th variant in {self.var_candidate_f} is incompatible with the graph!")
                count_total += 1
        if len(self.user_variant_paths):
            logger.info(
                f"{len(self.user_variant_paths)} unique valid variants "
                f"out of {count_total} user assigned variants loaded.")

    def generate_read_paths(self, graph_alignment, filter_by_graph=True, min_alignment_counts=1):
        """
        Two-step filtering
        """
        # filter 1
        raw_lengths = []
        if filter_by_graph:
            for go_record, record in enumerate(graph_alignment.raw_records):
                this_path = self.graph.get_standardized_path(record.path)
                if len(this_path) == 0:
                    logger.warning(f"Record {go_record} is empty")
                    continue
                if this_path not in self.read_paths:
                    if self.graph.contain_path(this_path):
                        self.read_paths[this_path] = [go_record]
                        raw_lengths.append(record.p_align_len)
                else:
                    self.read_paths[this_path].append(go_record)
                    raw_lengths.append(record.p_align_len)
        else:
            for go_record, record in enumerate(graph_alignment.raw_records):
                this_path = self.graph.get_standardized_path(record.path)
                if this_path not in self.read_paths:
                    self.read_paths[this_path] = []
                self.read_paths[this_path].append(go_record)
                raw_lengths.append(record.p_align_len)
        # self._raw_read_paths will only keep the counts
        for this_path in self.read_paths:
            self._raw_read_paths[this_path] = len(self.read_paths[this_path])
        self._raw_min_alignment_length = min(raw_lengths) if raw_lengths else 0
        self._raw_max_alignment_length = max(raw_lengths) if raw_lengths else 0

        # filter 2
        if min_alignment_counts > 1:
            # 2023-12-28 use longer read paths to support shorter ones
            # draft filtering
            shallow_candidates = {}
            for this_path in list(self.read_paths):
                if len(self.read_paths[this_path]) < min_alignment_counts:
                    # del self.read_paths[this_path]
                    shallow_candidates[this_path] = len(self.read_paths[this_path])
                    # disable masking, which was designed to apply min_align_counts only to model fitting
                    # self.read_paths_masked.add(this_path)
            # reduce shallow_candidates by keeping those with supports from longer reads alive
            sizes = {}
            # logger.debug(f"### {len(shallow_candidates)} candidate shallow ones")
            for r_path in shallow_candidates:
                if len(r_path) in sizes:
                    sizes[len(r_path)].add(r_path)
                else:
                    sizes[len(r_path)] = {r_path}
            while sizes:
                r_size, candidate_paths = sizes.popitem()
                for longer_path in self.read_paths:
                    longer_path_len = len(longer_path)
                    if longer_path_len > r_size:
                        for go_s in range(longer_path_len - r_size + 1):
                            sub_path = self.graph.get_standardized_path(longer_path[go_s: go_s + r_size])
                            if sub_path in shallow_candidates:
                                shallow_candidates[sub_path] += 1
                                if shallow_candidates[sub_path] >= min_alignment_counts:
                                    del shallow_candidates[sub_path]
                                    candidate_paths.discard(sub_path)
                                    logger.trace(f"### keep {sub_path}")
                    if not candidate_paths:
                        break
            for this_path in shallow_candidates:
                del self.read_paths[this_path]
            logger.debug(f"### remaining {len(shallow_candidates)} candidate shallow ones with supports")
        # check
        # if len(self.read_paths_masked) == len(self.read_paths):
        if not self.read_paths:
            logger.error("No valid alignment records remains after filtering! "
                         "Please reset the filtering parameters ('--min-align-*' flags) or check the input data.")
            raise SystemExit(1)

        # align_len_at_path = []
        # if filter_by_graph:
        #     for go_record, record in enumerate(graph_alignment.raw_records):
        #         this_path = self.graph.get_standardized_path(record.path)
        #         if len(this_path) == 0:
        #             logger.warning(f"Record {go_record} is empty")
        #             continue
        #         if this_path not in self.read_paths:
        #             if self.graph.contain_path(this_path):
        #                 self.read_paths[this_path] = [go_record]
        #                 align_len_at_path.append(record.p_align_len)
        #         else:
        #             self.read_paths[this_path].append(go_record)
        #             align_len_at_path.append(record.p_align_len)
        # else:
        #     for go_record, record in enumerate(graph_alignment.raw_records):
        #         this_path = self.graph.get_standardized_path(record.path)
        #         if this_path not in self.read_paths:
        #             self.read_paths[this_path] = []
        #         self.read_paths[this_path].append(go_record)
        #         align_len_at_path.append(record.p_align_len)
        # generate align_len_at_path_map  # will be used in bootstrap
        self.align_len_at_path_map = {}
        for records in self.read_paths.values():
            for go_record in records:
                self.align_len_at_path_map[go_record] = graph_alignment.raw_records[go_record].p_align_len
        self.num_valid_records = len(self.align_len_at_path_map)
        align_len_at_path_sorted = sorted(self.align_len_at_path_map.values())
        # store min/max value
        self.min_alignment_length = align_len_at_path_sorted[0]
        self.max_alignment_length = align_len_at_path_sorted[-1]
        self.max_read_path_size = self.get_max_read_path_size(self.read_paths)

    def get_max_read_path_size(self, paths):
        assert bool(self.read_paths), "empty read paths!"
        return max([len(rp) for rp in paths])

    def detect_alignment_abnormals(self, alignment):
        """Detect unusual breaks accumulated within a contig
        :param alignment: the alignment object
        :return: True if the alignment is regenerated
        """
        logger.info("Detecting abnormal vertices in the graph alignment")
        detect_conflict = GraphAlignConflicts(graph=self.graph, graph_alignment=alignment, output_dir=self.outdir)
        abnormal_vertices, max_loads = detect_conflict.detect()
        if abnormal_vertices:
            logger.info(f"Total number of windows: {detect_conflict.n_bins}")
            logger.info(f"Total number of conflicts: {detect_conflict.n_balls}")
            logger.info(f"Expected maximum conflicts per window: {detect_conflict.max_load}")
            if len(abnormal_vertices) > 20:
                info_line = f"Detected abnormal vertices (max=[{min(max_loads)}, {max(max_loads)}]): {', '.join([f'{v}(max={ld})' for v, ld in zip(abnormal_vertices[:20], max_loads[:20])])} ..."
            else:
                info_line = f"Detected abnormal vertices (max=[{min(max_loads)}, {max(max_loads)}]): {', '.join([f'{v}(max={ld})' for v, ld in zip(abnormal_vertices, max_loads)])}"
            if max(max_loads) <= 1:
                logger.info("No conflicts detected.")
            else:
                min_conflict_reads = self.kwargs.get("add_conflict_edges", 0)
                gmm_max_std = self.kwargs.get("gmm_max_std", 50.)
                if min_conflict_reads > 0:
                    if max(max_loads) >= max(min_conflict_reads, detect_conflict.max_load):
                        # modify the graph
                        new_graph = detect_conflict.modify_graph(min_n_reads=min_conflict_reads, gmm_max_std=gmm_max_std)
                        if new_graph is None:
                            logger.info(info_line)
                            logger.info(f"No consensus conflicts above max({min_conflict_reads}, {detect_conflict.max_load}) detected, ignored.")
                        else:
                            logger.info(info_line)
                            logger.debug(f"Conflict edges with minimum {max(min_conflict_reads, detect_conflict.max_load)} reads assessed ..")
                            new_graph_f = os.path.join(self.outdir, "tmp.add_conflicts.gfa")
                            # logger.info(f"Output modified graph: {new_graph_f}")
                            logger.info("Output modified graph: tmp.add_conflicts.gfa")
                            new_graph.write_to_gfa(new_graph_f)
                            self.graph_file = new_graph_f
                            self.graph = new_graph
                            if self.reads_file:
                                # redo the alignment
                                logger.info("Redoing the alignment with the modified graph ..")
                                self.alignment_file = os.path.join(self.outdir, "alignment_add_conflicts.gaf")
                                run_graph_aligner(
                                    graph_file=self.graph_file,
                                    seq_file=self.reads_file,
                                    alignment_file=self.alignment_file,
                                    num_processes=self.num_processes,
                                    other_params=self.kwargs.get("graph_aligner_params", ""))
                                return True
                            else:
                                logger.error("No reads file provided! Cannot redo the alignment.")
                                logger.error("Please regenerate the assembly graph, "
                                            "or redo the alignment using the modified graph, "
                                            "or add '--ignore-conflicts' to skip.")
                                raise SystemExit(1)
                    else:
                        logger.info(info_line)
                        logger.info(f"All windows have conflicts below max({min_conflict_reads}, {detect_conflict.max_load}), ignored.")
                else:
                    if max(max_loads) <= 1:
                        logger.info("No conflicts detected.")
                    else:
                        logger.error(info_line)
                        logger.error("Please regenerate the assembly graph, "
                                     "or add '--add-conflict-edges INT' to add conflict edges, "
                                     "or add '--ignore-conflicts' to skip.")
                        raise SystemExit(1)
        return False

    # def clean_graph(self, min_effective_count=10, ignore_ratio=0.001):
    #     """ deprecated for now"""
    #     clean_graph_obj = CleanGraph(self)
    #     clean_graph_obj.run(min_effective_count=min_effective_count, ignore_ratio=ignore_ratio)

    # def get_align_len_dist(self, graph_alignment):
    #     """
    #     Get sorted alignment lengths, optionally save to file
    #     and store longest to self.
    #     """
    #     logger.debug("Summarizing alignment length distribution")
    #
    #     # get sorted alignment lengths
    #     self.align_len_at_path_sorted = sorted([rec.p_align_len for rec in graph_alignment.raw_records])
    #
    #     # optionally save temp files
    #     if self.keep_temp:
    #         opath = os.path.join(self.outdir, "align_len_at_path_sorted.txt")
    #         with open(opath, "w") as out:
    #             out.write("\n".join(map(str, self.align_len_at_path_sorted)))
    #
    #     # store max value
    #     self.min_alignment_len = self.align_len_at_path_sorted[0]
    #     self.max_alignment_len = self.align_len_at_path_sorted[-1]
    #
    #     # report result
    #     logger.info(
    #         "Alignment length range at path: [{}, {}]".format(self.min_alignment_len, self.max_alignment_len))

    def prune_unaligned_contigs(
            self):
        drop_contigs = set(self.graph.vertex_info)
        for read_p in self.read_paths:
            for contig_n, contig_e in read_p:
                if contig_n in drop_contigs:
                    drop_contigs.discard(contig_n)
            if not drop_contigs:
                break
        if drop_contigs:
            self.graph.remove_vertex(drop_contigs)

    def estimate_contig_coverages_from_read_paths(self):
        """
        Counting the contig coverage using the occurrences in the read paths.
        Note: this will proportionally overestimate the coverage values comparing to base coverage values,
        """
        # TODO: alignment details (path len) can be used
        # use minimum value of 1e-8 to avoid zero total weights
        contig_coverages = OrderedDict([(v_name, 1e-8) for v_name in self.graph.vertex_info])
        for read_path, records in self.read_paths.items():
            n_rec = len(records)
            for v_name, v_end in read_path:
                if v_name in contig_coverages:
                    contig_coverages[v_name] += n_rec
        for v_name in self.graph.vertex_info:
            self.graph.vertex_info[v_name].cov = contig_coverages[v_name]

    def gen_candidate_variants(
            self,
            # path_generator="H",
            graph_based_multiplicity_wiggle=0.,
            penalty_for_size=1e-5,
            fix_shallowest_contig=0,
            # start_strategy="random",
            min_num_search=1000,
            max_num_search=10000,
            max_num_traversals=50000,
            max_uniq_traversal=200,
            max_uncover_ratio=0.001,
            num_processes=1,
            uni_chromosome=False,
            size_ratio=0.,
            **kwargs):
        """
        generate candidate variant paths from the graph
        """
        # if path_generator == "H":
        tmp_dir = fpath(self.outdir).joinpath("tmp.candidates")
        generator = VariantGenerator(
            traversome_obj=self,
            graph_based_multiplicity_wiggle=graph_based_multiplicity_wiggle,
            penalty_for_size=penalty_for_size,
            fix_shallowest_contig=fix_shallowest_contig,
            # start_strategy=start_strategy,
            min_num_valid_search=min_num_search,
            max_num_valid_search=max_num_search,
            max_num_traversals=max_num_traversals,
            max_uniq_traversal=max_uniq_traversal,
            max_uncover_ratio=max_uncover_ratio,
            num_processes=num_processes,
            force_circular=self.force_circular,
            uni_chromosome=uni_chromosome,
            decay_f=kwargs.get("search_decay_factor"),
            temp_dir=tmp_dir,
            use_alignment_cov=self.kwargs.get("use_alignment_cov", False),
            decompose_circular_unit=self.kwargs.get("decompose_circular_unit", True),
            resume=self.resume,
            )
        generator.generate_heuristic_paths()

        # 1. calculate the variant base lengths and store them to all_variant_sizes
        # 2. calculate the minimum cutoff for variant size
        all_variant_sizes = []
        for go_variant in self.user_variant_paths:
            all_variant_sizes.append(self.graph.get_path_length(go_variant, check_valid=False, adjust_for_cyclic=True))
        for go_variant in generator.variants:
            all_variant_sizes.append(self.graph.get_path_length(go_variant, check_valid=False, adjust_for_cyclic=True))
        min_size = max(all_variant_sizes) * size_ratio

        # TODO add self.user_variant_paths during heuristic search instead of here
        self.variant_paths = []
        # 1. record the sid to cid table for debugging
        # 2. add candidate variants to self.variant_paths
        # 3. calculate variant lengths and store them to self.variant_sizes
        # TODO resume/overwrite mode
        with open(tmp_dir.joinpath("sid.to.cid.tab"), "a") as sid_to_cid_h:
            self.variant_sizes = []
            self.variant_topos = []
            cid = 0
            for go_p, go_variant in enumerate(self.user_variant_paths):
                v_size = all_variant_sizes.pop(0)
                if go_p in self.user_variant_fixed_ids or v_size >= min_size:
                    self.variant_paths.append(go_variant)
                    self.variant_sizes.append(v_size)
                    self.variant_topos.append(self.graph.is_circular_path(go_variant))
                    sid_to_cid_h.write(f"user_{go_p}\tcid_{cid}\n")
                    cid += 1
                else:
                    logger.info(f"user_{go_p} discarded due to size filtering (see '--v-len'). ")

            user_v_p_set = set(self.user_variant_paths)
            for go_v, go_variant in enumerate(generator.variants):
                if go_variant not in user_v_p_set:
                    v_size = all_variant_sizes.pop(0)
                    if v_size >= min_size:
                        self.variant_paths.append(go_variant)
                        self.variant_sizes.append(v_size)
                        self.variant_topos.append(self.graph.is_circular_path(go_variant))
                        sid_to_cid_h.write(f"sid_{go_v}\tcid_{cid}\n")
                        cid += 1
                    else:
                        # TODO change to debug
                        logger.info(f"sid_{go_v} discarded due to size filtering (see '--v-len'). ")
                else:
                    logger.info(f"searched candidate variant {go_v} existed in user designed.")

        # self.variant_paths_sorted = self.user_variant_paths + generator.variants
        # self.variant_paths_sorted = generator.variants

    # def update_params_for_variants(self):
    #     self.variant_sizes = [self.graph.get_path_length(variant_p, check_valid=False, adjust_for_cyclic=True)
    #                           for variant_p in self.variant_paths_sorted]
    #     for go_p, path in enumerate(self.variant_paths_sorted):
    #         logger.debug("PATH{}: {}".format(go_p, self.graph.repr_path(path)))
    #
    #     # generate subpaths: the binomial sets
    #     if self.num_put_variants > 1:
    #         logger.info("Generating sub-paths ..")
    #         self.gen_all_informative_sub_paths()
    #         self.generate_sub_path_stats()
    #         # build an index
    #         self.update_sp_to_sp_id_dict()
    #         # TODO: check the number
    #         logger.info("Generated {} valid informative sub-paths".format(len(self.all_sub_paths)))
    #     elif self.num_put_variants == 0:
    #         logger.warning("No valid configuration found for the input assembly graph.")
    #     else:
    #         # self.repr_to_merged_variants = OrderedDict([(0, [0])])
    #         logger.warning("Only one genomic configuration found for the input assembly graph.")

    def get_variant_sub_paths(self, variant_path):
        return self.subpath_generator.gen_subpaths(variant_path)

    def gen_all_informative_sub_paths(self, silent=False):
        """
        generate all sub paths and their occurrences for each candidate variant
        """
        # count sub path occurrences for each candidate variant and recorded in self.variant_readpath_counters
        # this_overlap = self.graph.uni_overlap()
        # self.variant_readpath_counters = OrderedDict()
        for this_var_p in self.variant_paths:
            # foo = self.get_variant_sub_paths(this_var_p)
            self.variant_readpath_counters[this_var_p] = self.get_variant_sub_paths(this_var_p)
        # self.variant_readpath_counters should be only a subset of self.subpath_generator.variant_readpath_counters

        # create unidentifiable table
        # TODO we did not include it into the bootstrap??
        self.be_unidentifiable_to = OrderedDict()
        for represent_iso_id in range(self.num_put_variants):
            for check_iso_id in range(represent_iso_id, self.num_put_variants):
                # every id will be checked only once, either unidentifiable to a previous id or represent itself
                if check_iso_id not in self.be_unidentifiable_to:
                    if self.identifiable_by_unique_rp:
                        if check_iso_id == represent_iso_id:  # represent itself
                            self.be_unidentifiable_to[check_iso_id] = check_iso_id
                        elif set(self.variant_readpath_counters[self.variant_paths[check_iso_id]]) == \
                                set(self.variant_readpath_counters[self.variant_paths[represent_iso_id]]):
                            # if the same unique read path are shared by two variants
                            self.be_unidentifiable_to[check_iso_id] = represent_iso_id
                    else:
                        if check_iso_id == represent_iso_id:  # represent itself
                            self.be_unidentifiable_to[check_iso_id] = check_iso_id
                        elif self.variant_readpath_counters[self.variant_paths[check_iso_id]] == \
                                self.variant_readpath_counters[self.variant_paths[represent_iso_id]] and \
                                self.variant_sizes[check_iso_id] == self.variant_sizes[represent_iso_id]:
                            # if the same read path, same read path counts, and size are shared by two variants
                            self.be_unidentifiable_to[check_iso_id] = represent_iso_id
        # logger.info(str(self.be_unidentifiable_to))
        self.repr_to_merged_variants = \
            OrderedDict([(rps_id, []) for rps_id in sorted(set(self.be_unidentifiable_to.values()))])
        for check_iso_id, rps_iso_id in self.be_unidentifiable_to.items():
            self.repr_to_merged_variants[rps_iso_id].append(check_iso_id)
        # sort the merged variants to find the one with the smallest path size as the new representative
        new_repr_to_merged_variants = OrderedDict()
        for rps_id, unidentifiable_ids in self.repr_to_merged_variants.items():
            new_rps = sorted(unidentifiable_ids, key=lambda x: (self.variant_sizes[x], self.variant_paths[x]))[0]
            new_repr_to_merged_variants[new_rps] = unidentifiable_ids
        self.repr_to_merged_variants = new_repr_to_merged_variants
        # 
        for rps_id, unidentifiable_ids in self.repr_to_merged_variants.items():
            if len(unidentifiable_ids) > 1:
                logger.warning(f"Mutually unidentifiable paths in current alignment: {[str(_x) if _x != rps_id else str(_x)+'*' for _x in unidentifiable_ids]}")

        # transform self.variant_readpath_counters to self.all_sub_paths
        self.all_sub_paths = OrderedDict()
        for go_variant, variant_path in enumerate(self.variant_paths):
            sub_paths_group = self.variant_readpath_counters[variant_path]
            for this_sub_path, this_sub_count in sub_paths_group.items():
                if this_sub_path not in self.all_sub_paths:
                    self.all_sub_paths[this_sub_path] = spi = SubPathInfo()
                    spi.inner_len = self.graph.get_path_internal_length(this_sub_path)
                    # # read_paths with overlaps should be and were already trimmed,
                    # so we should proceed without overlaps
                    # external_len_without_overlap = self.graph.get_path_len_without_terminal_overlaps(this_sub_path)
                    # 2023-03-09: get_path_len_without_terminal_overlaps -> get_path_length
                    # Because the end of the alignment can still stretch to the overlapped region
                    # and will not be recorded in the path.
                    # Thus, the start point can be the path length not the uni_overlap-trimmed one.
                    spi.full_len = self.graph.get_path_length(
                        this_sub_path, check_valid=False, adjust_for_cyclic=False)
                self.all_sub_paths[this_sub_path].from_variants[go_variant] = this_sub_count


        # shared sub-paths (with the same counts) are also informative because their frequencies may vary
        #     upon the change of averaged genome size after the change of proportions
        # WRONG: to simplify downstream calculation, remove shared sub-paths (with same counts) shared by all variants
        # for this_sub_path, this_sub_path_info in list(self.all_sub_paths.items()):
        #     if len(this_sub_path_info.from_variants) == self.num_put_variants and \
        #             len(set(this_sub_path_info.from_variants.values())) == 1:
        #         for variant_p, sub_paths_group in self.variant_readpath_counters.items():
        #             del sub_paths_group[this_sub_path]
        #         del self.all_sub_paths[this_sub_path]

        if not self.all_sub_paths:
            logger.error("No valid subpath found!")
            # raise SystemExit(0)
            # sys.exit(1)
            return

        # match graph alignments to all_sub_paths
        # this info will also be used in bootstrap
        for read_path, record_ids in self.read_paths.items():
            if read_path in self.all_sub_paths:
                self.all_sub_paths[read_path].mapped_records = record_ids
            # else:  # DEBUG
            #     for go_variant, variant_path in enumerate(self.variant_paths_sorted):
            #         sub_paths_group = self.variant_readpath_counters[variant_path]
            #         if read_path in sub_paths_group:
            #             print("found", len(record_ids), read_path)
            #             break
            #     else:
            #         print("lost", len(record_ids), read_path)
        if not silent:
            logger.info(f"Generated {len(self.all_sub_paths)} informative sub-paths based on "
                        f"{sum([len(sbp.mapped_records) for sbp in self.all_sub_paths.values()])} records in total")

    @staticmethod
    def update_sp_to_sp_id_dict(all_sub_paths):
        sbp_to_sbp_id = {}
        for go_sp, this_sub_path in enumerate(all_sub_paths):
            sbp_to_sbp_id[this_sub_path] = go_sp
        return sbp_to_sbp_id

    def generate_multinomial_bin_stats(self, all_sub_paths, rec_id_sorted_by_len, align_len_at_path_sorted, quiet=True):
        if quiet:
            logger.trace("Generating multinomial bin statistics ..")
        else:
            logger.debug("Generating multinomial bin statistics ..")
        # generate number of multinomial distributions according to the overlap between read path length ranges
        align_len_id_lookup_table = \
            self._generate_align_len_id_lookup_table(
                align_len_at_path_sorted=align_len_at_path_sorted,
                min_alignment_length=align_len_at_path_sorted[0],
                max_alignment_length=align_len_at_path_sorted[-1])
        ranges = self._identify_bins(all_sub_paths=all_sub_paths, align_len_at_path_sorted=align_len_at_path_sorted)
        bins_list = []
        max_id = len(align_len_at_path_sorted) - 1
        # logger.debug(f"align_len_at_path_sorted: {align_len_at_path_sorted}")
        # logger.debug(f"align_len_id_lookup_table: {align_len_id_lookup_table}")
        for min_len, max_len in ranges:
            left_id, right_id = self._get_id_range_in_increasing_lengths(
                min_len=min_len,
                max_len=max_len,
                align_len_id_lookup_table=align_len_id_lookup_table,
                max_id=max_id)
            #
            # if min_len == 32636 and max_len == 32673:
            #     print("check id range ===== ")
            #     print(left_id, right_id)
            #     print(align_len_at_path_sorted[left_id], align_len_at_path_sorted[right_id])

            if left_id > right_id:
                # no read found within this scope: shouldn't happen because self._identify_bins has already handled this
                logger.warning(f"Remove range {min_len} {max_len}")
            else:
                # TODO: min_len and max_len seems to be of no use and can be deleted after debugging
                bins_list.append(Bins(min_len=min_len, max_len=max_len, min_id=left_id, max_id=right_id))
        if quiet:
            logger.trace(f"Total # Multinomial Ranges: {len(bins_list)}")
        else:
            logger.info(f"Total # Multinomial Ranges: {len(bins_list)}")
        # index len id to read_path
        r_id_to_sp = {}
        for this_sub_path, this_sub_path_info in list(all_sub_paths.items()):
            for r_id in this_sub_path_info.mapped_records:
                r_id_to_sp[r_id] = this_sub_path
        if quiet:
            logger.trace(f"from id_{min(r_id_to_sp)} to id_{max(r_id_to_sp)}: {len(r_id_to_sp)} records")
        else:
            logger.debug(f"from id_{min(r_id_to_sp)} to id_{max(r_id_to_sp)}: {len(r_id_to_sp)} records")
        # each rp_bins contain read_path & alignment information to be modeled as multinomial distribution
        count_rp_bins = 0
        for bins in bins_list:
            rp_bins = {}
            # logger.debug(f"from id_{bins.min_id} to id_{bins.max_id}")
            for sort_l_id in range(bins.min_id, bins.max_id + 1):
                r_id = rec_id_sorted_by_len[sort_l_id]
                if r_id in r_id_to_sp:
                    this_sub_path = r_id_to_sp[r_id]
                    rp_bins.setdefault(this_sub_path, BinInfo()).num_matched += 1
            # logger.debug(f"num_matched: {[bininfo.num_matched for bininfo in rp_bins.values()]}")
            for this_sub_path, bininfo in rp_bins.items():
                bininfo.from_variants = self.all_sub_paths[this_sub_path].from_variants
                bininfo.num_possible_X = self.get_averaged_possible_alignment_start_points(
                    align_len_at_path_sorted=align_len_at_path_sorted,
                    min_id=bins.min_id,
                    max_id=bins.max_id,
                    this_sub_path=this_sub_path)
                count_rp_bins += 1
            bins.rp_bins = list(rp_bins.values())  # no sorting should be fine

            # for go_rp, rp in enumerate(tests):
            #     if rp in rp_bins:
            #         print(go_rp,
            #               rp_bins[rp].num_matched,
            #               rp_bins[rp].num_possible_X,
            #               rp_bins[rp].from_variants)

        if quiet:
            logger.trace(f"Total # Bins: {count_rp_bins}")
        else:
            logger.info(f"Total # Bins: {count_rp_bins}")
        return bins_list

    @staticmethod
    def _identify_bins(all_sub_paths, align_len_at_path_sorted):
        # 1. identify the points and point types (0: min&max, 1:min, 2:max)
        points_to_sb = {}
        for this_sub_path, this_sub_path_info in list(all_sub_paths.items()):
            min_len = this_sub_path_info.inner_len + 2 if len(this_sub_path) > 1 else 1
            max_len = this_sub_path_info.full_len

            # count_print += 1
            # if this_sub_path in tests:
            #     print(tests.index(this_sub_path), min_len, max_len,
            #           [self.align_len_at_path_map[_id] for _id in this_sub_path_info.mapped_records])
            # else:
            #     if count_print < 30:
            #         print("-", min_len, max_len,
            #               [self.align_len_at_path_map[_id] for _id in this_sub_path_info.mapped_records])

            if min_len > max_len:
                del all_sub_paths[this_sub_path]  # will be weird
                logger.warning(f"deleting illegal path: {path_to_gaf_str(this_sub_path)}")
            elif min_len == max_len:
                points_to_sb.setdefault(min_len, {})[this_sub_path] = 1  # 1 means both min and max
            else:
                points_to_sb.setdefault(min_len, {})[this_sub_path] = 0  # 0 means min
                points_to_sb.setdefault(max_len, {})[this_sub_path] = 2  # 2 means max

        # 2. identify the ranges split by overlaps among read path length ranges,
        #    each range has the potential to be modeled as multinomial distribution
        ranges = []
        last_type = False  # indicating the last action was a start (True) or end (False) of a range
        sorted_points = sorted(points_to_sb)
        for go_p, point in enumerate(sorted_points):
            margin_info = points_to_sb[point]
            is_start_point = any(point_type < 2 for point_type in margin_info.values())
            is_end_point = any(point_type > 0 for point_type in margin_info.values())
            if is_start_point:
                if ranges and last_type:
                    ranges[-1][1] = point - 1
                ranges.append([point, None])
                last_type = True
            if is_end_point:
                # Always update the max_len of the latest bin since an end point is found
                ranges[-1][-1] = point
                last_type = False
                # Prepare for a possible new range
                next_point_index = go_p + 1
                if next_point_index < len(sorted_points):
                    next_point = sorted_points[next_point_index]
                    if not (next_point == point + 1 and is_start_point):
                        # Initiate a new range if the next point does not immediately continue
                        ranges.append([point + 1, None])
                        last_type = True

        # print(f"len ranges= {len(ranges)}")
        # with open(os.path.join(self.outdir, "old.ranges.txt"), "w") as output_h:
        #     for from_p, to_p in ranges:
        #         output_h.write(f"{from_p}\t{to_p}\n")

        # 3. filter ranges to only keep those with observations
        new_ranges = []
        i, j = 0, 0
        while i < len(ranges) and j < len(align_len_at_path_sorted):
            # Check if the current range in ranges and the current len in align_len_at_path_sorted overlap
            if ranges[i][1] >= align_len_at_path_sorted[j] >= ranges[i][0]:
                new_ranges.append(ranges[i])
                i += 1  # Move to the next range since ranges is gapless
            elif align_len_at_path_sorted[j] > ranges[i][1]:
                i += 1  # Current len in align_len_at_path_sorted is beyond the current range, move to the next range
            else:
                j += 1  # Current len in align_len_at_path_sorted is before the current range, move to the next integer

        # print(f"len new_ranges= {len(new_ranges)}")
        # with open(os.path.join(self.outdir, "new.ranges.txt"), "w") as output_h:
        #     for from_p, to_p in new_ranges:
        #         output_h.write(f"{from_p}\t{to_p}\n")

        return new_ranges
        # bins_list = []
        # active_read_paths = set()
        # last_type = False  # True: start, False: end
        # sorted_points = sorted(points_to_sb)
        # for go_p, point in enumerate(sorted_points):
        #     margin_info = points_to_sb[point]
        #     start_points = {path for path, point_type in margin_info.items() if point_type < 2}
        #     end_points = {path for path, point_type in margin_info.items() if point_type > 0}
        #     # Process start points
        #     if start_points:
        #         active_read_paths.update(start_points)
        #         if bins_list and last_type:
        #             bins_list[-1].max_len = point - 1
        #         bins_list.append(Bins(min_len=point, rp_bins=list(active_read_paths)))
        #         last_type = True
        #         # Process end points
        #     if end_points:
        #         if not last_type or not bins_list or bins_list[-1].max_len is None:
        #             bins_list[-1].max_len = point
        #         active_read_paths.difference_update(end_points)
        #         last_type = False
        #         next_point_index = go_p + 1
        #         if next_point_index < len(sorted_points):
        #             next_point = sorted_points[next_point_index]
        #             if not (next_point == point + 1 and start_points):
        #                 if active_read_paths:
        #                     bins_list.append(Bins(min_len=point + 1, rp_bins=list(active_read_paths)))
        #                     last_type = True

    def get_averaged_possible_alignment_start_points(
            self, align_len_at_path_sorted, min_id, max_id, this_sub_path):
        """
        Combining start points from multiple alignments
            by calling self.get_num_of_possible_alignment_start_points
        """
        n_records = max_id - min_id + 1
        if not n_records:
            return 0
        # print_it = this_sub_path in tests
        # if print_it:
        #     print(tests.index(this_sub_path), "id range", min_id, max_id,
        #           self.all_sub_paths[this_sub_path].inner_len, self.all_sub_paths[this_sub_path].full_len)
        num_possible_Xs = {}
        align_len_id = min_id
        while align_len_id <= max_id:
            this_len = align_len_at_path_sorted[align_len_id]
            this_x = self.get_num_of_possible_alignment_start_points(
                read_len_aligned=this_len, this_sub_path=this_sub_path)
            if this_len not in num_possible_Xs:
                num_possible_Xs[this_len] = 0
            num_possible_Xs[this_len] += this_x
            align_len_id += 1
            # each alignment record will be counted once, if the next align_len_id has the same length
            while align_len_id <= max_id and align_len_at_path_sorted[align_len_id] == this_len:
                num_possible_Xs[this_len] += this_x
                align_len_id += 1
        # sum_Xs is the numerator for generating the distribution rate,
        # with the denominator approximating the genome size
        return sum(num_possible_Xs.values()) / float(n_records)

    def get_num_of_possible_alignment_start_points(self, read_len_aligned, this_sub_path):
        r"""
        If a read with certain length could be aligned to a path (size>=2),
        calculate how many possible start points could this alignment happen.

        Example:
        ----------------------------------------
        |      \                               |
        |     b \          e          / a      |
        |        \___________________/         |
        |        /                   \         |
        |     c /                     \ d      |
        |      /                       \       |
        |     /                         \      |
        |                                \     |
        ----------------------------------------
        for graph(a=2,b=3,c=4,d=5,e=6), if read has length of 11 and be aligned to b->e->d,
        then there could be 3 possible alignment start points

        If a read with certain length could be aligned to a path (size==1),
            the start points can be simply calculated as max_len - read_len_aligned + 1

        :param read_len_aligned:
        :param this_sub_path:
        :return:
        """
        input_tuple = (read_len_aligned, this_sub_path)
        if input_tuple in self._cache_num_ali_starts:
            return self._cache_num_ali_starts[input_tuple]
        #
        if len(this_sub_path) > 1:
            # TODO cache two ends to speed up calculation
            # prepare the end stats for the path
            left_n1, left_e1 = this_sub_path[0]
            left_n2, left_e2 = this_sub_path[1]
            left_info = self.graph.vertex_info[left_n1]
            left_1_len = left_info.len
            left_12_overlap = left_info.connections[left_e1][(left_n2, not left_e2)]
            right_n1, right_e1 = this_sub_path[-1]
            right_n2, right_e2 = this_sub_path[-2]
            right_info = self.graph.vertex_info[right_n1]
            right_1_len = right_info.len
            right_12_overlap = right_info.connections[not right_e1][(right_n2, right_e2)]
            # when a, b, c, d is longer than the read_len_aligned,
            # the result is the maximum_num_cat without trimming
            maximum_num_cat = read_len_aligned - (self.all_sub_paths[this_sub_path].inner_len + 2) + 1
            # the number of starts cannot be longer than either end
            # trim left
            left_trim = max(maximum_num_cat - (left_1_len - left_12_overlap), 0)
            # trim right
            right_trim = max(maximum_num_cat - (right_1_len - right_12_overlap), 0)
            # result
            self._cache_num_ali_starts[input_tuple] = maximum_num_cat - left_trim - right_trim
            return self._cache_num_ali_starts[input_tuple]
        else:
            self._cache_num_ali_starts[input_tuple] = self.all_sub_paths[this_sub_path].full_len - read_len_aligned + 1
            return self._cache_num_ali_starts[input_tuple]

    def generate_sub_path_stats(self, all_sub_paths, align_len_at_path_sorted):
        """
        """
        logger.debug("Generating sub-path statistics ..")
        align_len_id_lookup_table = \
            self._generate_align_len_id_lookup_table(
                align_len_at_path_sorted=align_len_at_path_sorted,
                min_alignment_length=align_len_at_path_sorted[0],
                max_alignment_length=align_len_at_path_sorted[-1])

        if self.num_processes == 1:
            for this_sub_path, this_sub_path_info in list(all_sub_paths.items()):
                num_in_range, sum_Xs = self.__subpath_info_filler(
                    this_sub_path=this_sub_path,
                    align_len_at_path_sorted=align_len_at_path_sorted,
                    align_len_id_lookup_table=align_len_id_lookup_table,
                    all_sub_paths=all_sub_paths)
                if num_in_range:
                    this_sub_path_info.num_in_range = num_in_range
                    this_sub_path_info.num_possible_X = sum_Xs / num_in_range
                    this_sub_path_info.num_matched = len(this_sub_path_info.mapped_records)
                else:
                    del all_sub_paths[this_sub_path]
        else:
            # TODO multiprocess
            # it took 2 minutes 1,511 read paths represented by 177,869 records
            # manager = Manager()
            # pool_obj = Pool(processes=self.num_processes)
            for this_sub_path, this_sub_path_info in list(all_sub_paths.items()):
                num_in_range, sum_Xs = self.__subpath_info_filler(
                    this_sub_path=this_sub_path,
                    align_len_at_path_sorted=align_len_at_path_sorted,
                    align_len_id_lookup_table=align_len_id_lookup_table,
                    all_sub_paths=all_sub_paths)
                if num_in_range:
                    this_sub_path_info.num_in_range = num_in_range
                    this_sub_path_info.num_possible_X = sum_Xs / num_in_range
                    this_sub_path_info.num_matched = len(this_sub_path_info.mapped_records)
                else:
                    del all_sub_paths[this_sub_path]

    def _prepare_for_sampling(self):
        """
        Generate records pool in the form of alignment records map to read path ids
        """
        logger.info("Generate records pool for sampling ..")
        self.records_pool_to_sbp_ids = {}
        for go_sbp, sp_info in enumerate(self.all_sub_paths.values()):
            for record_id in sp_info.mapped_records:
                if record_id not in self.records_pool_to_sbp_ids:
                    self.records_pool_to_sbp_ids[record_id] = go_sbp
                else:
                    raise ValueError(f"{record_id} in self.records_pool_to_sbp_ids!")
        self.records_pool_sorted = sorted(self.records_pool_to_sbp_ids)

    def sample_sub_paths(
            self,
            bootstrap_size=None,
            # jackknife_size=None,
            masking=None):
        """
        According to sampling strategies and masking set, sample aligned records to generate
        1) a new set of sub_paths;
        2) sorted alignment length distribution
        """
        masking = set() if masking is None else masking
        if not self.records_pool_to_sbp_ids:
            self._prepare_for_sampling()
        if bootstrap_size:
            # TODO move the info to the run() function
            # logger.debug("Using bootstrap.")
            new_records_pool = self.random.choices(self.records_pool_sorted, k=bootstrap_size)
            if self.kwargs.get("augmented_bootstrap", False):
                # additionally add the original records to augmented bootstrap and avoid underrepresentation
                new_records_pool = self.records_pool_sorted + new_records_pool
        # elif jackknife_size:
        #     logger.debug(f"Using Jackknife: leave-{jackknife_size}-out.")
        #     keep_ids = self.random.sample(range(self.num_valid_records), k=self.num_valid_records - jackknife_size)
        #     new_records_pool = [self.records_pool_sorted[s_id_] for s_id_ in keep_ids]
        else:
            # only do filtering using self.read_paths_masked
            new_records_pool = list(self.records_pool_sorted)
        # cluster new pool
        sbp_id_to_rec_id = {}
        for rec_id in new_records_pool:
            sbp_id = self.records_pool_to_sbp_ids[rec_id]
            if sbp_id not in sbp_id_to_rec_id:
                sbp_id_to_rec_id[sbp_id] = [rec_id]
            else:
                sbp_id_to_rec_id[sbp_id].append(rec_id)
        # create new all_sub_paths
        new_sub_paths = OrderedDict()
        masking_rec_ids = set()
        for go_sbp, (read_path, sbp_info) in enumerate(self.all_sub_paths.items()):
            if go_sbp in sbp_id_to_rec_id:  # if it is sampled subpaths
                if read_path not in masking:
                    new_sbp_info = SubPathInfo()
                    # main info to be copied from previous sbp_info
                    new_sbp_info.from_variants = sbp_info.from_variants
                    # ignore new_sbp_info.mapped_records, which is useless in following analysis
                    new_sbp_info.mapped_records = sbp_id_to_rec_id[go_sbp]
                    #
                    new_sbp_info.inner_len = sbp_info.inner_len
                    new_sbp_info.full_len = sbp_info.full_len
                    # The X in binomial:
                    # theoretical num of matched chances per data, to be filled in self.generate_sub_path_stats
                    # new_sbp_info.num_possible_X = sbp_info.num_possible_X
                    # The n in binomial: observed num of reads in range, to be filled in self.generate_sub_path_stats
                    # new_sbp_info.num_in_range = sbp_info.num_in_range
                    # The x in binomial: observed num of matched reads to be filled in self.generate_sub_path_stats
                    # len(sbp_id_to_rec_id[go_sbp])
                    # new_sbp_info.num_matched = len(sbp_id_to_rec_id[go_sbp])
                    new_sub_paths[read_path] = new_sbp_info
                else:
                    for rec_id in sbp_id_to_rec_id[go_sbp]:
                        masking_rec_ids.add(rec_id)
        # generate new align_len_at_path_sorted
        rec_id_sorted_by_len, align_len_at_path_sorted = \
            zip(*sorted([(rec_id, self.align_len_at_path_map[rec_id])
                          for rec_id in new_records_pool if rec_id not in masking_rec_ids],
                        key=lambda x: (x[1], x[0])))  # sort by length then by rec id
        return new_sub_paths, rec_id_sorted_by_len, align_len_at_path_sorted

    def __subpath_info_filler(
            self, this_sub_path, align_len_at_path_sorted, align_len_id_lookup_table, all_sub_paths):
        # internal_len = self.graph.get_path_internal_length(this_sub_path)
        # # # read_paths with overlaps should be and were already trimmed, so we should proceed without overlaps
        # # external_len_without_overlap = self.graph.get_path_len_without_terminal_overlaps(this_sub_path)
        # # 2023-03-09: get_path_len_without_terminal_overlaps -> get_path_length
        # # Because the end of the alignment can still stretch to the overlapped region
        # # and will not be recorded in the path.
        # # Thus, the start point can be the path length not the uni_overlap-trimmed one.
        # external_len = self.graph.get_path_length(this_sub_path, check_valid=False, adjust_for_cyclic=False)
        min_valid_len = all_sub_paths[this_sub_path].inner_len + 2
        max_valid_len = all_sub_paths[this_sub_path].full_len
        #
        left_id, right_id = self._get_id_range_in_increasing_lengths(
            min_len=min_valid_len,
            max_len=max_valid_len,
            align_len_id_lookup_table=align_len_id_lookup_table,
            max_id=len(align_len_at_path_sorted) - 1)
        if left_id > right_id:
            # no read found within this scope
            logger.trace("Remove {} after pruning contig overlap ..".format(this_sub_path))
            del all_sub_paths[this_sub_path]
            return 0, 0

        if len(this_sub_path) > 1:
            # prepare the end stats for the path
            left_n1, left_e1 = this_sub_path[0]
            left_n2, left_e2 = this_sub_path[1]
            left_info = self.graph.vertex_info[left_n1]
            left_1_len = left_info.len
            left_12_overlap = left_info.connections[left_e1][(left_n2, not left_e2)]
            right_n1, right_e1 = this_sub_path[-1]
            right_n2, right_e2 = this_sub_path[-2]
            right_info = self.graph.vertex_info[right_n1]
            right_1_len = right_info.len
            right_12_overlap = right_info.connections[not right_e1][(right_n2, right_e2)]

            def get_num_of_possible_alignment_start_points(read_len_aligned):
                r"""
                If a read with certain length could be aligned to a path (size>=2),
                calculate how many possible start points could this alignment happen.

                Example:
                ----------------------------------------
                |      \                               |
                |     b \          e          / a      |
                |        \___________________/         |
                |        /                   \         |
                |     c /                     \ d      |
                |      /                       \       |
                |     /                         \      |
                |                                \     |
                ----------------------------------------
                for graph(a=2,b=3,c=4,d=5,e=6), if read has length of 11 and be aligned to b->e->d,
                then there could be 3 possible alignment start points

                :param read_len_aligned:
                :return:
                """
                # when a, b, c, d is longer than the read_len_aligned,
                # the result is the maximum_num_cat without trimming
                maximum_num_cat = read_len_aligned - min_valid_len
                # trim left
                left_trim = max(maximum_num_cat - left_1_len - left_12_overlap, 0)
                # trim right
                right_trim = max(maximum_num_cat - right_1_len - right_12_overlap, 0)
                return maximum_num_cat - left_trim - right_trim
        else:
            def get_num_of_possible_alignment_start_points(read_len_aligned):
                """
                If a read with certain length could be aligned to a path (size==1),
                the start points can be simply calculated as follows
                """
                return max_valid_len - read_len_aligned + 1

        num_possible_Xs = {}
        align_len_id = left_id
        while align_len_id <= right_id:
            this_len = align_len_at_path_sorted[align_len_id]
            this_x = get_num_of_possible_alignment_start_points(read_len_aligned=this_len)
            if this_len not in num_possible_Xs:
                num_possible_Xs[this_len] = 0
            num_possible_Xs[this_len] += this_x
            align_len_id += 1
            # each alignment record will be counted once, if the next align_len_id has the same length
            while align_len_id <= right_id and align_len_at_path_sorted[align_len_id] == this_len:
                num_possible_Xs[this_len] += this_x
                align_len_id += 1
        # num_in_range is the total number of alignments (observations) in range
        num_in_range = right_id + 1 - left_id
        # sum_Xs is the numerator for generating the distribution rate,
        # with the denominator approximating the genome size
        sum_Xs = sum(num_possible_Xs.values())

        return num_in_range, sum_Xs

    @staticmethod
    def _generate_align_len_id_lookup_table(
            align_len_at_path_sorted,
            min_alignment_length,
            max_alignment_length):
        """
        called by generate_sub_path_stats
        to speed up self.__get_id_range_in_increasing_lengths
        """
        its_left_id = 0
        its_right_id = 0
        max_id = len(align_len_at_path_sorted) - 1
        align_len_id_lookup_table = \
            {potential_len: {"as_left_lim_id": None, "as_right_lim_id": None}
             for potential_len in range(min_alignment_length, max_alignment_length + 1)}
        for potential_len in range(min_alignment_length, max_alignment_length + 1):
            if potential_len == align_len_at_path_sorted[its_right_id]:
                align_len_id_lookup_table[potential_len]["as_left_lim_id"] = its_left_id = its_right_id
                while potential_len == align_len_at_path_sorted[its_right_id]:
                    align_len_id_lookup_table[potential_len]["as_right_lim_id"] = its_right_id
                    if its_right_id == max_id:
                        break
                    else:
                        its_left_id = its_right_id
                        its_right_id += 1
            else:
                align_len_id_lookup_table[potential_len]["as_left_lim_id"] = its_right_id
                align_len_id_lookup_table[potential_len]["as_right_lim_id"] = its_left_id
        return align_len_id_lookup_table

    @staticmethod
    def _get_id_range_in_increasing_lengths(min_len, max_len, align_len_id_lookup_table, max_id):
        """Given a range of length (min_len, max_len), return the len_id of them,
        which helps quickly count the number of reads and number of possible matches given a subpath

        called by self.__subpath_info_filler.
        replace get_id_range_in_increasing_values func in utils.py
        """
        left_id = align_len_id_lookup_table.get(min_len, {}).\
            get("as_left_lim_id", 0)
        right_id = align_len_id_lookup_table.get(max_len, {}).\
            get("as_right_lim_id", max_id)
        return left_id, right_id

    def get_multinomial_like_formula(self, variant_percents, log_func, within_variant_ids: Set = None):
        self.model.get_like_formula(variant_percents, log_func, within_variant_ids)

    def fit_model_using_point_maximum_likelihood(self,
                                                 model,
                                                 sbp_to_sbp_id,
                                                 criterion=Criterion.BIC,
                                                 chosen_ids: Union[typingODict[int, bool], Set] = None,
                                                 init_self_max_like: bool = True):
        max_like_fit = ModelFitMaxLike(
            model=model,
            variant_paths=self.variant_paths,
            variant_readpath_counters=self.variant_readpath_counters,
            sbp_to_sbp_id=sbp_to_sbp_id,
            repr_to_merged_variants=self.repr_to_merged_variants,
            be_unidentifiable_to=self.be_unidentifiable_to,
            logfile=self.logfile,
            loglevel=self.loglevel)
        if init_self_max_like:
            self.max_like_fit = max_like_fit
        use_prop, echo_prop, this_like, this_criterion =\
            self.max_like_fit.point_estimate(chosen_ids=chosen_ids, criterion=criterion)
        return use_prop, echo_prop, this_like, this_criterion

    def fit_model_using_genetic_algorithm(
            self,
            model,
            sbp_to_sbp_id,
            criterion=Criterion.BIC,
            chosen_ids: Union[typingODict[int, bool], Set] = None,
            init_self_max_like: bool = True,
            bootstrap_str: str = ""):
        """
        :param model: the model to be fitted
        :param sbp_to_sbp_id: used to access all read paths are covered
        :param criterion: the criterion to be used for model selection
        :param chosen_ids: ids of variants to be used in the model  # actually not used
        :param init_self_max_like: whether to add the max_like_fit to self.max_like_fit
        :param bootstrap_str: turn on to only print simple information and mark the bootstrap id
        :return: the best variant proportions"""
        from traversome.ModelFitMaxLike import ModelFitMaxLike
        max_like_fit = ModelFitMaxLike(
            model=model,
            variant_paths=self.variant_paths,
            variant_readpath_counters=self.variant_readpath_counters,
            sbp_to_sbp_id=sbp_to_sbp_id,
            repr_to_merged_variants=self.repr_to_merged_variants,
            be_unidentifiable_to=self.be_unidentifiable_to,
            loglevel=self.loglevel,
            logfile=self.logfile,
            bootstrap_mode=bootstrap_str)
        if init_self_max_like:
            self.max_like_fit = max_like_fit
        return max_like_fit.genetic_algorithm_search(
            n_proc=self.num_processes,
            criterion=criterion,
            chosen_ids=chosen_ids,
            user_fixed_ids=self.user_variant_fixed_ids,
            population_size=self.kwargs.get("ga_pop_size", 300),
            max_generations=self.kwargs.get("ga_max_gen", 200),
            crossover_prob=self.kwargs.get("ga_cross_prob", 0.8),
            mutation_prob=self.kwargs.get("ga_mut_prob", 0.05),
            tournament_size=self.kwargs.get("ga_tour_size", 3),
            num_elites=self.kwargs.get("ga_num_elites", 2),
            patience=self.kwargs.get("ga_patience", 5))

    def fit_model_using_reverse_model_selection(self,
                                                model,
                                                sbp_to_sbp_id,
                                                criterion=Criterion.BIC,
                                                chosen_ids: Union[typingODict[int, bool], Set] = None,
                                                init_self_max_like: bool = True,
                                                bootstrap_str: str = "",
                                                num_processes: int = 1,
                                                event = None):
        """
        :param sbp_to_sbp_id: used to access all read paths are covered
        :param bootstrap_str: turn on to only print simple information and mark the bootstrap id
        :param event: Manager.Event, if provided along with num_processes > 1,
            the event will be used to check if the subprocess should stop
        """
        # intermediate level of RES is not working properly in different environments
        # if bootstrap_str and logger.level(self.loglevel).no >= 20:  # not in {"TRACE", "DEBUG"}:
        #     new_log_level = "RES"  # set the level to be higher
        #     setup_logger(loglevel=new_log_level, timed=True, log_file=self.logfile)
        # else:
        #     new_log_level = self.loglevel
        # run fit
        max_like_fit = ModelFitMaxLike(
            model=model,
            variant_paths=self.variant_paths,
            variant_readpath_counters=self.variant_readpath_counters,
            sbp_to_sbp_id=sbp_to_sbp_id,
            repr_to_merged_variants=self.repr_to_merged_variants,
            be_unidentifiable_to=self.be_unidentifiable_to,
            loglevel=self.loglevel,
            logfile=self.logfile,
            bootstrap_mode=bootstrap_str)
        if init_self_max_like:
            self.max_like_fit = max_like_fit
        return max_like_fit.reverse_model_selection(
            n_proc=num_processes,
            criterion=criterion,
            chosen_ids=chosen_ids,
            user_fixed_ids=self.user_variant_fixed_ids,
            max_queue_size=self.kwargs.get("rms_max_queue_size", None),
            max_end_hits=self.kwargs.get("rms_max_end_hits", None),
            max_unchanged=self.kwargs.get("rms_max_unchanged", None),
            parent_event=event)

    # def update_candidate_info(self, ext_component_proportions: typingODict[int, float] = None):
    #     """
    #     ext_component_proportions: if provided, use external component proportions to update the candidate information
    #     """
    #     if ext_component_proportions:
    #         assert isinstance(ext_component_proportions, OrderedDict)
    #         self.variant_proportions = ext_component_proportions
    #
    #     self.variant_paths_sorted = []  # each element is a tuple(path)
    #     self.variant_sizes = []
    #     self.num_put_variants = None
    #     self.variant_readpath_counters = OrderedDict()  # each value is a dict(sub_path->sub_path_counts)
    #     self.all_sub_paths = OrderedDict()
    #     self.sbp_to_sbp_id = {}
    #     self.observed_sbp_id_set = set()
    #     #
    #     self.be_unidentifiable_to = {}
    #     # use the merged variants to represent each set of variants.
    #     # within each set the variants are unidentifiable to each other
    #     self.repr_to_merged_variants = {}
    #
    #     for

    def fit_model_using_bayesian_mcmc(self, chosen_ids: typingODict[int, bool] = None):
        from traversome.ModelFitBayesian import ModelFitBayesian
        self.bayesian_fit = ModelFitBayesian(self)
        return self.bayesian_fit.run_mcmc(
            self.kwargs.get("n_generations", 0), self.kwargs.get("n_burn", 0), chosen_ids=chosen_ids)

    def output_seqs(self):
        fid_cid_to_prob_unidentifiable = {}
        for solution_id, (_v_tuple, (prop_dict, *foo)) in enumerate(self._variant_proportions_cmb_best.items()):
            for cid, prob in prop_dict.items():
                if prob >= self.out_prob_threshold:
                    cid_list = self.repr_to_merged_variants[cid]
                    fid_list = tuple(sorted([self._cid_to_fid[each_cid] for each_cid in cid_list]))
                    for each_fid, each_cid in zip(fid_list, cid_list):
                        if (each_fid, each_cid) in fid_cid_to_prob_unidentifiable:
                            fid_cid_to_prob_unidentifiable[(each_fid, each_cid)].append([solution_id, prob, fid_list])
                        else:
                            fid_cid_to_prob_unidentifiable[(each_fid, each_cid)] = [[solution_id, prob, fid_list]]
        if not fid_cid_to_prob_unidentifiable:
            logger.warning("No variants with probability >= {} found, "
                           "no output will be generated.".format(self.out_prob_threshold))
            return
        sorted_fid_cid = sorted(fid_cid_to_prob_unidentifiable.keys())
        out_seq_num = len(sorted_fid_cid)
        out_digit = len(str(out_seq_num))
        for go_seq, (each_fid, each_cid) in enumerate(sorted_fid_cid):
            base_label = "FID_%0{}i".format(out_digit) % each_fid
            this_base_name = "variant.%0{}i".format(out_digit) % each_fid
            seq_file_name = os.path.join(self.outdir, this_base_name + ".fasta")
            sid_str = ",".join([str(sid + 1) for sid, _, _ in fid_cid_to_prob_unidentifiable[(each_fid, each_cid)]])
            freq_str = ",".join(["@S{}={:.4f}{}".format(sid + 1, prob,
                "" if len(fid_list) == 1 else f"({'+'.join(map(str, fid_list))})")
                for sid, prob, fid_list in fid_cid_to_prob_unidentifiable[(each_fid, each_cid)]])
            with open(seq_file_name, "w") as output_handler:
                this_seq = self.graph.export_path(self.variant_paths[each_cid], check_valid=False)
                this_len = len(this_seq.seq)
                seq_head = f">{base_label} len={this_len}bp path={this_seq.label} solution={sid_str} freq[{freq_str}]"
                output_handler.write(seq_head + "\n" + this_seq.seq + "\n")
                logger.log("RES", f"{this_base_name} len={this_len} freq[{freq_str}]")
        # logger.info("Output {} seqs (%.4f to %.{}f): ".format(count_seq, len(str(self.out_prob_threshold)) - 2)
        #             % (max(self.variant_proportions.values()), self.out_prob_threshold))
        file_wildcard = os.path.join(self.outdir, "variant.*.fasta")
        logger.info(f"Resulting {out_seq_num} seqs written to {os.path.relpath(file_wildcard)}")

    def compare_sub_path_counts(self, c_id_1, c_id_2):
        """
        Compare the sub-path counts between two candidate variants
        """
        pass

