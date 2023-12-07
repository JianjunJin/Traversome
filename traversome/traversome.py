#!/usr/bin/env python

"""
Top-level class for running the CLI
"""

import os
import sys
import random

from loguru import logger
from copy import deepcopy
from pathlib import Path as fpath
from collections import OrderedDict
from traversome.Assembly import Assembly
from traversome.PanGenome import PanGenome
from traversome.GraphAlignRecords import GraphAlignRecords
from traversome.utils import \
    SubPathInfo, Criterion, VariantSubPathsGenerator, executable, run_graph_aligner, user_paths_reader, setup_logger, \
    path_to_gaf_str
from traversome.ModelFitMaxLike import ModelFitMaxLike
from traversome.VariantGenerator import VariantGenerator
from traversome.ModelGenerator import PathMultinomialModel
# from traversome.CleanGraph import CleanGraph
from typing import OrderedDict as typingODict
from typing import Set, Union
from multiprocessing import Manager, Pool
import gc
import numpy as np
# import time


class Traversome(object):
    """
    keep_temp (bool):
        If True then alignment lengths are saved to a tmp file during
        the get_align_len_dist function call.
    """

    def __init__(
            self,
            graph,
            alignment,
            reads_file,
            outdir,
            var_fixed=None,
            var_candidate=None,
            num_processes=1,
            force_circular=True,
            uni_chromosome=False,
            out_prob_threshold=0.001,
            keep_temp=False,
            random_seed=12345,
            loglevel="DEBUG",
            resume=False,
            **kwargs):
        # store input files and params
        self.graph_gfa = graph
        self.alignment_file = alignment
        self.reads_file = reads_file
        self.outdir = outdir
        self.var_fixed_f = var_fixed
        self.var_candidate_f = var_candidate
        self.num_processes = num_processes
        self.force_circular = force_circular
        self.uni_chromosome = uni_chromosome
        self.out_prob_threshold = out_prob_threshold
        self.keep_temp = keep_temp
        self.resume = resume
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
        self.align_len_at_path_sorted = None
        self.max_alignment_length = None
        self.min_alignment_length = None
        self.read_paths = OrderedDict()
        self.read_paths_masked = set()
        self.max_read_path_size = None
        self.subpath_generator = None

        # variant model to be generated
        self.user_variant_paths = []
        self.user_variant_fixed_ids = set()
        self.variant_paths = []  # each element is a tuple(path)
        self.variant_sizes = []
        self.num_put_variants = None
        self.variant_subpath_counters = OrderedDict()  # variant -> dict(sub_path->sub_path_counts)
        self.all_sub_paths = OrderedDict()
        self.sbp_to_sbp_id = {}
        self.observed_sbp_id_set = set()
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
        self.variant_proportions = OrderedDict()
        self.res_loglike = None
        self.res_criterion = None
        self.variant_proportions_reps = []
        self.vp_unique_results = {}
        self.vp_unique_results_sorted = []
        self.cid_to_fid = OrderedDict()
        self.variant_proportions_best = OrderedDict()

        # self.pal_len_sbp_Xs = OrderedDict()
        # self.sbp_Xs = []

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
        self.graph = Assembly(self.graph_gfa)
        self.graph.update_vertex_clusters()
        # choose the component
        graph_component_selection = self.kwargs.get("graph_component_selection", 0)
        if isinstance(graph_component_selection, int) or isinstance(graph_component_selection, slice):
            self.graph.reduce_graph_by_weight(component_ids=graph_component_selection)
        elif isinstance(graph_component_selection, float):
            self.graph.reduce_graph_by_weight(cutoff_to_total=graph_component_selection)
        # merge graph if possible
        if self.kwargs.get("keep_graph_redundancy", False):
            pass
        else:
            if self.graph.merge_all_possible_vertices():
                if self.alignment_file:
                    # the graph was changed and cannot be traced back to the names in the alignment
                    # this can also be moved down considering structure, however more computational efficient to be here
                    raise Exception("Graph be simplified by merging all possible nodes ! "
                                    "Please provide the raw reads, or simplify the graph and redo the alignment,"
                                    "or use (NOT RECOMMENDED) '--keep-graph-redundancy' to skip!")
                else:
                    logger.info("  graph merged")
                    self.graph_gfa = os.path.join(self.outdir, "processed.gfa")
                    self.graph.write_to_gfa(self.graph_gfa)
            self.graph.update_vertex_clusters()
        logger.info("  #contigs in total: {}".format(len(self.graph.vertex_info)))
        logger.info("  #contigs in each component: {}".format(sorted([len(cls) for cls in self.graph.vertex_clusters])))

        self.load_user_paths()

        if not self.alignment_file:
            if self.resume and os.path.exists(os.path.join(self.outdir, "alignment.gaf")):
                self.alignment_file = os.path.join(self.outdir, "alignment.gaf")
            elif executable("GraphAligner"):  # TODO: add path option
                self.alignment_file = os.path.join(self.outdir, "alignment.gaf")
                # TODO: pass options to GraphAligner
                run_graph_aligner(
                    graph_file=self.graph_gfa,
                    seq_file=self.reads_file,
                    alignment_file=self.alignment_file,
                    num_processes=self.num_processes)
            else:
                raise Exception("GraphAligner not available or damaged!")

        alignment = GraphAlignRecords(
            self.alignment_file,
            min_align_len=self.kwargs.get("min_alignment_len_cutoff", 100),
            min_identity=self.kwargs.get("min_alignment_identity_cutoff", 0.85),
        )
        min_alignment_counts = self.kwargs.get("min_alignment_counts", 1)
        self.generate_read_paths(
            graph_alignment=alignment,
            filter_by_graph=True,
            min_alignment_counts=min_alignment_counts)
        logger.info("  #reads aligned: %i" % len(alignment.read_records))
        logger.info("  #records aligned: %i" % len(alignment.raw_records))
        logger.info("  #read paths: %i" % len(self.read_paths))
        # self.get_align_len_dist(graph_alignment=alignment)
        logger.info(
            "Alignment length range at path: [{}, {}]".format(self.min_alignment_length, self.max_alignment_length))
        logger.info("Alignment max size at path: {}".format(self.max_read_path_size))
        logger.info("  #read paths masked: %i" % len(self.read_paths_masked))
        # free memory to reduce the burden for potential downstream parallelization
        del alignment
        gc.collect()
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
            # self._update_params_for_variants()
        else:
            logger.debug("Generating candidate variant paths ...")
            self.gen_candidate_variants(
                # path_generator=path_gen_scheme,
                start_strategy=self.kwargs.get("search_start_scheme").value,
                search_decay_factor=self.kwargs.get("search_decay_factor"),
                min_num_search=self.kwargs.get("min_valid_search"),
                max_num_search=self.kwargs.get("max_valid_search"),
                max_num_traversals=self.kwargs.get("max_num_traversals"),
                max_uniq_traversal=self.kwargs.get("max_uniq_traversal"),
                num_processes=self.num_processes,
                uni_chromosome=self.uni_chromosome
            )

        self.num_put_variants = len(self.variant_paths)
        self.variant_sizes = [self.graph.get_path_length(variant_p, check_valid=False, adjust_for_cyclic=True)
                              for variant_p in self.variant_paths]
        if self.num_put_variants == 0:
            logger.error("No candidate variants found!")
            logger.info("======== VARIANTS SEARCHING ENDS ========\n")
            raise SystemExit(0)
        elif self.num_put_variants == 1 or len(self.repr_to_merged_variants) == 1:
            self.variant_proportions_best[0] = self.variant_proportions[0] = 1.
            logger.info("======== VARIANTS SEARCHING ENDS ========\n")
        else:
            logger.info("======== VARIANTS SEARCHING ENDS ========\n")
            logger.info("======== MODEL SELECTION & FITTING STARTS ========")
            for go_p, path in enumerate(self.variant_paths):
                logger.debug("PATH{}: {}".format(go_p, self.graph.repr_path(path)))

            logger.info("Generating sub-paths ..")
            self.gen_all_informative_sub_paths()
            # ONLY apply self.read_paths_masked to model selection and fitting
            sampled_sub_paths, align_len_at_path_sorted = \
                self.sample_sub_paths(masking=self.read_paths_masked)
            logger.info("Indexing {} valid informative sub-paths after masking ".format(len(sampled_sub_paths)))
            self.generate_sub_path_stats(sampled_sub_paths, align_len_at_path_sorted)
            # build an index
            sbp_to_sbp_id = self.update_sp_to_sp_id_dict(sampled_sub_paths)
            # difference between this number and total number of sub-paths
            #            will happen when the current variants cannot cover all read paths
            #                     or when an alignable path is not informative
            logger.debug("Estimating candidate variant frequencies using Maximum Likelihood...")
            self.model = PathMultinomialModel(variant_sizes=self.variant_sizes, all_sub_paths=sampled_sub_paths)
            self.variant_proportions, self.res_loglike, self.res_criterion = \
                self.fit_model_using_reverse_model_selection(
                    model=self.model,
                    sbp_to_sbp_id=sbp_to_sbp_id,
                    criterion=self.kwargs["model_criterion"])
            logger.info("======== MODEL SELECTION & FITTING ENDS ========\n")

            if self.kwargs.get("bootstrap", 0) or self.kwargs.get("jackknife", 0):
                logger.info("======== BOOTSTRAPPING STARTS ========")
                self.do_subsampling()
                logger.info("======== BOOTSTRAPPING ENDS ========\n")

            if not self.variant_proportions_best:  # if it is not modified during subsampling
                self.variant_proportions_best = deepcopy(self.variant_proportions)

            # update candidate info according to the result of reverse model selection
            # assure self.repr_to_merged_variants was generated
            if self.kwargs["n_generations"] > 0 and \
                    len([repr_v
                         for repr_v in self.variant_proportions_best
                         if repr_v in self.repr_to_merged_variants]) > 1:
                # TODO add mcmc result to the summary table
                logger.info("======== BAYESIAN ESTIMATION STARTS ========")
                logger.debug("Estimating candidate variant frequencies using Bayesian MCMC ...")
                self.variant_proportions_best = \
                    self.fit_model_using_bayesian_mcmc(chosen_ids=self.variant_proportions_best)
                logger.info("======== BAYESIAN ESTIMATION ENDS ========\n")

        logger.info("======== OUTPUT FILES STARTS ========")
        self.output_variant_info()
        self.output_sampling_info()
        self.output_result_info()
        self.output_pangenome_graph()
        self.output_seqs()
        logger.info("======== OUTPUT FILES ENDS ========\n")

    def do_subsampling(self):
        """
        using bootstrap or jackknife
        """
        # TODO, not urgent, parallelize bootstrap if necessary
        self._prepare_for_sampling()
        n_replicate = self.kwargs.get("bootstrap", 0) \
            if self.kwargs.get("bootstrap", 0) else self.kwargs.get("jackknife", 0)
        self.variant_proportions_reps = []
        n_digit = len(str(n_replicate))
        go_bs = 0
        while go_bs < n_replicate:
            logger.debug(f"Sampling {go_bs + 1} --------")
            logger.debug("Generating sub-paths ..")
            if self.kwargs.get("bootstrap", 0):
                sampled_sub_paths, align_len_at_path_sorted = \
                    self.sample_sub_paths(bootstrap_size=self.num_valid_records, masking=self.read_paths_masked)
            else:
                sampled_sub_paths, align_len_at_path_sorted = \
                    self.sample_sub_paths(
                        jackknife_size=int(self.num_valid_records / float(n_replicate)),
                        masking=self.read_paths_masked)
            logger.debug("Indexing {} valid informative sub-paths after masking ".format(len(sampled_sub_paths)))
            if not sampled_sub_paths:
                continue
            self.generate_sub_path_stats(sampled_sub_paths, align_len_at_path_sorted=align_len_at_path_sorted)
            sbp_to_sbp_id = self.update_sp_to_sp_id_dict(sampled_sub_paths)
            sampled_model = PathMultinomialModel(
                variant_sizes=self.variant_sizes, all_sub_paths=sampled_sub_paths)
            v_prop, *foo = self.fit_model_using_reverse_model_selection(
                model=sampled_model,
                sbp_to_sbp_id=sbp_to_sbp_id,
                criterion=self.kwargs["model_criterion"],
                init_self_max_like=False,
                bootstrap_str=f"BS{go_bs + 1: 0{n_digit}d}")
            self.variant_proportions_reps.append(v_prop)
            # self.res_loglike_reps.append(loglike)
            # self.res_criteria_reps.append(criteria)
            go_bs += 1

        # if loglevel is reset, set it back
        setup_logger(loglevel=self.loglevel, timed=True, log_file=self.logfile)
        # if there are replicates, summarize the replicates
        logger.info("Summarizing the replicates ..")
        if self.variant_proportions_reps:
            num_reps = len(self.variant_proportions_reps)
            self.vp_unique_results = {}
            for go_r, v_prop in enumerate(self.variant_proportions_reps):
                tuple_v_chosen = tuple(v_prop.keys())
                tuple_v_props = tuple(v_prop.values())
                if tuple_v_chosen not in self.vp_unique_results:
                    self.vp_unique_results[tuple_v_chosen] = {"rep_ids": [], "support": None, "values": []}
                self.vp_unique_results[tuple_v_chosen]["rep_ids"].append(go_r)
                self.vp_unique_results[tuple_v_chosen]["values"].append(tuple_v_props)
            # calculate supports
            for res_info in self.vp_unique_results.values():
                support = len(res_info["rep_ids"]) / float(num_reps)
                res_info["support"] = support
            # sort by supports AND whether concordant with the estimate from raw dataset
            raw_res = tuple(self.variant_proportions.keys())
            self.vp_unique_results_sorted = \
                sorted(self.vp_unique_results, key=lambda x: (-self.vp_unique_results[x]["support"], x != raw_res, x))

            # use the entire dataset to recalculate loglike and criterion
            # for these results with bootstrap larger than 0.1 (arbitrarily)
            logger.info("Re-evaluate the supported models using whole dataset ..")
            sbp_to_sbp_id = self.update_sp_to_sp_id_dict(self.all_sub_paths)
            for go_s, tuple_v_chosen in enumerate(self.vp_unique_results_sorted):
                chosen_dict = self.vp_unique_results[tuple_v_chosen]
                if tuple_v_chosen == raw_res:
                    raw_prop, raw_like, raw_criterion = \
                        self.variant_proportions, self.res_loglike, self.res_criterion
                else:
                    if chosen_dict["support"] > 0.05 or go_s == 0:  # the first one
                        raw_prop, raw_like, raw_criterion = self.fit_model_using_point_maximum_likelihood(
                            model=self.model,
                            sbp_to_sbp_id=sbp_to_sbp_id,
                            criterion=self.kwargs["model_criterion"],
                            chosen_ids=set(tuple_v_chosen),
                            init_self_max_like=False)
                    else:  # skip minor results
                        raw_prop, raw_like, raw_criterion = "*", "*", "*"
                chosen_dict["raw_prop"] = raw_prop
                chosen_dict["raw_like"] = raw_like
                chosen_dict["raw_criterion"] = raw_criterion
                # reset the best result to be the best supported by replicates (go_s == 0) given other conditions
                # mcmc (if requested) will be based on the new best result
                if go_s == 0 and tuple_v_chosen != raw_res and chosen_dict["support"] > 0.05:
                    self.variant_proportions_best = raw_prop

    def output_pangenome_graph(self):
        """
        # TODO can be improved later
        # TODO, isolated as an independent module
        """
        # prepare sorted output
        out_paths = [self.variant_paths[cid] for cid in self.variant_proportions_best]
        out_path_prop = [prop for prop in self.variant_proportions_best.values()]
        out_fids = [self.cid_to_fid[cid] for cid in self.variant_proportions_best]
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
        pangenome.pan_graph.write_to_gfa(os.path.join(self.outdir, "pangenome.gfa"))

    def output_result_info(self):
        with open(os.path.join(self.outdir, f"final.result.tab"), "w") as output_h:
            criterion = self.kwargs["model_criterion"].value
            met_str = "BOOTSTRAP" if self.kwargs.get("bootstrap", 0) else "JACKKNIFE"
            num_fids = len(self.cid_to_fid)
            output_h.write(f"SOLUTIONS\t{met_str}_SUPPORT\tLOGLIKELIHOOD\t{criterion}\t" +
                           "\t".join([f"fid_{_id + 1}" for _id in range(num_fids)]) +
                           f"\tFREQ_STD_OF_{met_str}\n")  # TODO add mcmc range column here and below
            raw_res = tuple(self.variant_proportions.keys())
            if raw_res in self.vp_unique_results:
                num_solutions = len(self.vp_unique_results)
            else:
                num_solutions = len(self.vp_unique_results) + 1
            s_digit = len(str(num_solutions))
            for go_s, tuple_v_chosen in enumerate(self.vp_unique_results_sorted):
                chosen_dict = self.vp_unique_results[tuple_v_chosen]
                if len(chosen_dict['values']) > 1:
                    replicate_std = np.std(chosen_dict['values'], axis=0, ddof=1)  # ddof=1 for a sample taken full
                    replicate_std_str = ",".join([f"{std_:.4f}" for std_ in replicate_std])
                else:
                    replicate_std_str = ",".join(["*" for foo_ in chosen_dict['values'][0]])
                this_line = [f"{go_s + 1:0{s_digit}d}"
                             f"\t{chosen_dict['support']}"
                             f"\t{chosen_dict['raw_like']}"
                             f"\t{chosen_dict['raw_criterion']}"] + \
                            ["-" for _id in range(num_fids)] + \
                            [replicate_std_str]
                if chosen_dict["raw_prop"] == "*":  # low bootstrap support ones without re-assessment
                    for cid in tuple_v_chosen:
                        this_line[self.cid_to_fid[cid]] = "*"
                else:
                    for cid, prop_val in chosen_dict["raw_prop"].items():
                        this_line[self.cid_to_fid[cid]] = f"{prop_val:.4f}"  # given that fid is 1-based
                output_h.write("\t".join(this_line) + "\n")
            if not self.vp_unique_results_sorted:
                this_line = [f"{num_solutions:0{s_digit}d}"
                             f"\t1.0"
                             f"\t-"
                             f"\t-"] + \
                            ["-" for _id in range(num_fids)] + \
                            ["-"]
                for cid, prop_val in self.variant_proportions.items():
                    this_line[self.cid_to_fid[cid]] = f"{prop_val:.4f}"  # given that fid is 1-based
                output_h.write("\t".join(this_line) + "\n")
            elif raw_res not in self.vp_unique_results:  # if there is no support for the raw-dataset-based best result
                this_line = [f"{num_solutions:0{s_digit}d}"
                             f"\t0"
                             f"\t{self.res_loglike}"
                             f"\t{self.res_criterion}"] + \
                            ["-" for _id in range(num_fids)] + \
                            ["-"]
                for cid, prop_val in self.variant_proportions.items():
                    this_line[self.cid_to_fid[cid]] = f"{prop_val:.4f}"  # given that fid is 1-based
                output_h.write("\t".join(this_line) + "\n")

    def output_sampling_info(self):
        """
        output bootstrap results
        """
        if self.variant_proportions_reps:
            met_str = "bootstraps" if self.kwargs.get("bootstrap", 0) else "jackknife"
            with open(os.path.join(self.outdir, f"{met_str}.replicates.tab"), "w") as output_h:
                num_fids = len(self.cid_to_fid)
                output_h.write(f"{met_str.upper()} ID\t" +
                               "\t".join([f"fid_{_id + 1}" for _id in range(num_fids)]) + "\n")
                s_digit = len(str(len(self.variant_proportions_reps)))
                for go_r, v_prop in enumerate(self.variant_proportions_reps):
                    this_line = [f"{met_str}_{go_r:0{s_digit}d}"] + ["-" for _id in range(num_fids)]
                    for cid, prop_val in v_prop.items():
                        this_line[self.cid_to_fid[cid]] = f"{prop_val:.4f}"  # given that fid is 1-based
                    output_h.write("\t".join(this_line) + "\n")

    def output_variant_info(self):
        self.cid_to_fid = OrderedDict()
        count_fid = 0
        # find candidate path ids chosen in all replicates and export them as final path ids
        for tuple_v_prop in self.vp_unique_results_sorted:
            for cid in tuple_v_prop:
                if cid not in self.cid_to_fid:
                    count_fid += 1  # fid is therefore 1-based
                    self.cid_to_fid[cid] = count_fid
        # if the result from raw dataset does not show in any of the replicates, which is terrible
        for cid in self.variant_proportions:
            if cid not in self.cid_to_fid:
                count_fid += 1
                self.cid_to_fid[cid] = count_fid
        # output information
        with open(os.path.join(self.outdir, "variants.info.tab"), "w") as output_v_h:
            output_v_h.write("FID\tCID\tUnidentifiable_to_cid\tCONTIGS\tBASES\tPATH\n")
            for cid, fid in self.cid_to_fid.items():
                uid = self.be_unidentifiable_to.get(cid, cid)
                uid = "-" if uid == cid else uid
                this_vp = self.variant_paths[cid]
                this_size = self.variant_sizes[cid]
                output_v_h.write(f"{fid}\t{cid}\t{uid}\t{len(this_vp)}\t{this_size}\t{path_to_gaf_str(this_vp)}\n")

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
        if filter_by_graph:
            for go_record, record in enumerate(graph_alignment.raw_records):
                this_path = self.graph.get_standardized_path(record.path)
                if len(this_path) == 0:
                    logger.warning(f"Record {go_record} is empty")
                    continue
                if this_path not in self.read_paths:
                    if self.graph.contain_path(this_path):
                        self.read_paths[this_path] = [go_record]
                else:
                    self.read_paths[this_path].append(go_record)
        else:
            for go_record, record in enumerate(graph_alignment.raw_records):
                this_path = self.graph.get_standardized_path(record.path)
                if this_path not in self.read_paths:
                    self.read_paths[this_path] = []
                self.read_paths[this_path].append(go_record)
        # filter 2
        if min_alignment_counts > 1:
            for this_path in list(self.read_paths):
                if len(self.read_paths[this_path]) < min_alignment_counts:
                    self.read_paths_masked.add(this_path)
        # check
        # TODO move to bootstrap
        if len(self.read_paths_masked) == len(self.read_paths):
            logger.error("No valid alignment records remains after filtering!")
            raise SystemExit(0)
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
        self.align_len_at_path_sorted = sorted(self.align_len_at_path_map.values())
        # store min/max value
        self.min_alignment_length = self.align_len_at_path_sorted[0]
        self.max_alignment_length = self.align_len_at_path_sorted[-1]
        self.generate_maximum_read_path_size()

    def generate_maximum_read_path_size(self):
        assert bool(self.read_paths), "empty read paths!"
        self.max_read_path_size = 0
        for this_read_path in self.read_paths:
            self.max_read_path_size = max(self.max_read_path_size, len(this_read_path))

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

    def gen_candidate_variants(
            self,
            # path_generator="H",
            start_strategy="random",
            min_num_search=1000,
            max_num_search=10000,
            max_num_traversals=50000,
            max_uniq_traversal=200,
            num_processes=1,
            uni_chromosome=False,
            **kwargs):
        """
        generate candidate variant paths from the graph
        """
        # if path_generator == "H":
        generator = VariantGenerator(
            traversome_obj=self,
            start_strategy=start_strategy,
            min_num_valid_search=min_num_search,
            max_num_valid_search=max_num_search,
            max_num_traversals=max_num_traversals,
            max_uniq_traversal=max_uniq_traversal,
            num_processes=num_processes,
            force_circular=self.force_circular,
            uni_chromosome=uni_chromosome,
            decay_f=kwargs.get("search_decay_factor"),
            temp_dir=fpath(self.outdir).joinpath("tmp.candidates"),
            resume=self.resume,
            )
        generator.generate_heuristic_paths()
        # TODO add self.user_variant_paths during heuristic search instead of here
        self.variant_paths = list(self.user_variant_paths)
        user_v_p_set = set(self.user_variant_paths)
        for go_v, g_variant in enumerate(generator.variants):
            if g_variant not in user_v_p_set:
                self.variant_paths.append(g_variant)
            else:
                logger.info(f"searched variant {go_v + 1} existed in user designed.")
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

    def gen_all_informative_sub_paths(self):
        """
        generate all sub paths and their occurrences for each candidate variant
        """
        # count sub path occurrences for each candidate variant and recorded in self.variant_subpath_counters
        # this_overlap = self.graph.uni_overlap()
        # self.variant_subpath_counters = OrderedDict()
        for this_var_p in self.variant_paths:
            # foo = self.get_variant_sub_paths(this_var_p)
            self.variant_subpath_counters[this_var_p] = self.get_variant_sub_paths(this_var_p)
        # self.variant_subpath_counters should be only a subset of self.subpath_generator.variant_subpath_counters

        # create unidentifiable table
        # NOTE unidentifiable senario is not common for fine dataset with clear abundant variants
        # so we do not include it into the bootstrap
        self.be_unidentifiable_to = OrderedDict()
        for represent_iso_id in range(self.num_put_variants):
            for check_iso_id in range(represent_iso_id, self.num_put_variants):
                # every id will be checked only once, either unidentifiable to a previous id or represent itself
                if check_iso_id not in self.be_unidentifiable_to:
                    if check_iso_id == represent_iso_id:  # represent itself
                        self.be_unidentifiable_to[check_iso_id] = check_iso_id
                    elif self.variant_subpath_counters[self.variant_paths[check_iso_id]] == \
                            self.variant_subpath_counters[self.variant_paths[represent_iso_id]] and \
                            self.variant_sizes[check_iso_id] == self.variant_sizes[represent_iso_id]:
                        self.be_unidentifiable_to[check_iso_id] = represent_iso_id
        # logger.info(str(self.be_unidentifiable_to))
        self.repr_to_merged_variants = \
            OrderedDict([(rps_id, []) for rps_id in sorted(set(self.be_unidentifiable_to.values()))])
        for check_iso_id, rps_iso_id in self.be_unidentifiable_to.items():
            self.repr_to_merged_variants[rps_iso_id].append(check_iso_id)
        for unidentifiable_ids in self.repr_to_merged_variants.values():
            if len(unidentifiable_ids) > 1:
                logger.warning("Mutually unidentifiable paths in current alignment: %s" % unidentifiable_ids)

        # transform self.variant_subpath_counters to self.all_sub_paths
        self.all_sub_paths = OrderedDict()
        for go_variant, variant_path in enumerate(self.variant_paths):
            sub_paths_group = self.variant_subpath_counters[variant_path]
            for this_sub_path, this_sub_count in sub_paths_group.items():
                if this_sub_path not in self.all_sub_paths:
                    self.all_sub_paths[this_sub_path] = SubPathInfo()
                self.all_sub_paths[this_sub_path].from_variants[go_variant] = this_sub_count

        # shared sub-paths (with the same counts) are also informative because their frequencies may vary
        #     upon the change of averaged genome size after the change of proportions
        # WRONG: to simplify downstream calculation, remove shared sub-paths (with same counts) shared by all variants
        # for this_sub_path, this_sub_path_info in list(self.all_sub_paths.items()):
        #     if len(this_sub_path_info.from_variants) == self.num_put_variants and \
        #             len(set(this_sub_path_info.from_variants.values())) == 1:
        #         for variant_p, sub_paths_group in self.variant_subpath_counters.items():
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
            #         sub_paths_group = self.variant_subpath_counters[variant_path]
            #         if read_path in sub_paths_group:
            #             print("found", len(record_ids), read_path)
            #             break
            #     else:
            #         print("lost", len(record_ids), read_path)

        logger.info(f"Generated {len(self.all_sub_paths)} informative sub-paths based on "
                    f"{sum([len(sbp.mapped_records) for sbp in self.all_sub_paths.values()])} records in total")

    @staticmethod
    def update_sp_to_sp_id_dict(all_sub_paths):
        sbp_to_sbp_id = {}
        for go_sp, this_sub_path in enumerate(all_sub_paths):
            sbp_to_sbp_id[this_sub_path] = go_sp
        return sbp_to_sbp_id

    def generate_sub_path_stats(self, all_sub_paths, align_len_at_path_sorted):
        """
        """
        logger.debug("Generating sub-path statistics ..")
        # It is not proper to use the median, because both ends lead to a small estimation,
        # while the median may lead to the maximum, use quarter may be a solution;
        # if len(self.all_sub_paths) > 1e5:
        #     # to seed up
        #     use_median = True
        # else:
        #     # maybe slightly precise than above, not assessed yet
        #     use_median = False
        # use_median = False

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
            jackknife_size=None,
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
            logger.debug("Using bootstrap.")
            new_records_pool = self.random.choices(self.records_pool_sorted, k=bootstrap_size)
        elif jackknife_size:
            logger.debug(f"Using Jackknife: leave-{jackknife_size}-out.")
            keep_ids = self.random.sample(range(self.num_valid_records), k=self.num_valid_records - jackknife_size)
            new_records_pool = [self.records_pool_sorted[s_id_] for s_id_ in keep_ids]
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
        align_len_at_path_sorted = sorted([self.align_len_at_path_map[rec_id]
                                           for rec_id in new_records_pool
                                           if rec_id not in masking_rec_ids])
        return new_sub_paths, align_len_at_path_sorted

    def __subpath_info_filler(
            self, this_sub_path, align_len_at_path_sorted, align_len_id_lookup_table, all_sub_paths):
        internal_len = self.graph.get_path_internal_length(this_sub_path)
        # # read_paths with overlaps should be and were already trimmed, so we should proceed without overlaps
        # external_len_without_overlap = self.graph.get_path_len_without_terminal_overlaps(this_sub_path)
        # 2023-03-09: get_path_len_without_terminal_overlaps -> get_path_length
        # Because the end of the alignment can still stretch to the overlapped region
        # and will not be recorded in the path.
        # Thus, the start point can be the path length not the uni_overlap-trimmed one.
        external_len = self.graph.get_path_length(this_sub_path, check_valid=False, adjust_for_cyclic=False)
        left_id, right_id = self._get_id_range_in_increasing_lengths(
            min_len=internal_len + 2,
            max_len=external_len,
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
                """
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
                maximum_num_cat = read_len_aligned - internal_len - 2
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
                return external_len - read_len_aligned + 1

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

    def _get_id_range_in_increasing_lengths(self, min_len, max_len, align_len_id_lookup_table, max_id):
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
            variant_subpath_counters=self.variant_subpath_counters,
            sbp_to_sbp_id=sbp_to_sbp_id,
            repr_to_merged_variants=self.repr_to_merged_variants,
            be_unidentifiable_to=self.be_unidentifiable_to,
            loglevel=self.loglevel)
        if init_self_max_like:
            self.max_like_fit = max_like_fit
        use_prop, this_like, this_criterion =\
            self.max_like_fit.point_estimate(chosen_ids=chosen_ids, criterion=criterion)
        return use_prop, this_like, this_criterion

    def fit_model_using_reverse_model_selection(self,
                                                model,
                                                sbp_to_sbp_id,
                                                criterion=Criterion.BIC,
                                                chosen_ids: Union[typingODict[int, bool], Set] = None,
                                                init_self_max_like: bool = True,
                                                bootstrap_str: str = ""):
        """
        :param bootstrap_str: turn on to only print simple information and mark the bootstrap id
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
            variant_subpath_counters=self.variant_subpath_counters,
            sbp_to_sbp_id=sbp_to_sbp_id,
            repr_to_merged_variants=self.repr_to_merged_variants,
            be_unidentifiable_to=self.be_unidentifiable_to,
            loglevel=self.loglevel,
            bootstrap_mode=bootstrap_str)
        if init_self_max_like:
            self.max_like_fit = max_like_fit
        return max_like_fit.reverse_model_selection(
            n_proc=self.num_processes, criterion=criterion, chosen_ids=chosen_ids,
            user_fixed_ids=self.user_variant_fixed_ids)

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
    #     self.variant_subpath_counters = OrderedDict()  # each value is a dict(sub_path->sub_path_counts)
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
        return self.bayesian_fit.run_mcmc(self.kwargs["n_generations"], self.kwargs["n_burn"], chosen_ids=chosen_ids)

    def output_seqs(self):
        out_seq_file_num = len([x for x in self.variant_proportions_best.values() if x >= self.out_prob_threshold])
        if not out_seq_file_num:
            return
        out_digit = len(str(out_seq_file_num))
        count_seq = 0
        # sorted_rank = sorted(list(self.variant_proportions_best), key=lambda x: -self.variant_proportions_best[x])
        # for go_isomer, this_prob in self.ext_component_proportions.items():
        for count_out, cid in enumerate(self.variant_proportions_best):
            if self.repr_to_merged_variants and cid not in self.repr_to_merged_variants:
                # when self.generate_all_informative_sub_paths() was not called,
                # bool(repr_to_merged_variants)==False - TODO
                continue
            this_prob = self.variant_proportions_best[cid]
            if this_prob >= self.out_prob_threshold:
                this_base_name = "variant.%0{}i".format(out_digit) % count_out
                seq_file_name = os.path.join(self.outdir, this_base_name + ".fasta")
                with open(seq_file_name, "w") as output_handler:
                    if self.repr_to_merged_variants:
                        unidentifiable_ids = self.repr_to_merged_variants[cid]
                    else:
                        # when self.generate_all_informative_sub_paths() was not called - TODO
                        unidentifiable_ids = [cid]
                    len_un_id = len(unidentifiable_ids)
                    lengths = []
                    for _ss, comp_id in enumerate(unidentifiable_ids):
                        this_seq = self.graph.export_path(self.variant_paths[comp_id], check_valid=False)
                        this_seq.label = self.cid_to_fid[comp_id]  # simplify the output fasta head
                        this_len = len(this_seq.seq)
                        lengths.append(this_len)
                        if len_un_id > 1:
                            seq_label = \
                                f">{this_seq.label} freq<={this_prob:.4f} len={this_len}bp uid={count_seq + 1}.{_ss}"
                            logger.debug("{}.{} path={}".format(this_base_name, _ss, this_seq.label))
                        else:
                            seq_label = \
                                f">{this_seq.label} freq={this_prob:.4f} len={this_len}bp uid={count_seq + 1}"
                            logger.debug("{} path={}".format(this_base_name, this_seq.label))
                        output_handler.write(seq_label + "\n" + this_seq.seq + "\n")
                        count_seq += 1
                    logger.log("RES",
                               f"{this_base_name}x{len_un_id} "
                               f"freq={this_prob:.4f} len={'/'.join([str(_l) for _l in lengths])}")
        # logger.info("Output {} seqs (%.4f to %.{}f): ".format(count_seq, len(str(self.out_prob_threshold)) - 2)
        #             % (max(self.variant_proportions.values()), self.out_prob_threshold))
        logger.info("Output {} seqs in {} files".format(count_seq, out_seq_file_num))

    def shuffled(self, sorted_list):
        sorted_list = deepcopy(sorted_list)
        self.random.shuffle(sorted_list)
        return sorted_list


