#!/usr/bin/env python

"""
Top-level class for running the CLI
"""

import os
import sys
import random

from loguru import logger
from copy import deepcopy
from collections import OrderedDict
from traversome.Assembly import Assembly
from traversome.GraphAlignRecords import GraphAlignRecords
from traversome.utils import \
    SubPathInfo, LogLikeFormulaInfo, Criterion  # ProcessingGraphFailed,
from traversome.ModelFitMaxLike import ModelFitMaxLike
from traversome.ModelFitBayesian import ModelFitBayesian
from traversome.PathGenerator import PathGenerator
# from traversome.CleanGraph import CleanGraph
from typing import OrderedDict as typingODict
from typing import Set
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
            outdir,
            num_processes=1,
            force_circular=True,
            out_prob_threshold=0.001,
            keep_temp=False,
            random_seed=12345,
            loglevel="DEBUG",
            **kwargs):
        # store input files and params
        self.graph_gfa = graph
        self.alignment_file = alignment
        self.alignment_format = self.parse_alignment_format_from_postfix()
        self.outdir = outdir
        self.num_processes = num_processes
        self.force_circular = force_circular
        self.out_prob_threshold = out_prob_threshold
        self.keep_temp = keep_temp
        self.kwargs = kwargs

        # init logger
        self.logfile = os.path.join(self.outdir, "traversome.log.txt")
        self.loglevel = loglevel.upper()
        self.setup_timed_logger(loglevel.upper())

        # alignment and graph values
        self.graph = None
        self.alignment = None
        self.align_len_at_path_sorted = None
        self.__align_len_lookup_table = {}
        self.max_alignment_length = None
        self.min_alignment_length = None
        self.read_paths = OrderedDict()
        self.max_read_path_size = None

        # variant model to be generated
        self.variant_paths = []  # each element is a tuple(path)
        self.variant_sizes = []
        self.num_put_variants = None
        self.variant_subpath_counters = OrderedDict()  # variant -> dict(sub_path->sub_path_counts)
        self.all_sub_paths = OrderedDict()
        self.sbp_to_sbp_id = {}
        self.observed_sbp_id_set = set()
        self.be_unidentifiable_to = {}
        # use the merged variants to represent each set of variants.
        # within each set the variants are unidentifiable to each other
        self.merged_variants = {}
        # The result of model fitting using ml/mc, the base to update above model information for a second fitting run
        # {variant_id: percent}
        self.variant_proportions = OrderedDict()

        # self.pal_len_sbp_Xs = OrderedDict()
        # self.sbp_Xs = []

        self.max_like_fit = None
        self.bayesian_fit = None
        self.random = random
        self.random.seed(random_seed)

    def run(self, path_generator="H", hetero_chromosomes=True):
        """
        Parse the assembly graph files ...
        """
        self.graph = Assembly(self.graph_gfa)
        self.graph.update_vertex_clusters()
        if self.kwargs.get("graph_component_selection", 0):
            self.graph.reduce_graph_by_weight(cutoff=self.kwargs["graph_component_selection"])
        logger.info("#contigs in total: {}".format(len(self.graph.vertex_info)))
        logger.info("#contigs in each component: {}".format(sorted([len(cls) for cls in self.graph.vertex_clusters])))

        self.alignment = GraphAlignRecords(
            self.alignment_file,
            alignment_format=self.alignment_format,
            min_align_len=self.kwargs.get("min_alignment_len_cutoff", 100),
            min_identity=self.kwargs.get("min_alignment_identity_cutoff", 0.8),
        )
        self.generate_read_paths()
        logger.info("#reads aligned: %i" % len(self.alignment.read_records))
        logger.info("#records aligned: %i" % len(self.alignment.raw_records))
        logger.info("#read paths: %i" % len(self.read_paths))
        self.get_align_len_dist()
        # logger.debug("Cleaning graph ...")
        # self.clean_graph()
        logger.debug("Generating candidate variant paths ...")
        self.gen_candidate_paths(
            path_generator=path_generator,
            num_search=self.kwargs.get("num_valid_search", 1000),
            num_processes=self.num_processes,
            hetero_chromosomes=hetero_chromosomes
        )
        if self.num_put_variants == 0:
            raise Exception("No candidate variants found!")
        elif self.num_put_variants == 1:
            self.variant_proportions[0] = 1.
        else:
            logger.debug("Estimating candidate variant frequencies using Maximum Likelihood...")
            # if self.kwargs["criterion"] == "single":
            #     # single point estimate
            #     self.ext_variant_proportions = self.fit_model_using_point_maximum_likelihood()
            # else:
            self.variant_proportions = self.fit_model_using_reverse_model_selection(
                criterion=self.kwargs["model_criterion"])
            # update candidate info according to the result of reverse model selection
            if len(self.variant_proportions) > 1:
                logger.debug("Estimating candidate variant frequencies using Bayesian MCMC ...")
                self.variant_proportions = self.fit_model_using_bayesian_mcmc(chosen_ids=self.variant_proportions)

        self.output_seqs()

    def parse_alignment_format_from_postfix(self):
        if self.alignment_file.lower().endswith(".gaf"):
            alignment_format = "GAF"
        elif self.alignment_file.lower().endswith(".tsv"):
            alignment_format = "SPA-TSV"
        else:
            raise Exception("Please denote the alignment format using adequate postfix (.gaf/.tsv)")
        return alignment_format

    def generate_read_paths(self):
        for go_record, record in enumerate(self.alignment.raw_records):
            this_read_path = self.graph.get_standardized_path(record.path)
            if this_read_path not in self.read_paths:
                self.read_paths[this_read_path] = []
            self.read_paths[this_read_path].append(go_record)
        self.generate_maximum_read_path_size()

    def generate_maximum_read_path_size(self):
        if not self.read_paths:
            self.generate_read_paths()
        self.max_read_path_size = 0
        for this_read_path in self.read_paths:
            self.max_read_path_size = max(self.max_read_path_size, len(this_read_path))

    # def clean_graph(self, min_effective_count=10, ignore_ratio=0.001):
    #     """ deprecated for now"""
    #     clean_graph_obj = CleanGraph(self)
    #     clean_graph_obj.run(min_effective_count=min_effective_count, ignore_ratio=ignore_ratio)

    def get_align_len_dist(self):
        """
        Get sorted alignment lengths, optionally save to file 
        and store longest to self.
        """
        logger.debug("Summarizing alignment length distribution")

        # get sorted alignment lengths
        self.align_len_at_path_sorted = sorted([rec.p_align_len for rec in self.alignment.raw_records])

        # optionally save temp files
        if self.keep_temp:
            opath = os.path.join(self.outdir, "align_len_at_path_sorted.txt")
            with open(opath, "w") as out:
                out.write("\n".join(map(str, self.align_len_at_path_sorted)))

        # store max value 
        self.min_alignment_length = self.align_len_at_path_sorted[0]
        self.max_alignment_length = self.align_len_at_path_sorted[-1]

        # report result
        logger.info(
            "Alignment length range at path: [{}, {}]".format(self.min_alignment_length, self.max_alignment_length))

    def gen_candidate_paths(
            self,
            path_generator="H",
            num_search=1000,
            num_processes=1,
            hetero_chromosomes=True):
        """
        generate candidate variant paths from the graph
        """
        if path_generator == "H":
            generator = PathGenerator(
                traversome_obj=self,
                min_num_valid_search=num_search,
                num_processes=num_processes,
                force_circular=self.force_circular,
                hetero_chromosome=hetero_chromosomes)
            generator.generate_heuristic_paths()
            self.variant_paths = generator.variants
        self.__update_params_for_variants()

    def __update_params_for_variants(self):
        self.variant_sizes = [self.graph.get_path_length(variant_p, check_valid=False, adjust_for_cyclic=True)
                              for variant_p in self.variant_paths]
        self.num_put_variants = len(self.variant_paths)

        for go_p, path in enumerate(self.variant_paths):
            logger.debug("PATH{}: {}".format(go_p, self.graph.repr_path(path)))

        # generate subpaths: the binomial sets
        if self.num_put_variants > 1:
            logger.info("Generating sub-paths ..")
            self.gen_all_informative_sub_paths()
        elif self.num_put_variants == 0:
            logger.warning("No valid configuration found for the input assembly graph.")
        else:
            # self.merged_variants = OrderedDict([(0, [0])])
            logger.warning("Only one genomic configuration found for the input assembly graph.")

    # TODO get subpath adaptive to length=1, more general and less restrictions
    def get_variant_sub_paths(self, variant_path, return_sub_paths=True):
        if variant_path in self.variant_subpath_counters:
            if return_sub_paths:
                return self.variant_subpath_counters[variant_path]
        else:
            # if this_overlap is None:
            #     this_overlap = self.graph.uni_overlap()
            these_sub_paths = dict()
            num_seg = len(variant_path)
            # print("run get")
            if self.force_circular:
                for go_start_v, start_segment in enumerate(variant_path):
                    # find the longest sub_path,
                    # that begins with start_segment and be in the range of alignment length
                    this_longest_sub_path = [start_segment]
                    this_internal_path_len = 0
                    go_next = (go_start_v + 1) % num_seg
                    while this_internal_path_len < self.max_alignment_length:
                        next_n, next_e = variant_path[go_next]
                        next_v_info = self.graph.vertex_info[next_n]
                        pre_n, pre_e = this_longest_sub_path[-1]
                        this_overlap = next_v_info.connections[not next_e][(pre_n, pre_e)]
                        this_longest_sub_path.append((next_n, next_e))
                        this_internal_path_len += next_v_info.len - this_overlap
                        go_next = (go_next + 1) % num_seg
                    # print("this_longest_sub_path", this_longest_sub_path)
                    # print(self.graph.get_path_internal_length(this_longest_sub_path), self.min_alignment_length)
                    # when the overlap is long and the contig is short,
                    # the path with internal_length shorter than tha alignment length can still help
                    # so remove the condition for min_alignment_length
                    # if len(this_longest_sub_path) < 2 \
                    #         or self.graph.get_path_internal_length(this_longest_sub_path) < self.min_alignment_length:
                    #     continue
                    # TODO size of 1 can also be included
                    if len(this_longest_sub_path) < 2:
                        continue
                    # print("this_longest_sub_path", this_longest_sub_path, "passed")

                    # record shorter sub_paths starting from start_segment
                    len_this_sub_p = len(this_longest_sub_path)
                    for skip_tail in range(len_this_sub_p - 1):
                        this_sub_path = \
                            self.graph.get_standardized_path(this_longest_sub_path[:len_this_sub_p - skip_tail])
                        # print("checking subpath existence", this_sub_path)
                        if this_sub_path not in self.read_paths:
                            continue
                        # print("checking subpath existence", this_sub_path, "passed")
                        # when the uni_overlap is long and the contig is short,
                        # the path with internal_length shorter than tha alignment length can still help
                        # so remove the condition for min_alignment_length
                        # if self.graph.get_path_internal_length(this_sub_path) < self.min_alignment_length:
                        #     break
                        if this_sub_path not in these_sub_paths:
                            these_sub_paths[this_sub_path] = 0
                        these_sub_paths[this_sub_path] += 1
            else:
                for go_start_v, start_segment in enumerate(variant_path):
                    # find the longest sub_path,
                    # that begins with start_segment and be in the range of alignment length
                    this_longest_sub_path = [start_segment]
                    this_internal_path_len = 0
                    go_next = go_start_v + 1
                    while go_next < num_seg and this_internal_path_len < self.max_alignment_length:
                        next_n, next_e = variant_path[go_next]
                        next_v_info = self.graph.vertex_info[next_n]
                        pre_n, pre_e = this_longest_sub_path[-1]
                        this_overlap = next_v_info.connections[not next_e][(pre_n, pre_e)]
                        this_longest_sub_path.append((next_n, next_e))
                        this_internal_path_len += next_v_info.len - this_overlap
                        go_next += 1
                    if len(this_longest_sub_path) < 2 \
                            or self.graph.get_path_internal_length(this_longest_sub_path) < self.min_alignment_length:
                        continue
                    # record shorter sub_paths starting from start_segment
                    len_this_sub_p = len(this_longest_sub_path)
                    for skip_tail in range(len_this_sub_p - 1):
                        this_sub_path = \
                            self.graph.get_standardized_path_circ(this_longest_sub_path[:len_this_sub_p - skip_tail])
                        if this_sub_path not in self.read_paths:
                            continue
                        if self.graph.get_path_internal_length(this_sub_path) < self.min_alignment_length:
                            break
                        if this_sub_path not in these_sub_paths:
                            these_sub_paths[this_sub_path] = 0
                        these_sub_paths[this_sub_path] += 1
            self.variant_subpath_counters[variant_path] = these_sub_paths
            if return_sub_paths:
                return these_sub_paths

    def gen_all_informative_sub_paths(self):
        """
        generate all sub paths and their occurrences for each candidate variant
        """
        # count sub path occurrences for each candidate variant and recorded in self.variant_subpath_counters
        # this_overlap = self.graph.uni_overlap()
        # self.variant_subpath_counters = OrderedDict()
        for this_var_p in self.variant_paths:
            self.get_variant_sub_paths(this_var_p)

        # create unidentifiable table
        self.be_unidentifiable_to = OrderedDict()
        for represent_iso_id in range(self.num_put_variants):
            for check_iso_id in range(represent_iso_id, self.num_put_variants):
                # every id will be checked only once, either unidentifiable to a previous id or represent itself
                if check_iso_id not in self.be_unidentifiable_to:
                    if check_iso_id == represent_iso_id:  # represent itself
                        self.be_unidentifiable_to[check_iso_id] = check_iso_id
                    elif self.variant_subpath_counters[self.variant_paths[check_iso_id]] == \
                            self.variant_subpath_counters[self.variant_paths[represent_iso_id]]:
                        self.be_unidentifiable_to[check_iso_id] = represent_iso_id
        # logger.info(str(self.be_unidentifiable_to))
        self.merged_variants = \
            OrderedDict([(rps_id, []) for rps_id in sorted(set(self.be_unidentifiable_to.values()))])
        for check_iso_id, rps_iso_id in self.be_unidentifiable_to.items():
            self.merged_variants[rps_iso_id].append(check_iso_id)
        for unidentifiable_ids in self.merged_variants.values():
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

        # to simplify downstream calculation, remove shared sub-paths shared by all variants
        for this_sub_path, this_sub_path_info in list(self.all_sub_paths.items()):
            if len(this_sub_path_info.from_variants) == self.num_put_variants and \
                    len(set(this_sub_path_info.from_variants.values())) == 1:
                for variant_p, sub_paths_group in self.variant_subpath_counters.items():
                    # TODO: using simulated data to check what if sub_paths_group is empty
                    #       it should be fine though
                    del sub_paths_group[this_sub_path]
                del self.all_sub_paths[this_sub_path]

        if not self.all_sub_paths:
            logger.error("No valid subpath found!")
            # TODO how to exit quietly
            sys.exit(1)

        # match graph alignments to all_sub_paths
        for read_path, record_ids in self.read_paths.items():
            if read_path in self.all_sub_paths:
                self.all_sub_paths[read_path].mapped_records = record_ids
        # # remove sp with zero observations, commented because we have added constraits above
        # for this_sub_path in list(self.all_sub_paths):
        #     if not self.all_sub_paths[this_sub_path].mapped_records:
        #         del self.all_sub_paths[this_sub_path]

        logger.info("Generated {} informative sub-paths".format(len(self.all_sub_paths)))
        self.generate_sub_path_stats()
        # build an index
        self.update_sp_to_sp_id_dict()
        # TODO: check the number
        logger.info("Generated {} valid informative sub-paths".format(len(self.all_sub_paths)))

    def update_sp_to_sp_id_dict(self):
        self.sbp_to_sbp_id = {}
        for go_sp, this_sub_path in enumerate(self.all_sub_paths):
            self.sbp_to_sbp_id[this_sub_path] = go_sp

    def generate_sub_path_stats(self):
        """
        """
        logger.debug("Generating sub-path statistics ..")
        # It is not proper to use the median, because both ends lead to a small estimation,
        # while the median may lead to the maximum, use quarter may be a solution;
        # or TODO use parallelization to be precise and fast
        # if len(self.all_sub_paths) > 1e5:
        #     # to seed up
        #     use_median = True
        # else:
        #     # maybe slightly precise than above, not assessed yet
        #     use_median = False
        use_median = False
        self.__generate_align_len_lookup_table()

        for this_sub_path, this_sub_path_info in list(self.all_sub_paths.items()):
            # 0.2308657169342041
            internal_len = self.graph.get_path_internal_length(this_sub_path)
            # 0.18595576286315918
            # # read_paths with overlaps should be and were already trimmed, so we should proceed without overlaps
            # external_len_without_overlap = self.graph.get_path_len_without_terminal_overlaps(this_sub_path)

            # 2023-03-09: get_path_len_without_terminal_overlaps -> get_path_length
            # Because the end of the alignment can still stretch to the overlapped region
            # and will not be recorded in the path.
            # Thus, the start point can be the path length not the uni_overlap-trimmed one.
            external_len = self.graph.get_path_length(this_sub_path, check_valid=False, adjust_for_cyclic=False)
            # 0.15802343183582341
            # left_id, right_id = get_id_range_in_increasing_values(
            #     min_num=internal_len + 2, max_num=external_len_without_overlap,
            #     increasing_numbers=self.align_len_at_path_sorted)
            left_id, right_id = self.__get_id_range_in_increasing_values(min_num=internal_len + 2, max_num=external_len)
            if left_id > right_id:
                # no read found within this scope
                logger.trace("Remove {} after pruning contig overlap ..".format(this_sub_path))
                del self.all_sub_paths[this_sub_path]
                continue
            if use_median:
                # 0.12435293197631836
                if int((left_id + right_id) / 2) == (left_id + right_id) / 2.:
                    median_len = self.align_len_at_path_sorted[int((left_id + right_id) / 2)]
                else:
                    median_len = (self.align_len_at_path_sorted[int((left_id + right_id) / 2)] +
                                  self.align_len_at_path_sorted[int((left_id + right_id) / 2) + 1]) / 2.
                this_sub_path_info.num_possible_X = self.graph.get_num_of_possible_alignment_start_points(
                    read_len_aligned=median_len, align_to_path=this_sub_path, path_internal_len=internal_len)
            else:
                # 7.611977815628052
                # maybe slightly precise than above, assessed
                # this can be super time consuming in case of many subpaths, e.g.
                num_possible_Xs = {}
                align_len_id = left_id
                while align_len_id <= right_id:
                    this_len = self.align_len_at_path_sorted[align_len_id]
                    this_x = self.graph.get_num_of_possible_alignment_start_points(
                        read_len_aligned=this_len, align_to_path=this_sub_path, path_internal_len=internal_len)
                    if this_len not in num_possible_Xs:
                        num_possible_Xs[this_len] = 0
                    num_possible_Xs[this_len] += this_x
                    align_len_id += 1
                    while align_len_id <= right_id and self.align_len_at_path_sorted[align_len_id] == this_len:
                        num_possible_Xs[this_len] += this_x
                        align_len_id += 1
                this_sub_path_info.num_possible_X = sum(num_possible_Xs.values()) / (right_id - left_id + 1)
            this_sub_path_info.num_in_range = right_id + 1 - left_id
            this_sub_path_info.num_matched = len(this_sub_path_info.mapped_records)
        # if for_multinomial:
        #     # generate path alignment length occurrences at sub paths
        #     # could largely be simplified
        #     logger.debug("Counting path-alignment-length occurrences at sub paths ..")
        #     self.sbp_Xs = []
        #     self.pal_len_sbp_Xs = OrderedDict([(pa_len, {}) for pa_len in sorted(set(self.align_len_at_path_sorted))])
        #     for go_sp, (this_sub_path, this_sub_path_info) in enumerate(self.all_sub_paths.items()):
        #         # may be useless
        #         # for this_len, this_x in this_sub_path_info.num_possible_Xs.items():
        #         #     self.pal_len_sbp_Xs[this_len][go_sp] = this_x
        #         # try this
        #         self.sbp_Xs.append(sum(this_sub_path_info.num_possible_Xs.values()))

    def __generate_align_len_lookup_table(self):
        """
        called by generate_sub_path_stats
        to speed up align len id looking up
        """
        its_left_id = 0
        its_right_id = 0
        max_id = len(self.align_len_at_path_sorted) - 1
        self.__align_len_lookup_table = \
            {potential_len: {"as_left_lim_id": None, "as_right_lim_id": None}
             for potential_len in range(self.min_alignment_length, self.max_alignment_length + 1)}
        for potential_len in range(self.min_alignment_length, self.max_alignment_length + 1):
            if potential_len == self.align_len_at_path_sorted[its_right_id]:
                self.__align_len_lookup_table[potential_len]["as_left_lim_id"] = its_left_id = its_right_id
                while potential_len == self.align_len_at_path_sorted[its_right_id]:
                    self.__align_len_lookup_table[potential_len]["as_right_lim_id"] = its_right_id
                    if its_right_id == max_id:
                        break
                    else:
                        its_left_id = its_right_id
                        its_right_id += 1
            else:
                self.__align_len_lookup_table[potential_len]["as_left_lim_id"] = its_right_id
                self.__align_len_lookup_table[potential_len]["as_right_lim_id"] = its_left_id

    def __get_id_range_in_increasing_values(self, min_num, max_num):
        """
        called by __generate_align_len_lookup_table
        replace get_id_range_in_increasing_values func in utils.py
        """
        left_id = self.__align_len_lookup_table.get(min_num, {}).\
            get("as_left_lim_id", 0)
        right_id = self.__align_len_lookup_table.get(max_num, {}).\
            get("as_right_lim_id", len(self.align_len_at_path_sorted) - 1)
        return left_id, right_id

    def update_observed_sp_ids(self):
        self.observed_sbp_id_set = set()
        for go_sp, (this_sub_path, this_sub_path_info) in enumerate(self.all_sub_paths.items()):
            if this_sub_path_info.mapped_records:
                self.observed_sbp_id_set.add(go_sp)
            else:
                logger.trace("Drop subpath without observation: {}: {}".format(go_sp, this_sub_path))

    def cover_all_observed_subpaths(self, variant_ids):
        if not self.observed_sbp_id_set:
            self.update_observed_sp_ids()
        model_sp_ids = set()
        for go_var in variant_ids:
            for sub_path in self.variant_subpath_counters[self.variant_paths[go_var]]:
                if sub_path in self.sbp_to_sbp_id:
                    # if sub_path was not dropped after the construction of self.variant_subpath_counters
                    model_sp_ids.add(self.sbp_to_sbp_id[sub_path])
        if self.observed_sbp_id_set.issubset(model_sp_ids):
            return True
        else:
            return False

    def get_multinomial_like_formula(self,
                                     variant_percents,
                                     log_func,
                                     within_variant_ids: Set = None):
        """
        use a combination of multiple multinomial distributions
        :param variant_percents:
             input symengine.Symbols for maximum likelihood analysis (scipy),
                 e.g. [Symbol("P" + str(variant_id)) for variant_id in range(self.num_put_variants)].
             input pm.Dirichlet for bayesian analysis (pymc3),
                 e.g. pm.Dirichlet(name="comp", a=np.ones(variant_num), shape=(variant_num,)).
        :param log_func:
             input symengine.log for maximum likelihood analysis using scipy,
             input tt.log for bayesian analysis using pymc3
        :param within_variant_ids:
             constrain the variant testing scope. Test all variants by default.
                 e.g. set([0, 2])
        :return: LogLikeFormulaInfo object
        """
        if not within_variant_ids or within_variant_ids == set(range(self.num_put_variants)):
            within_variant_ids = None
        # total length (all possible matches, ignoring margin effect if not circular)
        total_length = 0
        if within_variant_ids:
            for go_variant, go_length in enumerate(self.variant_sizes):
                if go_variant in within_variant_ids:
                    total_length += variant_percents[go_variant] * float(go_length)
        else:
            for go_variant, go_length in enumerate(self.variant_sizes):
                total_length += variant_percents[go_variant] * float(go_length)

        # prepare subset of all_sub_paths in a list
        these_sp_info = OrderedDict()
        # if within_variant_ids:
        #     for go_sp, (this_sub_path, this_sp_info) in enumerate(self.all_sub_paths.items()):
        #         if set(this_sp_info.from_variants) & within_variant_ids:
        #             these_sp_info[go_sp] = this_sp_info
        # else:
        for go_sp, (this_sub_path, this_sp_info) in enumerate(self.all_sub_paths.items()):
            these_sp_info[go_sp] = this_sp_info
        # clean zero expectations to avoid nan formula
        for check_sp in list(these_sp_info):
            if these_sp_info[check_sp].num_possible_X < 1:
                del these_sp_info[check_sp]
                continue
            if within_variant_ids and not (set(these_sp_info[check_sp].from_variants) & within_variant_ids):
                del these_sp_info[check_sp]

        # calculate the observations
        observations = [len(this_sp_info.mapped_records) for this_sp_info in these_sp_info.values()]

        # sub path possible matches
        logger.debug("  Formulating the subpath probabilities ..")
        this_sbp_Xs = [these_sp_info[_go_sp_].num_possible_X for _go_sp_ in these_sp_info]
        for go_valid_sp, this_sp_info in enumerate(these_sp_info.values()):
            variant_weight = 0
            if within_variant_ids:
                sub_from_iso = {_go_iso_: _sp_freq_
                                for _go_iso_, _sp_freq_ in this_sp_info.from_variants.items()
                                if _go_iso_ in within_variant_ids}
                for go_variant, sp_freq in sub_from_iso.items():
                    variant_weight += variant_percents[go_variant] * sp_freq
            else:
                for go_variant, sp_freq in this_sp_info.from_variants.items():
                    variant_weight += variant_percents[go_variant] * sp_freq
            this_sbp_Xs[go_valid_sp] *= variant_weight
        this_sbp_prob = [_sbp_X / total_length for _sbp_X in this_sbp_Xs]

        # mark2, if include this, better removing code block under mark1
        # leading to nan like?
        # # the other unrecorded observed matches
        # observations.append(len(self.alignment.raw_records) - sum(observations))
        # # the other unrecorded expected matches
        # # Theano may not support sum, use for loop instead
        # other_prob = 1
        # for _sbp_prob in this_sbp_prob:
        #     other_prob -= _sbp_prob
        # this_sbp_prob.append(other_prob)

        for go_valid_sp, go_sp in enumerate(these_sp_info):
            logger.trace("  Subpath {} observation: {}".format(go_sp, observations[go_valid_sp]))
            logger.trace("  Subpath {} probability: {}".format(go_sp, this_sbp_prob[go_valid_sp]))
        # logger.trace("  Rest observation: {}".format(observations[-1]))
        # logger.trace("  Rest probability: {}".format(this_sbp_prob[-1]))

        # for go_sp, this_sp_info in these_sp_info.items():
        #     for record_id in this_sp_info.mapped_records:
        #         this_len_sp_xs = self.pal_len_sbp_Xs[self.alignment.raw_records[record_id].p_align_len]
        #         ...

        # likelihood
        logger.debug("  Summing up subpath likelihood function ..")
        loglike_expression = 0
        for go_sp, obs in enumerate(observations):
            loglike_expression += log_func(this_sbp_prob[go_sp]) * obs
        variable_size = len(within_variant_ids) if within_variant_ids else self.num_put_variants
        sample_size = sum(observations)

        return LogLikeFormulaInfo(loglike_expression, variable_size, sample_size)

    # def get_binomial_like_formula(self, variant_percents, log_func, within_variant_ids=None):
    #     """
    #     use a combination of multiple binormial distributions
    #     deprecated
    #     :param variant_percents:
    #          input sympy.Symbols for maximum likelihood analysis (scipy),
    #              e.g. [Symbol("P" + str(isomer_id)) for isomer_id in range(self.num_put_variants)].
    #          input pm.Dirichlet for bayesian analysis (pymc3),
    #              e.g. pm.Dirichlet(name="props", a=np.ones(isomer_num), shape=(isomer_num,)).
    #     :param log_func:
    #          input sympy.log for maximum likelihood analysis using scipy,
    #          inut tt.log for bayesian analysis using pymc3
    #     :param within_variant_ids:
    #          constrain the isomer testing scope. Test all isomers by default.
    #              e.g. set([0, 2])
    #     :return: LogLikeFormulaInfo object
    #     """
    #     # logger step
    #     total_sp_num = len(self.all_sub_paths)
    #     if total_sp_num > 200:
    #         logger_step = min(int(total_sp_num**0.5), 100)
    #     else:
    #         logger_step = 1
    #     # total length
    #     total_length = 0
    #     if within_variant_ids:
    #         for go_isomer, go_length in enumerate(self.variant_sizes):
    #             if go_isomer in within_variant_ids:
    #                 total_length += variant_percents[go_isomer] * float(go_length)
    #     else:
    #         for go_isomer, go_length in enumerate(self.variant_sizes):
    #             total_length += variant_percents[go_isomer] * float(go_length)
    #     # accumulate likelihood by sub path
    #     loglike_expression = 0
    #     variable_size = 0
    #     sample_size = 0
    #     for go_sp, (this_sub_path, this_sp_info) in enumerate(self.all_sub_paths.items()):
    #         if this_sp_info.num_possible_X < 1:
    #             continue
    #         total_starting_points = 0
    #         if within_variant_ids:
    #             # remove irrelevant isomers
    #             sub_from_iso = {_go_iso_: _sp_freq_
    #                             for _go_iso_, _sp_freq_ in this_sp_info.from_variants.items()
    #                             if _go_iso_ in within_variant_ids}
    #             if not sub_from_iso:
    #                 # drop unsupported sub paths
    #                 continue
    #             else:
    #                 for go_isomer, sp_freq in sub_from_iso.items():
    #                     total_starting_points += variant_percents[go_isomer] * sp_freq * this_sp_info.num_possible_X
    #         else:
    #             for go_isomer, sp_freq in this_sp_info.from_variants.items():
    #                 total_starting_points += variant_percents[go_isomer] * sp_freq * this_sp_info.num_possible_X
    #         this_prob = total_starting_points / total_length
    #         loglike_expression += this_sp_info.num_matched * log_func(this_prob) + \
    #                               (this_sp_info.num_in_range - this_sp_info.num_matched) * log_func(1 - this_prob)
    #         variable_size += 1
    #         sample_size += this_sp_info.num_in_range
    #         if go_sp % logger_step == 0:
    #             logger.debug("Summarized subpaths: %i/%i; Variables: %i; Samples: %i" %
    #                          (go_sp + 1, total_sp_num, variable_size, sample_size))
    #     logger.debug("Summarized subpaths: %i/%i; Variables: %i; Samples: %i" %
    #                  (total_sp_num, total_sp_num, variable_size, sample_size))
    #     return LogLikeFormulaInfo(loglike_expression, variable_size, sample_size)

    def fit_model_using_point_maximum_likelihood(self,
                                                 chosen_ids: typingODict[int, bool] = None):
        self.max_like_fit = ModelFitMaxLike(self)
        return self.max_like_fit.point_estimate(chosen_ids=chosen_ids)

    def fit_model_using_reverse_model_selection(self,
                                                criterion=Criterion.AIC,
                                                chosen_ids: typingODict[int, bool] = None):
        self.max_like_fit = ModelFitMaxLike(self)
        return self.max_like_fit.reverse_model_selection(criterion=criterion, chosen_ids=chosen_ids)

    # def update_candidate_info(self, ext_component_proportions: typingODict[int, float] = None):
    #     """
    #     ext_component_proportions: if provided, use external component proportions to update the candidate information
    #     """
    #     if ext_component_proportions:
    #         assert isinstance(ext_component_proportions, OrderedDict)
    #         self.variant_proportions = ext_component_proportions
    #
    #     self.variant_paths = []  # each element is a tuple(path)
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
    #     self.merged_variants = {}
    #
    #     for

    def fit_model_using_bayesian_mcmc(self, chosen_ids: typingODict[int, bool] = None):
        self.bayesian_fit = ModelFitBayesian(self)
        return self.bayesian_fit.run_mcmc(self.kwargs["n_generations"], self.kwargs["n_burn"], chosen_ids=chosen_ids)

    def output_seqs(self):
        out_seq_num = len([x for x in self.variant_proportions.values() if x > self.out_prob_threshold])
        out_digit = len(str(out_seq_num))
        count_seq = 0
        sorted_rank = sorted(list(self.variant_proportions), key=lambda x: -self.variant_proportions[x])
        # for go_isomer, this_prob in self.ext_component_proportions.items():
        for count_seq, go_variant_set in enumerate(sorted_rank):
            this_prob = self.variant_proportions[go_variant_set]
            if this_prob > self.out_prob_threshold:
                this_base_name = "variant.%0{}i".format(out_digit) % (count_seq + 1)
                seq_file_name = os.path.join(self.outdir, this_base_name + ".fasta")
                with open(seq_file_name, "w") as output_handler:
                    if self.merged_variants:
                        unidentifiable_ids = self.merged_variants[go_variant_set]
                    else:
                        # when self.generate_all_informative_sub_paths() was not called
                        unidentifiable_ids = [go_variant_set]
                    len_un_id = len(unidentifiable_ids)
                    freq_mark = "@{}.".format(count_seq + 1) if len_un_id > 1 else ""
                    lengths = []
                    for go_ss, comp_id in enumerate(unidentifiable_ids):
                        this_seq = self.graph.export_path(self.variant_paths[comp_id], check_valid=False)
                        this_len = len(this_seq.seq)
                        lengths.append(this_len)
                        if freq_mark:
                            seq_label = ">{} freq={:.4f}{}{} len={}bp".format(
                                this_seq.label, this_prob, freq_mark, go_ss, this_len)
                            logger.debug("{}.{} path={}".format(this_base_name, go_ss, this_seq.label))
                        else:
                            seq_label = ">{} freq={:.4f} len={}bp".format(this_seq.label, this_prob, this_len)
                            logger.debug("{} path={}".format(this_base_name, this_seq.label))
                        output_handler.write(seq_label + "\n" + this_seq.seq + "\n")
                        count_seq += 1
                    logger.info("{}x{} freq:{:.4f} len:{}".format(
                        this_base_name, len_un_id, this_prob, "/".join([str(_l) for _l in lengths])))
                    # if len_un_id > 1:
                    #     logger.info("{} freq:{:.4f} len:{} unidentifiable:{}".format(
                    #         this_base_name, this_prob, "/".join([str(_l) for _l in lengths]), len_un_id))
                    # else:
                    #     logger.info("{} freq:{:.4f} len:{}".format(
                    #         this_base_name, this_prob, "/".join([str(_l) for _l in lengths])))
        logger.info("Output {} seqs (%.4f to %.{}f): ".format(count_seq, len(str(self.out_prob_threshold)) - 2)
                    % (max(self.variant_proportions.values()), self.out_prob_threshold))

    def shuffled(self, sorted_list):
        sorted_list = deepcopy(sorted_list)
        self.random.shuffle(sorted_list)
        return sorted_list

    def setup_timed_logger(self, loglevel="INFO"):
        """
        Configure Loguru to log to stdout and logfile.
        """
        # add stdout logger
        timed_config = {
            "handlers": [
                {
                    "sink": sys.stdout, 
                    "format": (
                        "{time:YYYY-MM-DD-HH:mm:ss.SS} | "
                        "<magenta>{file: >20} | </magenta>"
                        "<cyan>{function: <30} | </cyan>"
                        "<level>{message}</level>"
                    ),
                    "level": loglevel,
                    },
                {
                    "sink": self.logfile,                   
                    "format": "{time:YYYY-MM-DD-HH:mm:ss.SS} | "
                              "<magenta>{file: >20} | </magenta>"
                              "<cyan>{function: <30} | </cyan>"
                              "<level>{message}</level>",
                    "level": loglevel,
                    }
            ]
        }
        logger.configure(**timed_config)
        logger.enable("traversome")

        # # if logfile exists then reset it.
        # if os.path.exists(self.logfile):
        #     logger.debug('Clearing previous log file.')
        #     open(self.logfile, 'w').close()

