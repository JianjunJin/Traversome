#!/usr/bin/env python
import os

from loguru import logger
from traversome.utils import harmony_weights, run_dill_encoded   # MaxTraversalReached
# WeightedGMMWithEM find_greatest_common_divisor,
from copy import deepcopy
from pathlib import Path as fpath
from scipy.stats import norm
from collections import OrderedDict
import numpy as np
from numpy import log, exp, inf
from multiprocessing import Manager, Pool
import warnings
import dill
import random
import json
import traceback
import sys


# suppress numpy warnings at exp()
warnings.filterwarnings('ignore')


class SingleTraversal(object):
    """

    """

    def __init__(self, path_generator_obj, random_seed):
        self.graph = path_generator_obj.graph
        self.read_paths = path_generator_obj.read_paths
        self.local_max_alignment_len = path_generator_obj.max_alignment_len
        self.contig_coverages = path_generator_obj.contig_coverages
        self.uni_chromosome = path_generator_obj.uni_chromosome
        self.__starting_subpath_to_readpaths = path_generator_obj.pass_starting_subpath_to_readpaths()
        self.__middle_subpath_to_readpaths = path_generator_obj.pass_middle_subpath_to_readpaths()
        self.__read_paths_counter = path_generator_obj.pass_read_paths_counter()
        self.__differ_f = path_generator_obj.pass_differ_f()
        self.__cov_inert = path_generator_obj.pass_cov_inert()
        self.__decay_f = path_generator_obj.pass_decay_f()
        self.__decay_t = path_generator_obj.pass_decay_t()
        self.__candidate_single_copy_vs = path_generator_obj.pass_candidate_single_copy_vs()
        random.seed(random_seed)
        self.result_path = None

    def run(self, start_path=None):
        start_path = [] if start_path is None else start_path
        self.result_path = \
            self.graph.get_standardized_path_circ(self.graph.roll_path(self.__heuristic_extend_path(start_path)))

    def __heuristic_extend_path(
            self,
            path,
            # not_do_reverse=False
    ):
        """
        :param path: empty path like [] or starting path like [("v1", True), ("v2", False)]

        # :param not_do_reverse: mainly for iteration, a mark to stop searching from the reverse end

        :return: a candidate variant path. e.g. [("v0", True), ("v1", True), ("v2", False), ("v3", True)]
        """
        if not path:
            # randomly choose the read path and the direction
            # # change the weight (-~ depth) to flatten the search
            # read_p_freq_reciprocal = [1. / self.__read_paths_counter[r_p] for r_p in self.read_paths]
            # read_path = random.choices(self.read_paths, weights=read_p_freq_reciprocal)[0]
            read_path = random.choices(self.read_paths)[0]
            if random.random() > 0.5:
                read_path = self.graph.reverse_path(read_path)
            path = list(read_path)
            logger.trace("      starting path(" + str(len(path)) + "): " + str(path))
        not_do_reverse = False
        while True:
            #
            #     # initial_mean, initial_std = self.__get_cov_mean(read_path, return_std=True)
            #     return self.__heuristic_extend_path(
            #         path=path,
            #         not_do_reverse=False)
            # else:

            # keep going in a circle util the path length reaches beyond the longest read alignment
            # stay within what data can tell
            # 2023-07-23 remove this arbitrary setting and move roll_path after proposal
            # repeating_unit = self.graph.roll_path(path)
            # if len(path) > len(repeating_unit) and \
            #         self.graph.get_path_internal_length(path) >= self.local_max_alignment_len:
            #     logger.trace("      traversal ended within a circle unit.")
            #     return deepcopy(repeating_unit)

            #
            current_ave_coverage = self.__get_cov_mean(path)
            # generate the extending candidates
            candidate_ls_list = []
            candidates_list_overlap_c_nums = []
            for overlap_c_num in range(2, len(path) + 1):  # TODO: set start to be 2 because 1 is like next_connections
                overlap_path = path[-overlap_c_num:]
                # stop adding extending candidate when the overlap is longer than our longest read alignment
                # stay within what data can tell
                if self.graph.get_path_internal_length(list(overlap_path) + [("", True)]) \
                        >= self.local_max_alignment_len:
                    break
                overlap_path = tuple(overlap_path)
                if overlap_path in self.__starting_subpath_to_readpaths:
                    # logger.debug("starting, " + str(self.__start_subpath_to_readpaths[overlap_path]))
                    candidate_ls_list.append(sorted(self.__starting_subpath_to_readpaths[overlap_path]))
                    candidates_list_overlap_c_nums.append(overlap_c_num)
            # logger.debug(candidate_ls_list)
            # logger.debug(candidates_list_overlap_c_nums)
            if not candidate_ls_list:
                # if no extending candidate based on starting subpath (from one end),
                # try to simultaneously extend both ends from middle
                path_tuple = tuple(path)
                if path_tuple in self.__middle_subpath_to_readpaths:
                    candidates = sorted(self.__middle_subpath_to_readpaths[path_tuple])
                    weights = [self.__read_paths_counter[self.read_paths[read_id]] for read_id, strand in candidates]
                    weights = harmony_weights(weights, diff=self.__differ_f)
                    if self.__cov_inert:
                        cdd_cov = [self.__get_cov_mean(self.read_paths[read_id], exclude_path=path)
                                   for read_id, strand in candidates]
                        weights = [exp(log(weights[go_c]) - abs(log(cov / current_ave_coverage))) * self.__cov_inert
                                   for go_c, cov in enumerate(cdd_cov)]
                    read_id, strand = random.choices(candidates, weights=weights)[0]
                    if strand:
                        path = list(self.read_paths[read_id])
                    else:
                        path = list(self.graph.reverse_path(self.read_paths[read_id]))
                    continue
                    # 2023-03-23: find a bug in previous recursive code
                    # return self.__heuristic_extend_path(path)

            # like_ls_cached will be calculated in if self.uni_chromosome
            # it may be further used in self.__heuristic_check_multiplicity()
            like_ls_cached = []
            if not candidate_ls_list:
                # if no extending candidates based on overlap info, try to extend based on the graph
                logger.trace("      no extending candidates based on overlap info, try extending based on the graph")
                last_name, last_end = path[-1]
                next_connections = self.graph.vertex_info[last_name].connections[last_end]
                logger.trace("      {}, {}: next_connections: {}".format(last_name, last_end, next_connections))
                if next_connections:
                    if len(next_connections) > 1:
                        candidates_next = sorted(next_connections)
                        logger.trace("      candidates_next: {}".format(candidates_next))
                        if self.uni_chromosome:
                            # weighting candidates by the likelihood change of the multiplicity change
                            old_cov_mean, old_cov_std = self.__get_cov_mean(path, return_std=True)
                            logger.trace("      path mean:" + str(old_cov_mean) + "," + str(old_cov_std))
                            single_cov_mean, single_cov_std = self.__get_cov_mean_of_single(path, return_std=True)
                            logger.trace("      path single mean:" + str(single_cov_mean) + "," + str(single_cov_std))
                            current_vs = [v_n for v_n, v_e in path]
                            weights = []
                            for next_v in candidates_next:
                                v_name, v_end = next_v
                                current_v_counts = {v_name: current_vs.count(v_name)}
                                loglike_ls = self.__cal_multiplicity_like(path=path,
                                                                          proposed_extension=[next_v],
                                                                          current_v_counts=current_v_counts,
                                                                          old_cov_mean=old_cov_mean,
                                                                          old_cov_std=old_cov_std,
                                                                          single_cov_mean=single_cov_mean,
                                                                          single_cov_std=single_cov_std,
                                                                          logarithm=True)
                                like_ls_cached.append(exp(loglike_ls))
                                weights.append(max(loglike_ls))  # the best scenario
                            logger.trace("      likes: {}".format(weights))
                            #
                            weights = exp(np.array(weights))
                            # ratio: likelihood proportion
                            weights = np.where(weights == np.inf, 1, weights / (1. + weights))
                            weights = weights / max(weights)
                            chosen_cdd_id = random.choices(range(len(candidates_next)), weights=weights)[0]
                            next_name, next_end = candidates_next[chosen_cdd_id]
                            like_ls_cached = like_ls_cached[chosen_cdd_id]
                        elif self.__cov_inert:
                            # coverage inertia (multi-chromosomes) and uni_chromosome are mutually exclusive
                            # coverage inertia, more likely to extend to contigs with similar depths,
                            # which are more likely to be the same target chromosome / organelle type
                            cdd_cov = [self.contig_coverages[_n_] for _n_, _e_ in candidates_next]
                            weights = [exp(-abs(log(cov / current_ave_coverage))) * self.__cov_inert for cov in cdd_cov]
                            logger.trace("      likes: {}".format(weights))
                            next_name, next_end = random.choices(candidates_next, weights=weights)[0]
                        else:
                            next_name, next_end = random.choice(candidates_next)
                    else:
                        next_name, next_end = list(next_connections)[0]
                        logger.trace("      single next: ({}, {})".format(next_name, next_end))
                        like_ls_cached = None
                    # if not self.uni_chromosome or
                    # self.graph.is_fully_covered_by(path + [(next_name, not next_end)]):
                    path, keep_extend, not_do_reverse = self.__heuristic_check_multiplicity(
                        path=path,
                        proposed_extension=[(next_name, not next_end)],
                        not_do_reverse=not_do_reverse,
                        cached_like_ls=like_ls_cached
                    )
                    if keep_extend:
                        continue
                    else:
                        return path
                else:
                    if not_do_reverse:
                        logger.trace("      traversal ended without next vertex.")
                        return path
                    else:
                        logger.trace("      traversal reversed without next vertex.")
                        path = list(self.graph.reverse_path(path))
                        not_do_reverse = True
                        continue
                        # return self.__heuristic_extend_path(
                        #     list(self.graph.reverse_path(path)),
                        #     not_do_reverse=True)
            else:
                logger.debug("    path(" + str(len(path)) + "): " + str(path))
                # if there is only one candidate
                if len(candidate_ls_list) == 1 and len(candidate_ls_list[0]) == 1:
                    read_id, strand = candidate_ls_list[0][0]
                    ovl_c_num = candidates_list_overlap_c_nums[0]
                else:
                    candidates = []
                    candidates_ovl_n = []
                    weights = []
                    num_reads_used = 0
                    max_ovl = max(candidates_list_overlap_c_nums)
                    for go_overlap, same_ov_cdd in enumerate(candidate_ls_list):
                        # flatten the candidate_ls_list:
                        # each item in candidate_ls_list is a list of candidates of the same overlap contig
                        candidates.extend(same_ov_cdd)
                        # record the overlap contig numbers in the flattened single-dimension candidates
                        ovl_c_num = candidates_list_overlap_c_nums[go_overlap]
                        candidates_ovl_n.extend([ovl_c_num] * len(same_ov_cdd))
                        # generate the weights for the single-dimension candidates
                        same_ov_w = [self.__read_paths_counter[self.read_paths[read_id]]
                                     for read_id, strand in same_ov_cdd]
                        num_reads_used += sum(same_ov_w)
                        same_ov_w = harmony_weights(same_ov_w, diff=self.__differ_f)
                        # RuntimeWarning: overflow encountered in exp of numpy, or math range error in exp of math
                        # change to dtype=np.float128?
                        same_ov_w = exp(np.array(log(same_ov_w) - log(self.__decay_f) * (max_ovl - ovl_c_num),
                                                 dtype=np.float128))
                        weights.extend(same_ov_w)
                        # To reduce computational burden, only reads that overlap most with current path
                        # will be considered in the extension. Proportions below 1/decay_t which will be neglected
                        # in the path proposal.
                        if num_reads_used >= self.__decay_t:
                            break
                    logger.trace("       Drawing candidates from {} reads, with [{},{}] overlaps".format(
                        num_reads_used, min(candidates_ovl_n), max(candidates_ovl_n)))
                    if self.uni_chromosome:
                        ######
                        # randomly chose a certain number of candidates to reduce computational burden
                        # then, re-weighting candidates by the likelihood change of adding the extension
                        ######
                        try:
                            pool_size = 10  # arbitrary pool size for re-weighting
                            pool_ids = random.choices(range(len(candidates)), weights=weights, k=pool_size)
                        except ValueError:  # TODO ValueError: Total of weights must be finite
                            pool_ids = list(range(len(candidates)))
                        pool_ids_set = set(pool_ids)
                        if len(pool_ids_set) == 1:
                            remaining_id = pool_ids_set.pop()
                            candidates = [candidates[remaining_id]]
                            candidates_ovl_n = [candidates_ovl_n[remaining_id]]
                            weights = [1.]
                        else:
                            # count the previous sampling and convert it into a new weights
                            new_candidates = []
                            new_candidates_ovl_n = []
                            new_weights = []
                            for remaining_id in sorted(pool_ids_set):
                                new_candidates.append(candidates[remaining_id])
                                new_candidates_ovl_n.append(candidates_ovl_n[remaining_id])
                                new_weights.append(pool_ids.count(remaining_id))
                            candidates = new_candidates
                            candidates_ovl_n = new_candidates_ovl_n
                            weights = new_weights
                            # re-weighting candidates by the likelihood change of adding the extension
                            current_vs = [v_n for v_n, v_e in path]
                            old_cov_mean, old_cov_std = self.__get_cov_mean(path, return_std=True)
                            logger.trace("      path({}): {}".format(len(path), path))
                            logger.trace("      path mean:" + str(old_cov_mean) + "," + str(old_cov_std))
                            single_cov_mean, single_cov_std = self.__get_cov_mean_of_single(path, return_std=True)
                            logger.trace("      path single mean:" + str(single_cov_mean) + "," + str(single_cov_std))
                            for go_c, (read_id, strand) in enumerate(candidates):
                                read_path = self.read_paths[read_id]
                                logger.trace("      candidate r-path {}: {}: {}".format(go_c, strand, read_path))
                                if not strand:
                                    read_path = list(self.graph.reverse_path(read_path))
                                cdd_extend = read_path[candidates_ovl_n[go_c]:]
                                logger.trace("      candidate ext {}: {}".format(go_c, cdd_extend))
                                current_v_counts = {_v_n: current_vs.count(_v_n) for _v_n, _v_e in cdd_extend}
                                like_ls = self.__cal_multiplicity_like(path=path,
                                                                       proposed_extension=cdd_extend,
                                                                       current_v_counts=current_v_counts,
                                                                       old_cov_mean=old_cov_mean,
                                                                       old_cov_std=old_cov_std,
                                                                       single_cov_mean=single_cov_mean,
                                                                       single_cov_std=single_cov_std)
                                like_ls_cached.append(like_ls)
                                max_like = max(like_ls)
                                weights[go_c] *= 1. if max_like == inf else max_like / (1. + max_like)
                            logger.trace("      like_ls_cached: {}".format(like_ls_cached))
                    elif self.__cov_inert:
                        # coverage inertia (multi-chromosomes) and uni_chromosome are mutually exclusive
                        # coverage inertia, more likely to extend to contigs with similar depths,
                        # which are more likely to be the same target chromosome / organelle type
                        # logger.debug(candidates)
                        # logger.debug(candidates_ovl_n)
                        cdd_cov = [self.__get_cov_mean(self.read_paths[r_id][candidates_ovl_n[go_c]:])
                                   for go_c, (r_id, r_strand) in enumerate(candidates)]
                        weights = exp(np.array([log(weights[go_c]) - abs(log(cov / current_ave_coverage))
                                                for go_c, cov in enumerate(cdd_cov)], dtype=np.float128))
                    try:
                        chosen_cdd_id = random.choices(range(len(candidates)), weights=weights)[0]
                    except ValueError:
                        print(weights)
                        print("---------------")
                        raise Exception
                    if like_ls_cached:
                        like_ls_cached = like_ls_cached[chosen_cdd_id]
                    else:
                        like_ls_cached = None
                    read_id, strand = candidates[chosen_cdd_id]
                    ovl_c_num = candidates_ovl_n[chosen_cdd_id]
                read_path = self.read_paths[read_id]
                logger.trace("      read_path({}-{})({}): {}".format(ovl_c_num, len(read_path), strand, read_path))
                if not strand:
                    read_path = list(self.graph.reverse_path(read_path))
                new_extend = read_path[ovl_c_num:]
                # if not strand:
                #     new_extend = list(self.graph.reverse_path(new_extend))

                logger.debug("      potential extend(" + str(len(new_extend)) + "): " + str(new_extend))
                # input("pause")
                # logger.debug("closed_from_start: " + str(closed_from_start))
                # logger.debug("    candidate path: {} .. {} .. {}".format(path[:3], len(path), path[-3:]))
                # logger.debug("    extend path   : {}".format(new_extend))

                # if not self.uni_chromosome or self.graph.is_fully_covered_by(path + new_extend):
                path, keep_extend, not_do_reverse = self.__heuristic_check_multiplicity(
                    # initial_mean=initial_mean,
                    # initial_std=initial_std,
                    path=path,
                    proposed_extension=new_extend,
                    not_do_reverse=not_do_reverse,
                    cached_like_ls=like_ls_cached)
                if keep_extend:
                    continue
                else:
                    return path
                # else:
                #     return self.__heuristic_extend_path(
                #         path + new_extend,
                #         not_do_reverse=not_do_reverse,
                #         initial_mean=initial_mean,
                #         initial_std=initial_std)

    def __cal_multiplicity_like(
            self,
            path,
            proposed_extension,
            current_v_counts=None,
            old_cov_mean=None,
            old_cov_std=None,
            single_cov_mean=None,
            single_cov_std=None,
            logarithm=False,
    ):
        """
        called by __heuristic_extend_path through directly or through self.__heuristic_check_multiplicity
        :param path:
        :param proposed_extension:
        :param current_v_counts: passed to avoid repeated calculation
        :param old_cov_mean: passed to avoid repeated calculation
        :param old_cov_std: passed to avoid repeated calculation
        :param single_cov_mean: passed to avoid repeated calculation
        :param single_cov_std: passed to avoid repeated calculation
        :return: log_like_ratio_list
        """
        path = list(deepcopy(path))
        if not current_v_counts:
            current_vs = [v_n for v_n, v_e in path]
            current_v_counts = {v_n: current_vs.count(v_n)
                                for v_n in set([v_n for v_n, v_e in proposed_extension])}
        if not (old_cov_mean and old_cov_std):
            old_cov_mean, old_cov_std = self.__get_cov_mean(path, return_std=True)
            logger.debug("        path mean:" + str(old_cov_mean) + "," + str(old_cov_std))
            # logger.debug("initial mean: " + str(initial_mean) + "," + str(initial_std))
        # logger.trace("    old_path: {}".format(self.graph.repr_path(path)))
        # logger.trace("    checking proposed_extension: {}".format(self.graph.repr_path(proposed_extension)))
        # use single-copy mean and std instead of initial
        if not (single_cov_mean and single_cov_std):
            single_cov_mean, single_cov_std = self.__get_cov_mean_of_single(path, return_std=True)
            logger.debug("        path single mean:" + str(single_cov_mean) + "," + str(single_cov_std))

        logger.trace("        current_v_counts:{}".format(current_v_counts))
        # check the multiplicity of every vertices
        # check the likelihood of varying size of extension
        log_like_ratio_list = []
        log_like_ratio = 0.
        proposed_lengths = {_v_: self.graph.vertex_info[_v_].len for _v_, _e_ in proposed_extension}
        _old_like_cache = {}  # avoid duplicate calculation
        for v_name, v_end in proposed_extension:
            current_c = current_v_counts[v_name]
            if v_name in _old_like_cache:
                old_like = _old_like_cache[v_name]
            else:
                if current_c:
                    old_single_cov = self.contig_coverages[v_name] / float(current_c)
                    logger.trace("        contig_cov: {}".format(self.contig_coverages[v_name]))
                    logger.trace("        old_single_cov: {}".format(old_single_cov))
                    # old_like = norm.logpdf(old_single_cov, loc=old_cov_mean, scale=old_cov_std)
                    # old_like += norm.logpdf(old_single_cov, loc=single_cov_mean, scale=single_cov_std)
                    old_like = norm.logpdf(old_single_cov, loc=single_cov_mean, scale=single_cov_std)
                else:
                    old_like = -inf
                _old_like_cache[v_name] = old_like
            # new_cov_mean, new_cov_std = self.__get_cov_mean(path + [(v_name, v_end)], return_std=True)
            if current_c:
                new_single_cov = self.contig_coverages[v_name] / float(current_c + 1)
                logger.trace("        new_single_cov: {}".format(new_single_cov))
                # new_like = norm.logpdf(new_single_cov, loc=new_cov_mean, scale=new_cov_std)
                # new_like += norm.logpdf(new_single_cov, loc=single_cov_mean, scale=single_cov_std)
                new_like = norm.logpdf(new_single_cov, loc=single_cov_mean, scale=single_cov_std)
            else:
                new_like = 0.
            # old_cov_mean, old_cov_std = new_cov_mean, new_cov_std
            # weighted by log(length), de-weight later for comparison
            logger.trace("        unweighted loglike ratio: {}".format(new_like - old_like))
            logger.trace("        weighting by length: {}".format(proposed_lengths[v_name]))
            logger.trace("        weighted loglike ratio: {}".format((new_like - old_like) * proposed_lengths[v_name]))
            log_like_ratio += (new_like - old_like) * proposed_lengths[v_name]
            # probability higher than 1.0 (log_ratio > 0.) will remain as 1.0 (log_ratio as 0.)
            # log_like_ratio = min(0, log_like_ratio)  # 2023-04-13 bug fixes
            log_like_ratio_list.append(log_like_ratio)
            # logger.trace("      initial_mean: %.4f, old_mean: %.4f (%.4f), proposed_mean: %.4f (%.4f)" % (
            #     initial_mean, old_cov_mean, old_cov_std, new_cov_mean, new_cov_std))
            # logger.trace("      old_like: {},     proposed_like: {}".format(old_like, new_like))

            # updating path so that new_cov_mean, new_cov_std will be updated
            path.append((v_name, v_end))
        # de-weight the log likes for comparison
        longest_ex_len = len(proposed_extension)
        v_lengths = [proposed_lengths[_v_] for _v_, _e_ in proposed_extension]
        accumulated_v_lengths = []
        for rev_go in range(len(log_like_ratio_list)):
            accumulated_v_lengths.insert(0, sum(v_lengths[:longest_ex_len - rev_go]))
        # logger.trace("    proposed_lengths: {}".format(proposed_lengths))
        # logger.trace("    accumulated_v_lengths: {}".format(accumulated_v_lengths))
        # TODO: weighting and de-weighting can be problematic
        log_like_ratio_list = [_llr / accumulated_v_lengths[_go] for _go, _llr in enumerate(log_like_ratio_list)]
        if logarithm:
            return np.array(log_like_ratio_list, dtype=np.float128)
        else:
            return exp(np.array(log_like_ratio_list, dtype=np.float128))

    def __heuristic_check_multiplicity(
            self, path, proposed_extension, not_do_reverse, current_v_counts=None, cached_like_ls=None):
        """
        heuristically check the multiplicity and call a stop according to the vertex coverage and current counts
        normal distribution
        :param path:
        :param proposed_extension:
        :param not_do_reverse: a mark to stop searching from the reverse end
        :param current_v_counts: Dict
        :param cached_like_ls: input cached likelihood ratio list instead of recalculating it

        :return: path:list, keep_extend:bool, not_do_reverse:bool
        """
        assert len(proposed_extension)
        # if there is a vertex of proposed_extension that was not used in current path,
        # accept the proposed_extension without further calculation
        if not current_v_counts:
            current_vs = [v_n for v_n, v_e in path]
            current_v_counts = {_v_n: current_vs.count(_v_n) for _v_n, _v_e in proposed_extension}
        if not (cached_like_ls is None or cached_like_ls == []):
            like_ratio_list = cached_like_ls
        else:
            like_ratio_list = self.__cal_multiplicity_like(
                path=path, proposed_extension=proposed_extension, current_v_counts=current_v_counts)
        # step-by-step shorten the extension
        # Given that the acceptance rate should be P_n=\prod_{i=1}^{n}{x_i} for extension with length of n,
        # where x_i is the probability of accepting contig i,
        # each intermediate random draw should follow the format below to consider the influence of accepting longer
        # extension
        #    d_{n-1} = (P_{n-1} - P_n) / (1 - P_n)
        longest_ex_len = len(proposed_extension)
        previous_like = 0.
        prob_list = np.where(like_ratio_list == np.inf, 1, like_ratio_list / (1. + like_ratio_list))  # 2023-04-13 bug fixes
        for rev_go, like_ratio in enumerate(prob_list[::-1]):  # 2023-04-13 bug fixes
            proposed_end = longest_ex_len - rev_go
            if rev_go == 0:
                # the first draw
                draw_prob = like_ratio
            else:
                draw_prob = (like_ratio - previous_like) / (1. - previous_like)
            previous_like = like_ratio
            logger.trace("      draw prob:{}".format(draw_prob))
            if draw_prob > random.random():
                logger.trace("      draw accepted:{}".format(proposed_extension[:proposed_end]))
                return list(deepcopy(path)) + list(proposed_extension[:proposed_end]), True, not_do_reverse
        else:
            # Tested to be a bad idea
            # if self.force_circular:
            #     if self.graph.is_circular_path(self.graph.roll_path(path)):
            #         logger.trace("        circular traversal ended to fit {}'s coverage.".format(
            #                      proposed_extension[0][0]))
            #         logger.trace("        checked likes: {}".format(like_ratio_list))
            #         return list(deepcopy(path))
            #     else:
            #         # do not waste current search, extend anyway
            #         return self.__heuristic_extend_path(
            #             list(deepcopy(path)) + list(proposed_extension[:1 + np.argmax(np.array(like_ratio_list))]),
            #             not_do_reverse=not_do_reverse)
            # else:
            if not_do_reverse:
                logger.trace("        linear traversal ended to fit {}'s coverage.".format(proposed_extension[0][0]))
                logger.trace("        checked likes: {}".format(like_ratio_list))
                return list(deepcopy(path)), False, None
            else:
                logger.trace("        linear traversal reversed to fit {}'s coverage.".format(proposed_extension[0][0]))
                logger.trace("        checked likes: {}".format(like_ratio_list))
                return list(self.graph.reverse_path(list(deepcopy(path)))), True, True

    def __check_path(self, path):
        assert len(path)
        try:
            for v_name, v_e in path:
                if v_name not in self.graph.vertex_info:
                    raise Exception(v_name + " not found in the assembly graph!")
        except Exception as e:
            logger.error("Invalid path: " + str(path))
            raise e

    def __get_cov_mean_of_single(self, path, return_std=False):
        """for approximate single-copy contigs"""
        self.__check_path(path)
        v_names = [v_n for v_n, v_e in path]
        v_names = {v_n: v_names.count(v_n) for v_n in set(v_names)}
        min_cov = min(v_names.values())
        v_names = [v_n for v_n, v_c in v_names.items() if v_c == min_cov]
        # use the graph structure information
        single_copy_likely = set(v_names) & self.__candidate_single_copy_vs
        if single_copy_likely:
            v_names = sorted(single_copy_likely)
        # calculate
        v_covers = []
        v_lengths = []
        for v_name in v_names:
            v_covers.append(self.contig_coverages[v_name])
            v_lengths.append(self.graph.vertex_info[v_name].len)
        mean = np.average(v_covers, weights=v_lengths)
        if return_std:
            if len(v_covers) > 1:
                std = np.average((np.array(v_covers) - mean) ** 2, weights=v_lengths) ** 0.5
                if std != 0:
                    return mean, std
                else:
                    return mean, mean * 0.1  # arbitrary set unknown to be mean * 0.1
            else:
                return mean, mean * 0.1  # arbitrary set unknown to be mean * 0.1
        else:
            return mean

    def __get_cov_mean(self, path, exclude_path=None, return_std=False):
        self.__check_path(path)
        v_names = [v_n for v_n, v_e in path]
        v_names = {v_n: v_names.count(v_n) for v_n in set(v_names)}
        if exclude_path:
            del_names = [v_n for v_n, v_e in exclude_path]
            del_names = {v_n: del_names.count(v_n) for v_n in set(del_names)}
            for del_n in del_names:
                if del_names[del_n] > v_names.get(del_n, 0):
                    logger.error("cannot exclude {} from {}: unequal in {}".format(exclude_path, path, del_n))
                else:
                    v_names[del_n] -= del_names[del_n]
                    if v_names[del_n] == 0:
                        del v_names[del_n]
        v_covers = []
        v_lengths = []
        for v_name in v_names:
            v_covers.append(self.contig_coverages[v_name] / float(v_names[v_name]))
            v_lengths.append(self.graph.vertex_info[v_name].len * v_names[v_name])
        # logger.trace("        > cal path: {}".format(self.graph.repr_path(path)))
        # logger.trace("        > cover values: {}".format(v_covers))
        # logger.trace("        > cover weights: {}".format(v_lengths))
        mean = np.average(v_covers, weights=v_lengths)
        if return_std:
            if len(v_covers) > 1:
                std = np.average((np.array(v_covers) - mean) ** 2, weights=v_lengths) ** 0.5
                if std != 0:
                    return mean, std
                else:
                    return mean, mean * 0.1  # arbitrary set to be mean * 0.1
            else:
                return mean, mean * 0.1  # arbitrary set to be mean * 0.1
        else:
            return mean


class PathGenerator(object):
    """
    generate heuristic variants (isomers & sub-chromosomes) from alignments.
    Here the variants are not necessarily identical in contig composition.
    TODO automatically estimate mum_valid_search using convergence-test like approach
    """

    def __init__(self,
                 traversome_obj,
                 min_num_valid_search=1000,
                 max_num_valid_search=100001,
                 num_processes=1,
                 force_circular=True,
                 uni_chromosome=True,
                 differ_f=1.,
                 decay_f=20.,
                 decay_t=1000,
                 cov_inert=1.,
                 use_alignment_cov=False,
                 min_unit_similarity=0.85,
                 resume=False,
                 temp_dir: fpath = None):
        """
        :param traversome_obj: traversome object
        :param min_num_valid_search: minimum number of valid searches
        :param max_num_valid_search: maximum number of valid searches
        :param force_circular: force the generated variant topology to be circular
        :param uni_chromosome: a variant is NOT allowed to only traverse part of the graph.
            Different variants must be composed of identical set of contigs if uni_chromosome=True.
        :param differ_f: difference factor [0, INF)
            Weighted by which, reads with the same overlap with current path will be used according to their counts.
            new_weights = (count_weights^differ_f)/sum(count_weights^differ_f)
            Zero leads to that the read counts play no effect in read choice.
        :param decay_f: decay factor [0, INF]
            Chance reduces by which, a read with less overlap with current path will be used to extend current path.
            probs_(N-m) = probs_(N) * decay_f^(-m)
            A large value leads to strictly following read paths.
        :param decay_t: decay threshold for number of reads [100, INF]
            Number of reads. Only reads that overlap most with current path will be considered in the extension.
            # also server as a cutoff version of decay_f
        :param cov_inert: coverage inertia [0, INF)
            The degree of tendency a path has to extend contigs with similar coverage.
            weight *= exp(-abs(log(extending_coverage/current_path_coverage)))^cov_inert
            Designed for the mixture of multiple-sourced genomes, e.g. plastome and mitogenome.
            Set to zero if the graph is a single-sourced graph.
        :param use_alignment_cov: use the coverage from assembly graph if False.
        :param min_unit_similarity: minimum contig-len-weighted path similarity shared among units [0.5, 1]
            Used to trigger the decomposition of a long path concatenated by multiple units.
        :param resume: resume a previous run
        :param temp_dir: directory recording the generated paths for resuming and debugging
        """
        assert 1 <= num_processes
        assert 0 <= differ_f
        assert 0 <= decay_f
        assert 100 <= decay_t
        assert 0 <= cov_inert
        assert 0.5 <= min_unit_similarity
        self.graph = traversome_obj.graph
        # self.alignment = traversome_obj.alignment
        # self.tvs = traversome_obj
        self.max_alignment_len = traversome_obj.max_alignment_length
        self.subpath_generator = traversome_obj.subpath_generator
        self.min_valid_search = min_num_valid_search
        self.max_valid_search = max_num_valid_search
        self.num_processes = num_processes
        self.resume = resume
        self.temp_dir = temp_dir
        self.force_circular = force_circular
        self.uni_chromosome = uni_chromosome
        self.__differ_f = differ_f
        self.__decay_f = decay_f
        self.__decay_t = decay_t
        self.__cov_inert = cov_inert
        self.__random = traversome_obj.random
        self.__use_alignment_cov = use_alignment_cov
        self.__min_unit_similarity = min_unit_similarity

        # to be generated
        self.read_paths = list()
        self.__read_paths_counter = dict()
        for read_path, loc_ids in traversome_obj.read_paths.items():  # make a simplified copy of tvs.read_paths
            self.read_paths.append(read_path)
            self.__read_paths_counter[read_path] = len(loc_ids)
        # self.__vertex_to_readpath = {vertex: set() for vertex in self.graph.vertex_info}
        self.__start_subpath_to_readpaths = {}
        self.__middle_subpath_to_readpaths = {}
        self.__read_paths_not_in_variants = {}
        self.__read_paths_counter_indexed = False
        self.contig_coverages = OrderedDict()
        # self.single_copy_vertices_prob = \
        #     OrderedDict([(_v, 1.) for _v in single_copy_vertices]) if single_copy_vertices \
        #         else OrderedDict()
        self.__candidate_single_copy_vs = set()
        self.__previous_len_variant = 0
        self.count_valid = 0
        self.count_search = 0
        self.variants = list()
        self.variants_counts = dict()

        if self.temp_dir:
            self.temp_dir.mkdir(exist_ok=self.resume)

    def generate_heuristic_paths(self, num_processes=None):
        # load previous
        self.__previous_len_variant = 0
        if self.resume and self.temp_dir.exists():
            self.load_temp()
            if sum(self.variants_counts.values()) >= self.max_valid_search:  # hit the hard bound
                logger.info("Maximum num of valid searches reached.")
                return

        self.index_readpaths_subpaths()

        if num_processes is None:  # use the user input value if provided
            num_processes = self.num_processes
        assert num_processes >= 1
        if self.__use_alignment_cov:
            logger.debug("estimating contig coverages from read paths ..")
            self.estimate_contig_coverages_from_read_paths()
            # TODO: how to remove low coverage contigs
        else:
            self.use_contig_coverage_from_assembly_graph()
        self.estimate_single_copy_vertices()

        logger.info("Generating heuristic variants .. ")
        if num_processes == 1:
            self.__gen_heuristic_paths_uni()
        else:
            self.__gen_heuristic_paths_mp(num_proc=num_processes)

    def load_temp(self):
        len_vars = len(list(self.temp_dir.glob("variant.*.tuple")))
        if len_vars:
            logger.info("Loading generated variants ..")
        for var_id in range(1, len_vars + 1):
            # although using pickle will be faster, txt is human-readable
            tuple_f = self.temp_dir.joinpath(f"variant.{var_id}.tuple")
            count_f = self.temp_dir.joinpath(f"variant.{var_id}.count")
            try:
                with open(tuple_f) as input_r, open(count_f) as input_i:
                    this_variant = eval(input_r.read())
                    this_count = int(input_i.read())
                    self.variants.append(this_variant)
                    self.variants_counts[this_variant] = this_count
            except FileNotFoundError as e:
                logger.error(str(e))
                logger.error("'--previous resume' is not usable when the generated files are illegal!")
                sys.exit(0)

    def __access_read_path_coverage(self,
                                    growing_variants,
                                    previous_len_variant,
                                    num_valid_search,
                                    path_not_traversed,
                                    previous_un_traversed_ratio,
                                    previous_un_traversed_ratio_count,
                                    reset_num_valid_search=True,
                                    report_detailed_warning=True):
        """
        Return: (expected_num_searches_to_add, un_traversed_path_ratio, counts_of_ratio_unchanged)
        """
        logger.info("Assessing read path coverage ..")
        logger.debug("  paths not traversed: " + str(path_not_traversed))
        logger.debug("  paths not traversed: " + str(bool(path_not_traversed)))
        """The minimum requirement is that all observed read_paths were covered"""
        if not path_not_traversed:
            return 0, 0, None
        else:
            # must use dict.keys() to iterate a manager-dict in a child process
            # logger.debug(str([str(_rp) + ";  " for _rp in path_not_traversed.keys()]))
            for variant_path in growing_variants[previous_len_variant:]:
                logger.debug("check variant_path " + str(variant_path))
                # TODO if memory can be shared in a pool without serialization, this can be much faster
                # TODO, separate this part into each worker, it will be much faster
                for sub_path in self.subpath_generator.gen_subpaths(variant_path):
                    # logger.debug("check subpath", sub_path)
                    if sub_path in path_not_traversed:
                        del path_not_traversed[sub_path]
                # the current get_variant_sub_paths function only consider subpaths with length > 1
                # TODO: get subpath adaptive to length=1, more general and less restrictions
                # TODO: important!!!!
                # after which the following block can be removed
                for single_v, single_e in variant_path:
                    single_sbp = ((single_v, False),)
                    if single_sbp in path_not_traversed:
                        del path_not_traversed[single_sbp]
                if not path_not_traversed:
                    return 0, 0, None
            if not path_not_traversed:
                return 0, 0, None
            # the distribution of sampled subpaths are complex due to graph-based heuristic extension
            # here we just approximate it as Poisson distribution
            # similar to the lander-waterman model in genome sequencing,
            # our hypothesized coverage (a=N*factor, where N is num_valid_search) and fraction in gaps (p) is
            #     a==-log(p)
            # we want the fraction in gaps to be smaller than 1/len(self.read_paths), e.g. 0.5/len(self.read_paths)
            current_ratio = len(path_not_traversed) / len(self.read_paths)
            logger.info("uncovered_paths/all_paths = %i/%i = %.4f" %
                        (len(path_not_traversed), len(self.read_paths), current_ratio))
            if not reset_num_valid_search:
                logger.warning("{} read paths not traversed".format(len(path_not_traversed)))
                if report_detailed_warning:
                    # must use dict.keys() to iterate a manager-dict in a child process
                    for go_p, p_n_t in enumerate(sorted(list(path_not_traversed.keys()))):
                        logger.warning("  read path %i (len=%i, reads=%i): %s" %
                                       (go_p, len(p_n_t), self.__read_paths_counter[p_n_t], p_n_t))
                logger.warning("This may due to 1) insufficient num of valid variants (-N), or "
                               "2) unrealistic constraints on the variant topology, or 3) chimeric reads.")
                return
            if current_ratio == previous_un_traversed_ratio:
                # if the same un_traversed ratio occurs more than 2 times, stop searching for variants
                if previous_un_traversed_ratio_count > 2:
                    logger.warning("{} read paths not traversed".format(len(path_not_traversed)))
                    if report_detailed_warning:
                        # must use dict.keys() to iterate a manager-dict in a child process
                        for go_p, p_n_t in enumerate(sorted(list(path_not_traversed.keys()))):
                            logger.warning("  read path %i (len=%i, reads=%i): %s" %
                                           (go_p, len(p_n_t), self.__read_paths_counter[p_n_t], p_n_t))
                    logger.warning("This may due to 1) insufficient num of valid variants (-N), or "
                                   "2) unrealistic constraints on the variant topology, or 3) chimeric reads.")
                    return 0, current_ratio, None
                else:
                    new_num_valid_search = num_valid_search * log(0.5 / len(self.read_paths)) / log(current_ratio)
                    new_num_valid_search = int(new_num_valid_search)
                    logger.info("resetting min_valid_search={}".format(new_num_valid_search))
                    return new_num_valid_search - num_valid_search, current_ratio, previous_un_traversed_ratio_count + 1
            else:
                new_num_valid_search = num_valid_search * log(0.5 / len(self.read_paths)) / log(current_ratio)
                new_num_valid_search = int(new_num_valid_search)
                logger.info("resetting min_valid_search={}".format(new_num_valid_search))
                return new_num_valid_search - num_valid_search, current_ratio, 1
                # TODO this process will never stop if the graph cannot generate a circular path on forced circular

    def index_readpaths_subpaths(self):
        logger.info("Indexing aligned read path records ..")
        # # alignment_lengths = []
        # if filter_by_graph:
        #     for gaf_record in self.alignment.raw_records:
        #         this_read_path = tuple(self.graph.get_standardized_path(gaf_record.path))
        #         # summarize only when the graph contain the path
        #         if self.graph.contain_path(this_read_path):
        #             if this_read_path in self.__read_paths_counter:
        #                 self.__read_paths_counter[this_read_path] += 1
        #             else:
        #                 self.__read_paths_counter[this_read_path] = 1
        #                 self.read_paths.append(this_read_path)
        #             # # record alignment length
        #             # alignment_lengths.append(gaf_record.p_align_len)
        # else:
        #     for gaf_record in self.alignment.raw_records:
        #         this_read_path = tuple(self.graph.get_standardized_path(gaf_record.path))
        #         if this_read_path in self.__read_paths_counter:
        #             self.__read_paths_counter[this_read_path] += 1
        #         else:
        #             self.__read_paths_counter[this_read_path] = 1
        #             self.read_paths.append(this_read_path)
        #         # # record alignment length
        #         # alignment_lengths.append(gaf_record.p_align_len)
        for read_id, this_read_path in enumerate(self.read_paths):
            read_contig_num = len(this_read_path)
            forward_read_path_tuple = tuple(this_read_path)
            reverse_read_path_tuple = tuple(self.graph.reverse_path(this_read_path))
            for sub_contig_num in range(1, read_contig_num):
                # index the starting subpaths
                self.__index_start_subpath(forward_read_path_tuple[:sub_contig_num], read_id, True)
                # reverse
                self.__index_start_subpath(reverse_read_path_tuple[: sub_contig_num], read_id, False)
                # index the middle subpaths
                # excluding the start and the end subpaths: range(0 + 1, read_contig_num - sub_contig_num + 1 - 1)
                for go_sub in range(1, read_contig_num - sub_contig_num):
                    # forward
                    self.__index_middle_subpath(
                        forward_read_path_tuple[go_sub: go_sub + sub_contig_num], read_id, True)
                    # reverse
                    self.__index_middle_subpath(
                        reverse_read_path_tuple[go_sub: go_sub + sub_contig_num], read_id, False)
        #
        # self.max_alignment_len = sorted(alignment_lengths)[-1]
        self.__read_paths_not_in_variants = {_rp: None for _rp in self.__read_paths_counter}
        self.__read_paths_counter_indexed = True

    def estimate_contig_coverages_from_read_paths(self):
        """
        Counting the contig coverage using the occurrences in the read paths.
        Note: this will proportionally overestimate the coverage values comparing to base coverage values,
        """
        self.contig_coverages = OrderedDict([(v_name, 0) for v_name in self.graph.vertex_info])
        for read_path in self.read_paths:
            for v_name, v_end in read_path:
                if v_name in self.contig_coverages:
                    self.contig_coverages[v_name] += 1

    def use_contig_coverage_from_assembly_graph(self):
        self.contig_coverages = \
            OrderedDict([(v_name, self.graph.vertex_info[v_name].cov) for v_name in self.graph.vertex_info])
        logger.trace(str(self.contig_coverages))

    def estimate_single_copy_vertices(self):
        """
        Currently only use the connection information

        TODO: use estimate variant separation and multiplicity estimation to better estimate this
        """
        for go_v, v_name in enumerate(self.graph.vertex_info):
            if len(self.graph.vertex_info[v_name].connections[True]) < 2 and \
                    len(self.graph.vertex_info[v_name].connections[False]) < 2:
                self.__candidate_single_copy_vs.add(v_name)

    #     np.random.seed(self.__random.randint(1, 10000))
    #     clusters_res = WeightedGMMWithEM(
    #         data_array=list(self.contig_coverages.values()),
    #         data_weights=[self.graph.vertex_info[v_name].len for v_name in self.graph.vertex_info]).run()
    #     mu_list = [params["mu"] for params in clusters_res["parameters"]]
    #     smallest_label = mu_list.index(min(mu_list))
    #     # self.contig_coverages[v_name],
    #     # loc = current_names[v_name] * old_cov_mean,
    #     # scale = old_cov_std
    #     if len(mu_list) == 1:
    #         for go_v, v_name in enumerate(self.graph.vertex_info):
    #             # looking for smallest vertices
    #             if clusters_res["labels"][go_v] == smallest_label:
    #                 if len(self.graph.vertex_info[v_name].connections[True]) < 2 and \
    #                         len(self.graph.vertex_info[v_name].connections[False]) < 2:
    #                     self.single_copy_vertices_prob[v_name] #
    #     else:

    def __gen_heuristic_paths_uni(self):
        """
        single-process version of generating heuristic paths
        """
        # logger.info("Start generating candidate paths ..")
        # v_len = len(self.graph.vertex_info)
        # while self.count_valid < self.num_valid_search:
        previous_ratio = 1.
        previous_ratio_c = 1
        num_valid_search = self.min_valid_search
        variant_ids = {}
        for go_v, variant in enumerate(self.variants):   # variant id is 1-based for easier manual inspection
            variant_ids[variant] = go_v + 1
        self.count_valid = sum(self.variants_counts.values())
        if self.count_valid >= num_valid_search:
            add_search, previous_ratio, previous_ratio_c = self.__access_read_path_coverage(
                growing_variants=self.variants,
                previous_len_variant=self.__previous_len_variant,
                num_valid_search=num_valid_search,
                path_not_traversed=self.__read_paths_not_in_variants,
                previous_un_traversed_ratio=previous_ratio,
                previous_un_traversed_ratio_count=previous_ratio_c)
            if add_search:
                self.__previous_len_variant = len(self.variants)
                num_valid_search += add_search
            else:
                logger.info("\t{}/{}/{}/{} uniq/valid/tvs/set variants".format(
                    len(self.variants), self.count_valid, "-", self.min_valid_search))
                # logger.info("  {} unique paths in {}/{} valid paths, {} traversals".format(
                #     len(self.variants), self.count_valid, self.min_valid_search, "-"))
                logger.info("Sufficient previous valid paths loaded.")
                return
        elif self.count_valid:
            logger.info("\t{}/{}/{}/{} uniq/valid/tvs/set variants".format(
                len(self.variants), self.count_valid, "-", self.min_valid_search))
            # logger.info("  {} unique paths in {}/{} valid paths, {} traversals".format(
            #     len(self.variants), self.count_valid, self.min_valid_search, "-"))
        do_traverse = True
        count_debug = 0
        while do_traverse:
            single_traversal = SingleTraversal(self, self.__random.randint(1, 1e5))
            single_traversal.run()
            new_path = single_traversal.result_path
            self.count_search += 1
            logger.debug("    traversal {}: {}".format(self.count_search, self.graph.repr_path(new_path)))
            # logger.trace("  {} unique paths in {}/{} valid paths, {} traversals".format(
            #     len(self.variants), count_valid, num_valid_search, count_search))
            is_circular_p = self.graph.is_circular_path(new_path)
            # for debug
            # logger.trace(str(self.force_circular))
            # logger.trace(str(is_circular_p))
            # logger.trace(str(self.uni_chromosome))
            # logger.trace(str(self.graph.is_fully_covered_by(new_path)))
            invalid_search = (self.force_circular and not is_circular_p) or \
                             (self.uni_chromosome and not self.graph.is_fully_covered_by(new_path))

            # forcing the searching to be running until a circular result was found, was tested to be a bad idea
            # switch back to the post searching judge
            if invalid_search:
                logger.debug("    traversal {} is invalid".format(self.count_search))
                count_debug += 1
                # if count_debug > 0:
                #     raise Exception
                continue
            else:
                # if len(new_path) >= v_len * 2:  # using path length to guess multiple units is not a good idea
                if is_circular_p:
                    new_path_list = self.__decompose_hetero_units(new_path)
                else:
                    new_path_list = [new_path]
                for new_path in new_path_list:
                    self.count_valid += 1
                    if new_path in self.variants_counts:
                        self.variants_counts[new_path] += 1
                        var_id = variant_ids[new_path]
                        self.__save_tmp_counts(var_id, self.variants_counts[new_path])
                        logger.debug("\t, {}/{}/{}/{} uniq/valid/tvs/set variants".format(
                            len(self.variants), self.count_valid, self.count_search, num_valid_search))
                        # logger.debug("  {} unique paths in {}/{} valid paths, {} traversals".format(
                        #     len(self.variants), self.count_valid, num_valid_search, self.count_search))
                    else:
                        self.variants_counts[new_path] = 1
                        self.variants.append(new_path)
                        # variant id is 1-based for easier manual inspection
                        var_id = variant_ids[new_path] = len(self.variants)
                        self.__save_tmp_counts(var_id, 1)
                        self.__save_tmp_path(var_id, new_path)
                        logger.info("\t{}/{}/{}/{} uniq/valid/tvs/set variants".format(
                            len(self.variants), self.count_valid, self.count_search, num_valid_search))
                        # logger.info("  {} unique paths in {}/{} valid paths, {} traversals".format(
                        #     len(self.variants), self.count_valid, num_valid_search, self.count_search))

                    # hard bound
                    if self.count_valid >= self.max_valid_search:
                        self.__access_read_path_coverage(
                            growing_variants=self.variants,
                            previous_len_variant=self.__previous_len_variant,
                            num_valid_search=num_valid_search,
                            path_not_traversed=self.__read_paths_not_in_variants,
                            previous_un_traversed_ratio=previous_ratio,
                            previous_un_traversed_ratio_count=previous_ratio_c,
                            reset_num_valid_search=False)
                        do_traverse = False
                        break

                    if self.count_valid >= num_valid_search:
                        add_search, previous_ratio, previous_ratio_c = self.__access_read_path_coverage(
                            growing_variants=self.variants,
                            previous_len_variant=self.__previous_len_variant,
                            num_valid_search=num_valid_search,
                            path_not_traversed=self.__read_paths_not_in_variants,
                            previous_un_traversed_ratio=previous_ratio,
                            previous_un_traversed_ratio_count=previous_ratio_c)
                        if add_search:
                            self.__previous_len_variant = len(self.variants)
                            num_valid_search += add_search
                        else:
                            do_traverse = False
                            break
            # if break_traverse:
            #     break
        logger.info("\t{}/{}/{}/{} uniq/valid/tvs/set variants".format(
            len(self.variants), self.count_valid, self.count_search, num_valid_search))
        # logger.info("  {} unique paths in {}/{} valid paths, {} traversals".format(
        #     len(self.variants), self.count_valid, num_valid_search, self.count_search))

    def __save_tmp_counts(self, var_id, counts):
        if self.temp_dir.exists():
            count_f_tmp = self.temp_dir.joinpath(f"variant.{var_id}.count.TMP")
            count_f = self.temp_dir.joinpath(f"variant.{var_id}.count")
            with open(count_f_tmp, "w") as output_i:
                output_i.write(str(counts))
            os.rename(count_f_tmp, count_f)

    def __save_tmp_path(self, var_id, new_path):
        if self.temp_dir.exists():
            tuple_f_tmp = self.temp_dir.joinpath(f"variant.{var_id}.tuple.TMP")
            tuple_f = self.temp_dir.joinpath(f"variant.{var_id}.tuple")
            with open(tuple_f_tmp, "w") as output_t:
                output_t.write(str(new_path))
            os.rename(tuple_f_tmp, tuple_f)

    def __heuristic_traversal_worker(
            self, variants, variant_ids, variants_counts, path_not_traversed, g_vars, lock, event, err_queue):
        """
        single worker of traversal, called by self.get_heuristic_paths_multiprocessing
        starting a new process from dill dumped python object: slow
        """
        try:
            # break_traverse = False
            while g_vars.count_valid < g_vars.num_valid_search:
                # move the parallelizable code block before the lock
                # <<<
                # TODO to cover all subpaths
                #      setting start path for traversal, simultaneously solve the random problem
                single_traversal = SingleTraversal(self, self.__random.randint(1, 1e5))
                single_traversal.run()
                new_path = single_traversal.result_path
                repr_path = self.graph.repr_path(new_path)
                is_circular_p = self.graph.is_circular_path(new_path)
                invalid_search = (self.force_circular and not is_circular_p) or \
                                 (self.uni_chromosome and not self.graph.is_fully_covered_by(new_path))
                if not invalid_search:
                    # if len(new_path) >= v_len * 2:  # using path length to guess multiple units is not a good idea
                    if is_circular_p:
                        new_path_list = self.__decompose_hetero_units(new_path)
                    else:
                        new_path_list = [new_path]
                else:
                    new_path_list = []
                # >>>
                # locking the counts and variants
                lock.acquire()
                g_vars.count_search += 1
                logger.trace("    traversal {}: {}".format(g_vars.count_search, repr_path))
                if invalid_search:
                    lock.release()
                    continue
                else:
                    for new_path in new_path_list:
                        g_vars.count_valid += 1
                        if new_path in variants_counts:
                            variants_counts[new_path] += 1
                            var_id = variant_ids[new_path]
                            self.__save_tmp_counts(var_id, variants_counts[new_path])
                            logger.trace("\t{}/{}/{}/{} uniq/valid/tvs/set variants".format(
                                len(variants), g_vars.count_valid, g_vars.count_search, g_vars.num_valid_search))
                            # logger.trace("  {} unique paths in {}/{} valid paths, {} traversals".format(
                            #     len(variants), g_vars.count_valid, g_vars.num_valid_search, g_vars.count_search))
                        else:
                            variants_counts[new_path] = 1
                            variants.append(new_path)
                            var_id = variant_ids[new_path] = len(variants)
                            self.__save_tmp_counts(var_id, 1)
                            self.__save_tmp_path(var_id, new_path)
                            logger.info("\t{}/{}/{}/{} uniq/valid/tvs/set variants".format(
                                len(variants), g_vars.count_valid, g_vars.count_search, g_vars.num_valid_search))
                            # logger.info("  {} unique paths in {}/{} valid paths, {} traversals".format(
                            #     len(variants), g_vars.count_valid, g_vars.num_valid_search, g_vars.count_search))

                        if g_vars.count_valid >= g_vars.max_valid_search:
                            # break_traverse = True
                            # kill all other workers
                            g_vars.run_status = "reached"
                            event.set()
                            return
                            # return "reached"
                            # raise MaxTraversalReached("")

                        if g_vars.count_valid >= g_vars.num_valid_search:
                            # logger.info("max valid search: " + str(g_vars.max_valid_search))
                            # logger.info("num valid search: " + str(g_vars.num_valid_search))
                            add_search, g_vars.previous_ratio, g_vars.previous_ratio_c = \
                                self.__access_read_path_coverage(
                                    growing_variants=variants,
                                    previous_len_variant=g_vars.previous_len_variant,
                                    num_valid_search=g_vars.num_valid_search,
                                    path_not_traversed=path_not_traversed,
                                    previous_un_traversed_ratio=g_vars.previous_ratio,
                                    previous_un_traversed_ratio_count=g_vars.previous_ratio_c)
                            logger.info("adding searches by " + str(add_search))
                            if add_search:
                                g_vars.previous_len_variant = len(variants)
                                g_vars.num_valid_search += add_search
                            else:
                                # break_traverse = True
                                # kill all other workers
                                event.set()
                                return "done"
                    lock.release()
                # if break_traverse:
                #     break
        except KeyboardInterrupt:
            g_vars.run_status = "interrupt"
            event.set()
            return
            # return "keyboard"
            # raise KeyboardInterrupt
        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            tb = traceback.format_exception(exc_type, exc_value, exc_traceback)
            location = traceback.extract_tb(exc_traceback)[-1]
            event.set()
            err_queue.put((e, tb, location))

    def __gen_heuristic_paths_mp(self, num_proc=2):
        """
        multiprocess version of generating heuristic paths
        starting a new process from dill dumped python object: slow
        """
        self.count_valid = sum(self.variants_counts.values())
        if self.count_valid >= self.min_valid_search:
            # TODO: previous_ratio, etc can be loaded using json
            add_search, previous_ratio, previous_ratio_c = \
                self.__access_read_path_coverage(
                    growing_variants=self.variants,
                    previous_len_variant=0,
                    num_valid_search=self.min_valid_search,
                    path_not_traversed=self.__read_paths_not_in_variants,
                    previous_un_traversed_ratio=1.,
                    previous_un_traversed_ratio_count=1)
            if not add_search:
                logger.info("\t{}/{}/{}/{} uniq/valid/tvs/set variants".format(
                    len(self.variants), self.count_valid, "-", self.min_valid_search))
                # logger.info("  {} unique paths in {}/{} valid paths, {} traversals".format(
                #     len(self.variants), self.count_valid, self.min_valid_search, "-"))
                logger.info("Sufficient previous valid paths loaded.")
                return
            else:
                self.min_valid_search += add_search
        else:
            previous_ratio, previous_ratio_c = 1., 1

        manager = Manager()
        variants_counts = manager.dict()
        variant_ids = manager.dict()
        variants = manager.list()
        path_not_traversed = manager.dict()
        global_vars = manager.Namespace()
        global_vars.run_status = ""
        global_vars.count_search = self.count_search
        global_vars.count_valid = self.count_valid
        global_vars.num_valid_search = self.min_valid_search
        global_vars.max_valid_search = self.max_valid_search
        global_vars.previous_len_variant = self.__previous_len_variant
        global_vars.previous_ratio = previous_ratio
        # initialize the shared variable for recording the ratio of un-traversed reads
        global_vars.previous_ratio_c = previous_ratio_c

        if self.variants_counts:
            variants_counts.update(self.variants_counts)
            variants.extend(self.variants)
            for go_v, variant in enumerate(self.variants):   # variant id is 1-based for easier manual inspection
                variant_ids[variant] = go_v + 1
            global_vars.previous_len_variant = self.__previous_len_variant = len(self.variants)
        for rp_not_in_v in self.__read_paths_not_in_variants:
            path_not_traversed[rp_not_in_v] = None

        lock = manager.Lock()
        event = manager.Event()
        error_queue = manager.Queue()
        # v_len = len(self.graph.vertex_info)
        pool_obj = Pool(processes=num_proc)  # the worker processes are daemonic
        # dump function and args
        logger.info("Serializing the heuristic searching for multiprocessing ..")
        payload = dill.dumps((self.__heuristic_traversal_worker,
                              (variants,
                               variant_ids,
                               variants_counts,
                               path_not_traversed,
                               global_vars,
                               lock,
                               event,
                               error_queue)))
        # logger.info("Start generating candidate paths ..")
        try:
            jobs = []
            for g_p in range(num_proc):
                logger.debug("assigning job to worker {}".format(g_p + 1))
                jobs.append(pool_obj.apply_async(run_dill_encoded, (payload,)))
                logger.debug("assigned job to worker {}".format(g_p + 1))
                if global_vars.count_valid >= global_vars.num_valid_search:
                    lock.acquire()
                    add_search, global_vars.previous_ratio, global_vars.previous_ratio_c = \
                        self.__access_read_path_coverage(
                            growing_variants=variants,
                            previous_len_variant=global_vars.previous_len_variant,
                            num_valid_search=global_vars.num_valid_search,
                            path_not_traversed=path_not_traversed,
                            previous_un_traversed_ratio=global_vars.previous_ratio,
                            previous_un_traversed_ratio_count=global_vars.previous_ratio_c)
                    if not add_search:
                        lock.release()
                        break
                    else:
                        global_vars.num_valid_search += add_search
                        lock.release()
            # for job in jobs:
            #     job.get()  # tracking errors
                # if "reached" in job_msg:
                #     raise MaxTraversalReached("")
                # elif "keyboard" in job_msg:
                #     raise KeyboardInterrupt
            pool_obj.close()
        except KeyboardInterrupt:
            # event.set()
            # pool_obj.terminate()
            # pool_obj.close()
            logger.info("<Keyboard interrupt>")
            self.__access_read_path_coverage(
                growing_variants=variants,
                previous_len_variant=global_vars.previous_len_variant,
                num_valid_search=global_vars.num_valid_search,
                path_not_traversed=path_not_traversed,
                previous_un_traversed_ratio=global_vars.previous_ratio,
                previous_un_traversed_ratio_count=global_vars.previous_ratio_c,
                reset_num_valid_search=False)
        else:
            event.wait()
            pool_obj.terminate()
            while not error_queue.empty():
                e, tb, location = error_queue.get()
                logger.error("\n" + "".join(tb))
                sys.exit(0)
                # raise error_queue.get()
            if global_vars.run_status in {"reached", "interrupt"}:
                # except MaxTraversalReached:
                # pool_obj.terminate()
                # pool_obj.join()  # maybe no need to join
                if global_vars.run_status == "reached":
                    logger.info("maximum num of valid searches reached.")
                elif global_vars.run_status == "interrupt":
                    logger.info("<Keyboard interrupt>")
                self.__access_read_path_coverage(
                    growing_variants=variants,
                    previous_len_variant=global_vars.previous_len_variant,
                    num_valid_search=global_vars.num_valid_search,
                    path_not_traversed=path_not_traversed,
                    previous_un_traversed_ratio=global_vars.previous_ratio,
                    previous_un_traversed_ratio_count=global_vars.previous_ratio_c,
                    reset_num_valid_search=False)
            # else:
            # logger.info("waiting ..")
            # pool_obj.join()  # maybe no need to join

        self.variants_counts = dict(variants_counts)
        self.variants = list(variants)
        self.count_valid += global_vars.count_valid
        self.count_search += global_vars.count_search

        logger.info("\t{}/{}/{}/{} uniq/valid/tvs/set variants".format(
            len(self.variants), global_vars.count_valid, global_vars.count_search, global_vars.num_valid_search))
        # logger.info("  {} unique paths in {}/{} valid paths, {} traversals".format(
        #     len(self.variants), global_vars.count_valid, global_vars.num_valid_search, global_vars.count_search))

    def __decompose_hetero_units(self, circular_path):
        """
        Decompose a path that may be composed of multiple circular paths (units) containing similar variants
        e.g. 1,2,3,4,5,1,-3,-2,4,5 was composed of 1,2,3,4,5 and 1,-3,-2,4,5,
             when all contigs were likely to be single copy
        e.g. 1,2,3,2,3,7,5,1,-3,-2,-3,-2,8,5 was composed of 1,2,3,2,3,7,5 and 1,-3,-2,-3,-2,8,5,
             when 1,5 were likely to be single copy
        """
        len_total = len(circular_path)
        if len_total < 4:
            return [circular_path]

        # 1.1 get the multiplicities (copy) information of (v_name, v_end) in the circular path
        logger.trace("circular_path: {}".format(circular_path))
        unique_vne_list = sorted(set(circular_path))
        copy_to_vne = OrderedDict()
        vne_to_copy = OrderedDict()
        for v_n_e in unique_vne_list:
            this_copy = circular_path.count(v_n_e)
            if this_copy not in copy_to_vne:
                copy_to_vne[this_copy] = []
            copy_to_vne[this_copy].append(v_n_e)
            vne_to_copy[v_n_e] = this_copy
        # 1.2 store v lengths
        v_lengths = OrderedDict([(v_n_, self.graph.vertex_info[v_n_].len) for v_n_, v_e_ in unique_vne_list])

        # 2. estimate candidate number of units.
        # The shared contig-len-weighted path should be larger than self.__min_unit_similarity
        candidate_sc_vertices = set([_v_n for _v_n, _v_e in vne_to_copy]) & self.__candidate_single_copy_vs
        logger.trace("      candidate_sc_vertices: {}".format(candidate_sc_vertices))
        copies = sorted(copy_to_vne)
        if candidate_sc_vertices:
            # limit the estimation to the candidate single copy vertices
            sum_lens = [sum([v_lengths[_v_n]
                             for _v_n, _v_e in copy_to_vne[copy_num] if _v_n in self.__candidate_single_copy_vs])
                        for copy_num in copies]
        else:
            # no candidate single copy vertices were present in the path, use all vertices (contigs)
            sum_lens = [sum([v_lengths[_v_n]
                             for _v_n, _v_e in copy_to_vne[copy_num]])
                        for copy_num in copies]
        weights = [_c * _l for _c, _l in zip(copies, sum_lens)]  # contig-len-weights
        sum_w = float(sum(weights))  # total weights
        weights = [_w / sum_w for _w in weights]   # percent of each copy-num set of contigs
        count_weights = len(weights)
        candidate_num_units = []
        for go_c, copy_num in enumerate(copies):
            if copy_num > 1:
                accumulated_weight = 0
                for go_w in range(go_c, count_weights):
                    # e.g. a fourfold contig may potentially contribute to the unit_similarity if the num of units is 2.
                    if copies[go_w] % copy_num == 0:
                        accumulated_weight += weights[go_w]
                if accumulated_weight >= self.__min_unit_similarity:
                    # append all candidate copy numbers here, no need to do prime factor
                    candidate_num_units.append(copy_num)

        # 3. try to decompose
        logger.trace("      candidate_num_units: {}".format(candidate_num_units))
        if not candidate_num_units:  # or candidate_num_units == [1]:
            return [circular_path]
        else:
            v_lengths = np.array([self.graph.vertex_info[v_n_].len for v_n_, v_e_ in unique_vne_list])
            v_copies = np.array(list(vne_to_copy.values()))
            # not very important TODO account for uni_overlap; mutable overlaps; uni_overlap = self.graph.uni_overlap()
            total_base_len = float(sum(v_lengths * v_copies))  # ignore overlap effect
            qualified_schemes = set()
            for num_units in candidate_num_units:
                logger.trace("      num_units: {}".format(num_units))
                unit_sc_vertices = [v_n_e for v_n_e in copy_to_vne[num_units] if v_n_e[0] in candidate_sc_vertices]
                logger.trace("      unit_sc_vertices: {}".format(unit_sc_vertices))
                candidate_starts = unit_sc_vertices if unit_sc_vertices else copy_to_vne[num_units]
                logger.trace("      candidate_starts: {}".format(candidate_starts))
                # 3.1. only keep the first start_n_e for a series of consecutive start_n_e
                #      because they generated the same circular units
                sne_indices = {s_n_e: [] for s_n_e in candidate_starts}
                for s_id, s_ne in enumerate(circular_path):
                    if s_ne in sne_indices:
                        sne_indices[s_ne].append(s_id)
                sne_indices = [[s_n_e] + sne_indices[s_n_e] for s_n_e in candidate_starts]
                # for start_n_e in candidate_starts:
                #     sne_indices.append([start_n_e])
                #     sne_indices[-1].extend([_id for _id, _ne in enumerate(circular_path) if _ne == start_n_e])
                sne_indices.sort(key=lambda x: x[1:])  # sort by the indices
                # 3.1.1 the start
                for first_id, end_id in zip(sne_indices[0][2:] + sne_indices[0][1:2], sne_indices[-1][1:]):
                    if (end_id + 1) % len_total != first_id:
                        consecutive_ends = False
                        break
                else:
                    consecutive_ends = True
                # 3.1.2
                go_keep_start = 0
                go_check_start = 1
                step = 1
                while go_check_start < len(sne_indices):
                    for k_id, c_id in zip(sne_indices[go_keep_start][1:], sne_indices[go_check_start][1:]):
                        if (k_id + step) % len_total != c_id:
                            go_keep_start = go_check_start
                            go_check_start += 1
                            step = 1
                            break
                    else:
                        del sne_indices[go_check_start]
                        step += 1
                if consecutive_ends and len(sne_indices) > 1:
                    # sne_indices[0] and sne_indices[-1] will decompose the circular_path into the same units
                    del sne_indices[0]
                logger.trace("      sne_indices: {}".format(sne_indices))

                # 3.2 try to decompose and calculate the shared variants
                #     to determine whether those starts are qualified
                #     to break the original path into units
                for start_n_e, *s_indices in sne_indices:
                    units = []
                    for from_id, to_id in zip(s_indices[:-1], s_indices[1:]):
                        units.append(circular_path[from_id:to_id])
                    units.append(circular_path[s_indices[-1]:] + circular_path[:s_indices[0]])
                    variant_counts = np.array([[_unit.count(v_n_e_)
                                                for v_n_e_ in unique_vne_list]
                                                for _unit in units])
                    # idx_shared = (variant_counts == variant_counts[0]).all(axis=0)
                    variants_shared = variant_counts.min(axis=0)
                    # logger.info("idx_shared ({}): {}".format(len(idx_shared), idx_shared))
                    # logger.info("variant_counts[0] ({}): {}".format(len(variant_counts[0]), variant_counts[0]))
                    # logger.info("v_lengths ({}): {}".format(len(v_lengths), v_lengths))
                    shared_len = \
                        num_units * sum(variants_shared * v_lengths) / total_base_len
                    logger.trace("      shared_len: {}".format(shared_len))
                    if shared_len > self.__min_unit_similarity:
                        this_scheme = tuple(sorted([self.graph.get_standardized_path_circ(self.graph.roll_path(_unit))
                                                    for _unit in units]))
                        if this_scheme not in qualified_schemes:
                            qualified_schemes.add(this_scheme)
                            logger.trace("new scheme added: {}".format(this_scheme))
            logger.trace("      qualified_schemes ({}): {}".format(len(qualified_schemes), qualified_schemes))

            # 3.3 calculate the support from read paths
            circular_units = []
            original_sub_paths = set(self.subpath_generator.gen_subpaths(circular_path))
            for this_scheme in qualified_schemes:
                these_sub_paths = set()
                for this_unit in this_scheme:
                    these_sub_paths |= set(self.subpath_generator.gen_subpaths(this_unit))
                if original_sub_paths - these_sub_paths:
                    # the original one contains unique subpath(s)
                    continue
                else:
                    for this_unit in this_scheme:
                        circular_units.append(this_unit)
                    # calculate the multiplicity-based likelihood will be weird,
                    # because if we believe the decomposed units are a reasonable scheme,
                    # the graph itself is a mixture of combination.
                    # Besides, the multiplicity-based likelihood would definitely prefer the decomposed ones,
                    # given that the candidate_num_units is generated from self.__candidate_single_copy_vs if available
            logger.trace("      circular_units ({}): {}".format(len(circular_units), circular_units))
            if not circular_units:
                return [circular_path]
            else:
                return circular_units

    # def __decompose_hetero_units_old(self, circular_path):
    #     """
    #     Decompose a path that may be composed of multiple circular paths (units), which shared the same variants
    #     e.g. 1,2,3,4,5,1,-3,-2,4,5 was composed of 1,2,3,4,5 and 1,-3,-2,4,5
    #     """
    #     def get_v_counts(_path): return [_path.count(_v_name) for _v_name in self.graph.vertex_info]
    #     v_list = [v_name for v_name, v_end in circular_path]
    #     v_counts = get_v_counts(v_list)
    #     gcd = find_greatest_common_divisor(v_counts)
    #     logger.trace("  checking gcd {} from {}".format(gcd, circular_path))
    #     if gcd == 1:
    #         # the greatest common divisor is 1
    #         return [circular_path]
    #     else:
    #         logger.debug("  decompose {}".format(circular_path))
    #         v_to_id = {v_name: go_id for go_id, v_name in enumerate(self.graph.vertex_info)}
    #         unit_counts = [int(v_count/gcd) for v_count in v_counts]
    #         unit_len = int(len(v_list) / gcd)
    #         reseed_at = self.__random.randint(0, unit_len - 1)
    #         v_list_shuffled = v_list[len(v_list) - reseed_at:] + v_list + v_list[:unit_len]
    #         counts_check = get_v_counts(v_list_shuffled[:unit_len])
    #         find_start = False
    #         try_start = 0
    #         for try_start in range(unit_len):
    #             # if each unit has the same composition
    #             if counts_check == unit_counts and \
    #                     set([get_v_counts(v_list_shuffled[try_start+unit_len*go_u:try_start + unit_len*(go_u + 1)])
    #                          == unit_counts
    #                          for go_u in range(1, gcd)]) \
    #                     == {True}:
    #                 find_start = True
    #                 break
    #             else:
    #                 counts_check[v_to_id[v_list_shuffled[try_start]]] -= 1
    #                 counts_check[v_to_id[v_list_shuffled[try_start + unit_len]]] += 1
    #         if find_start:
    #             path_shuffled = circular_path[len(v_list) - reseed_at:] + circular_path + circular_path[:unit_len]
    #             unit_seq_len = self.graph.get_path_length(path_shuffled[try_start: try_start + unit_len])
    #             unit_copy_num = min(max(int((self.max_alignment_len - 2) / unit_seq_len), 1), gcd)
    #             return_list = []
    #             for go_unit in range(int(gcd/unit_copy_num)):
    #                 go_from__ = try_start + unit_len * unit_copy_num * go_unit
    #                 go_to__ = try_start + unit_len * unit_copy_num * (go_unit + 1)
    #                 variant_path = path_shuffled[go_from__: go_to__]
    #                 if self.graph.is_circular_path(variant_path):
    #                     return_list.append(self.graph.get_standardized_path_circ(variant_path))
    #             return return_list
    #         else:
    #             return [circular_path]

    def __index_start_subpath(self, subpath, read_id, strand):
        """
        :param subpath: tuple
        :param read_id: int, read id in self.read_paths
        :param strand: bool
        :return:
        """
        if subpath in self.__start_subpath_to_readpaths:
            self.__start_subpath_to_readpaths[subpath].add((read_id, strand))
        else:
            self.__start_subpath_to_readpaths[subpath] = {(read_id, strand)}

    def __index_middle_subpath(self, subpath, read_id, strand):
        """
        :param subpath: tuple
        :param read_id: int, read id in self.read_paths
        :param strand: bool
        :param subpath_loc: int, the location of the subpath in a read
        :return:
        """
        if subpath in self.__middle_subpath_to_readpaths:
            self.__middle_subpath_to_readpaths[subpath].add((read_id, strand))
        else:
            self.__middle_subpath_to_readpaths[subpath] = {(read_id, strand)}

    def pass_starting_subpath_to_readpaths(self):
        return self.__start_subpath_to_readpaths

    def pass_middle_subpath_to_readpaths(self):
        return self.__middle_subpath_to_readpaths

    def pass_read_paths_counter(self):
        return self.__read_paths_counter

    def pass_candidate_single_copy_vs(self):
        return self.__candidate_single_copy_vs

    def pass_differ_f(self):
        return self.__differ_f

    def pass_cov_inert(self):
        return self.__cov_inert

    def pass_decay_f(self):
        return self.__decay_f

    def pass_decay_t(self):
        return self.__decay_t

