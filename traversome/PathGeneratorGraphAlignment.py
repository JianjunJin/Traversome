from loguru import logger
from traversome.utils import find_greatest_common_divisor, harmony_weights  # WeightedGMMWithEM
from copy import deepcopy
from scipy.stats import norm
# from math import log, exp
from collections import OrderedDict
import numpy as np
from numpy import log, exp



class PathGeneratorGraphAlignment(object):
    """
    generate heuristic components (isomers & sub-chromosomes) from alignments.
    Here the components are not necessarily identical in contig composition.
    TODO automatically estimate num_search using convergence-test like approach
    """

    def __init__(self,
                 assembly_graph,
                 graph_alignment,
                 random_obj,
                 num_search=1000,
                 differ_f=1.,
                 decay_f=20.,
                 cov_inert=1.,
                 use_alignment_cov=False):
        """
        :param assembly_graph:
        :param graph_alignment:
        :param random_obj: random
            passed from traversome.random [or from import random]
        :param num_search:
        :param differ_f: difference factor [0, INF)
            Weighted by which, reads with the same overlap with current path will be used according to their counts.
            new_weights = (count_weights^differ_f)/sum(count_weights^differ_f)
            Zero leads to that the read counts play no effect in read choice.
        :param decay_f: decay factor [0, INF]
            Chance reduces by which, a read with less overlap with current path will be used to extend current path.
            probs_(N-m) = probs_(N) * decay_f^(-m)
            A large value leads to strictly following read paths.
        :param cov_inert: coverage inertia [0, INF)
            The degree of tendency a path has to extend contigs with similar coverage.
            weight *= exp(-abs(log(extending_coverage/current_path_coverage)))^cov_inert
            Designed for the mixture of multiple-sourced genomes, e.g. plastome and mitogenome.
            Set to zero if the graph is a single-sourced graph.
        :param use_alignment_cov: use the coverage from assembly graph if False.
        """
        assert 0 <= differ_f
        assert 0 <= decay_f
        assert 0 <= cov_inert
        self.graph = assembly_graph
        self.alignment = graph_alignment
        self.num_search = num_search
        self.__differ_f = differ_f
        self.__decay_f = decay_f
        self.__cov_inert = cov_inert
        self.__random = random_obj
        self.__use_alignment_cov = use_alignment_cov

        # to be generated
        self.local_max_alignment_len = None
        self.read_paths = list()
        self.__read_paths_counter = dict()
        # self.__vertex_to_readpath = {vertex: set() for vertex in self.graph.vertex_info}
        self.__starting_subpath_to_readpaths = {}
        self.__middle_subpath_to_readpaths = {}
        self.__read_paths_counter_indexed = False
        self.contig_coverages = OrderedDict()
        # self.single_copy_vertices_prob = \
        #     OrderedDict([(_v, 1.) for _v in single_copy_vertices]) if single_copy_vertices \
        #         else OrderedDict()
        self.components = list()
        self.components_counts = dict()

    def generate_heuristic_components(self, force_circular=True):
        logger.info("generating heuristic components .. ")
        if not self.__read_paths_counter_indexed:
            self.index_readpaths_subpaths()
        if self.__use_alignment_cov:
            logger.debug("estimating contig coverages from read paths ..")
            self.estimate_contig_coverages_from_read_paths()
        else:
            self.use_contig_coverage_from_assembly_graph()
        # self.estimate_single_copy_vertices()
        logger.debug("start traversing ..")
        self.get_heuristic_paths(force_circular=force_circular)

    # def generate_heuristic_circular_isomers(self):
    #     # based on alignments
    #     logger.warning("This function is under testing .. ")
    #     if not self.__read_paths_counter_indexed:
    #         self.index_readpaths_subpaths()
    #
    ## different from PathGeneratorGraphOnly.get_all_circular_isomers()
    ## this seaching is within the scope of long reads-supported paths
    # def generate_all_circular_isomers(self):
    #     # based on alignments

    def index_readpaths_subpaths(self, filter_by_graph=True):
        self.__read_paths_counter = dict()
        alignment_lengths = []
        if filter_by_graph:
            for gaf_record in self.alignment:
                this_read_path = tuple(self.graph.get_standardized_path(gaf_record.path))
                # summarize only when the graph contain the path
                if self.graph.contain_path(this_read_path):
                    if this_read_path in self.__read_paths_counter:
                        self.__read_paths_counter[this_read_path] += 1
                    else:
                        self.__read_paths_counter[this_read_path] = 1
                        self.read_paths.append(this_read_path)
                    # record alignment length
                    alignment_lengths.append(gaf_record.p_align_len)
        else:
            for gaf_record in self.alignment:
                this_read_path = tuple(self.graph.get_standardized_path(gaf_record.path))
                if this_read_path in self.__read_paths_counter:
                    self.__read_paths_counter[this_read_path] += 1
                else:
                    self.__read_paths_counter[this_read_path] = 1
                    self.read_paths.append(this_read_path)
                # record alignment length
                alignment_lengths.append(gaf_record.p_align_len)
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
        self.local_max_alignment_len = sorted(alignment_lengths)[-1]
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

    # def estimate_single_copy_vertices(self):
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

    def get_heuristic_paths(self, force_circular):
        count_search = 0
        count_valid = 0
        v_len = len(self.graph.vertex_info)
        while count_valid < self.num_search:
            count_search += 1
            new_path = self.graph.get_standardized_circular_path(self.graph.roll_path(self.__heuristic_extend_path([])))
            logger.trace("    traversal {}: {}".format(count_search, self.graph.repr_path(new_path)))
            # logger.trace("  {} unique paths in {}/{} legal paths, {} traversals".format(
            #     len(self.components), count_valid, self.num_search, count_search))
            if force_circular and not self.graph.is_circular_path(new_path):
                continue
            count_search -= 1
            if len(new_path) >= v_len * 2:
                new_path_list = self.__decompose_hetero_units(new_path)
            else:
                new_path_list = [new_path]
            for new_path in new_path_list:
                count_search += 1
                count_valid += 1
                if new_path in self.components_counts:
                    self.components_counts[new_path] += 1
                    logger.trace("  {} unique paths in {}/{} legal paths, {} traversals".format(
                        len(self.components), count_valid, self.num_search, count_search))
                else:
                    self.components_counts[new_path] = 1
                    self.components.append(new_path)
                    logger.info("  {} unique paths in {}/{} legal paths, {} traversals".format(
                        len(self.components), count_valid, self.num_search, count_search))
        logger.info("  {} unique paths in {}/{} legal paths, {} traversals".format(
            len(self.components), count_valid, self.num_search, count_search))

    def __decompose_hetero_units(self, circular_path):
        """

        """
        def get_v_counts(_path): return [_path.count(_v_name) for _v_name in self.graph.vertex_info]
        v_list = [v_name for v_name, v_end in circular_path]
        v_counts = get_v_counts(v_list)
        gcd = find_greatest_common_divisor(v_counts)
        logger.trace("  checking gcd {} from {}".format(gcd, circular_path))
        if gcd == 1:
            return [circular_path]
        else:
            logger.debug("  decompose {}".format(circular_path))
            v_to_id = {v_name: go_id for go_id, v_name in enumerate(self.graph.vertex_info)}
            unit_counts = [int(v_count/gcd) for v_count in v_counts]
            unit_len = int(len(v_list) / gcd)
            reseed_at = self.__random.randint(0, unit_len - 1)
            v_list_shuffled = v_list[len(v_list) - reseed_at:] + v_list + v_list[:unit_len]
            counts_check = get_v_counts(v_list_shuffled[:unit_len])
            find_start = False
            try_start = 0
            for try_start in range(unit_len):
                # if each unit has the same composition
                if counts_check == unit_counts and \
                        set([get_v_counts(v_list_shuffled[try_start+unit_len*go_u:try_start + unit_len*(go_u + 1)])
                             == unit_counts
                             for go_u in range(1, gcd)]) \
                        == {True}:
                    find_start = True
                    break
                else:
                    counts_check[v_to_id[v_list_shuffled[try_start]]] -= 1
                    counts_check[v_to_id[v_list_shuffled[try_start + unit_len]]] += 1
            if find_start:
                path_shuffled = circular_path[len(v_list) - reseed_at:] + circular_path + circular_path[:unit_len]
                unit_seq_len = self.graph.get_path_length(path_shuffled[try_start: try_start + unit_len])
                unit_copy_num = min(max(int((self.local_max_alignment_len - 2) / unit_seq_len), 1), gcd)
                return_list = []
                for go_unit in range(int(gcd/unit_copy_num)):
                    go_from__ = try_start + unit_len * unit_copy_num * go_unit
                    go_to__ = try_start + unit_len * unit_copy_num * (go_unit + 1)
                    this_path = path_shuffled[go_from__: go_to__]
                    if self.graph.is_circular_path(this_path):
                        return_list.append(self.graph.get_standardized_circular_path(this_path))
                return return_list
            else:
                return [circular_path]

    def __heuristic_extend_path(
            self, path, initial_mean=None, initial_std=None, not_do_reverse=False):
        """
        improvement needed
        :param path: empty path like [] or starting path like [("v1", True), ("v2", False)]
        :param initial_mean:
        :param initial_std:
        :param not_do_reverse: mainly for iteration, a mark to stop searching from the reverse end
        :return: a candidate component. e.g. [("v0", True), ("v1", True), ("v2", False), ("v3", True)]
        """
        if not path:
            # randomly choose the read path and the direction
            # change the weight (-~ depth) to flatten the search
            read_p_freq_reciprocal = [1./ self.__read_paths_counter[r_p] for r_p in self.read_paths]
            read_path = self.__random.choices(self.read_paths, weights=read_p_freq_reciprocal)[0]
            if self.__random.random() > 0.5:
                read_path = self.graph.reverse_path(read_path)
            path = list(read_path)
            initial_mean, initial_std = self.__get_cov_mean(read_path, return_std=True)
            return self.__heuristic_extend_path(
                path=path,
                not_do_reverse=False,
                initial_mean=initial_mean,
                initial_std=initial_std)
        else:
            # keep going in a circle util the path length reaches beyond the longest read alignment
            # stay within what data can tell
            repeating_unit = self.graph.roll_path(path)
            if len(path) > len(repeating_unit) and \
                    self.graph.get_path_internal_length(path) >= self.local_max_alignment_len:
                logger.trace("      traversal ended within a circle unit.")
                return deepcopy(repeating_unit)
            #
            current_ave_coverage = self.__get_cov_mean(path)
            # generate the extending candidates
            candidates_list = []
            candidates_list_overlap_c_nums = []
            for overlap_c_num in range(1, len(path) + 1):
                overlap_path = path[-overlap_c_num:]
                # stop adding extending candidate when the overlap is longer than our longest read alignment
                # stay within what data can tell
                if overlap_c_num > 0 and \
                        self.graph.get_path_internal_length(list(overlap_path) + [("", True)]) \
                        >= self.local_max_alignment_len:
                    break
                overlap_path = tuple(overlap_path)
                if overlap_path in self.__starting_subpath_to_readpaths:
                    # logger.debug("starting, " + str(self.__starting_subpath_to_readpaths[overlap_path]))
                    candidates_list.append(sorted(self.__starting_subpath_to_readpaths[overlap_path]))
                    candidates_list_overlap_c_nums.append(overlap_c_num)
            # logger.debug(candidates_list)
            # logger.debug(candidates_list_overlap_c_nums)
            if not candidates_list:
                # if no extending candidate based on starting subpath (from one end),
                # try to simultaneously extend both ends
                path_tuple = tuple(path)
                if path_tuple in self.__middle_subpath_to_readpaths:
                    candidates = sorted(self.__middle_subpath_to_readpaths[path_tuple])
                    weights = [self.__read_paths_counter[self.read_paths[read_id]] for read_id, strand in candidates]
                    weights = harmony_weights(weights, diff=self.__differ_f)
                    if self.__cov_inert:
                        cdd_cov = [self.__get_cov_mean(rev_p, exclude_path=path) for rev_p in candidates]
                        weights = [exp(log(weights[go_c])-abs(log(cov/current_ave_coverage)))
                                   for go_c, cov in enumerate(cdd_cov)]
                    read_id, strand = self.__random.choices(candidates, weights=weights)[0]
                    if strand:
                        path = list(self.read_paths[read_id])
                    else:
                        path = self.graph.reverse_path(self.read_paths[read_id])
                    return self.__heuristic_extend_path(
                        path, initial_mean=initial_mean, initial_std=initial_std)
            if not candidates_list:
                # if no extending candidates based on overlap info, try to extend based on the graph
                last_name, last_end = path[-1]
                next_connections = self.graph.vertex_info[last_name].connections[last_end]
                if next_connections:
                    candidates_rev = sorted(next_connections)
                    if self.__cov_inert:
                        # coverage inertia, more likely to extend to contigs with similar depths
                        cdd_cov = [self.__get_cov_mean(rev_p) for rev_p in candidates_rev]
                        weights = [exp(-abs(log(cov/current_ave_coverage))) for cov in cdd_cov]
                        next_name, next_end = self.__random.choices(candidates_rev, weights=weights)[0]
                    else:
                        next_name, next_end = self.__random.choice(candidates_rev)
                    return self.__heuristic_check_multiplicity(
                        initial_mean=initial_mean,
                        initial_std=initial_std,
                        path=path,
                        proposed_extension=[(next_name, not next_end)],
                        not_do_reverse=not_do_reverse)
                else:
                    if not_do_reverse:
                        logger.trace("      traversal ended without next vertex.")
                        return path
                    else:
                        logger.trace("      traversal reversed without next vertex.")
                        return self.__heuristic_extend_path(
                            self.graph.reverse_path(path),
                            not_do_reverse=True,
                            initial_mean=initial_mean,
                            initial_std=initial_std)
            else:
                candidates = []
                candidates_ovl_n = []
                weights = []
                for go_overlap, same_ov_cdd in enumerate(candidates_list):
                    # flatten the candidates_list: convert multiple dimensions into one dimension
                    candidates.extend(same_ov_cdd)
                    # record the overlap contig numbers in the flattened single-dimension candidates
                    ovl_c_num = candidates_list_overlap_c_nums[go_overlap]
                    candidates_ovl_n.extend([ovl_c_num] * len(same_ov_cdd))
                    # generate the weights for the single-dimension candidates
                    same_ov_w = [self.__read_paths_counter[self.read_paths[read_id]]
                                 for read_id, strand in same_ov_cdd]
                    same_ov_w = harmony_weights(same_ov_w, diff=self.__differ_f)
                    # RuntimeWarning: overflow encountered in exp of numpy, or math range error in exp of math
                    # change to dtype=np.float128?
                    same_ov_w = exp(np.array(log(same_ov_w) + log(self.__decay_f) * ovl_c_num, dtype=np.float128))
                    weights.extend(same_ov_w)
                if self.__cov_inert:
                    # logger.debug(candidates)
                    # logger.debug(candidates_ovl_n)
                    cdd_cov = [self.__get_cov_mean(self.read_paths[r_id][candidates_ovl_n[go_c]:])
                               for go_c, (r_id, r_strand) in enumerate(candidates)]
                    weights = [exp(log(weights[go_c])-abs(log(cov / current_ave_coverage)))
                               for go_c, cov in enumerate(cdd_cov)]
                chosen_cdd_id = self.__random.choices(range(len(candidates)), weights=weights)[0]
                read_id, strand = candidates[chosen_cdd_id]
                if strand:
                    new_extend = list(self.read_paths[read_id][candidates_ovl_n[chosen_cdd_id]:])
                else:
                    new_extend = self.graph.reverse_path(self.read_paths[read_id])[candidates_ovl_n[chosen_cdd_id]:]
                # logger.debug("path: " + str(path))
                # logger.debug("extend: " + str(new_extend))
                # logger.debug("closed_from_start: " + str(closed_from_start))
                # logger.debug("    candidate path: {} .. {} .. {}".format(path[:3], len(path), path[-3:]))
                # logger.debug("    extend path   : {}".format(new_extend))
                return self.__heuristic_check_multiplicity(
                    initial_mean=initial_mean,
                    initial_std=initial_std,
                    path=path,
                    proposed_extension=new_extend,
                    not_do_reverse=not_do_reverse)

    def __heuristic_check_multiplicity(
            self, initial_mean, initial_std, path, proposed_extension, not_do_reverse):
        """
        heuristically check the multiplicity and call a stop according to the vertex coverage and current counts
        normal distribution
        :param initial_cov:
        :param path:
        :param proposed_extension:
        :return:
        """
        assert len(proposed_extension)
        current_names = [v_n for v_n, v_e in path]
        current_names = {v_n: current_names.count(v_n) for v_n in set(current_names)}
        # extend_names = [v_n for v_n, v_e in path]
        # extend_names = {v_n: extend_names.count(v_n) for v_n in set(extend_names)}
        new_path = list(deepcopy(path))
        # if there is a vertex of proposed_extension that was not used in current path, accept the proposed_extension
        for v_name, v_end in proposed_extension:
            if v_name not in current_names:
                return self.__heuristic_extend_path(
                    new_path + list(proposed_extension),
                    not_do_reverse=not_do_reverse,
                    initial_mean=initial_mean,
                    initial_std=initial_std)
        # check the multiplicity of every vertices
        # check the likelihood of making longest extension first, than shorten the extension
        log_like_ratio_list = []
        log_like_ratio = 0.
        proposed_lengths = {_v_: self.graph.vertex_info[_v_].len for _v_, _e_ in proposed_extension}
        old_cov_mean, old_cov_std = self.__get_cov_mean(new_path, return_std=True)
        # logger.trace("    old_path: {}".format(self.graph.repr_path(path)))
        # logger.trace("    checking proposed_extension: {}".format(self.graph.repr_path(proposed_extension)))
        for v_name, v_end in proposed_extension:
            old_like = norm.logpdf(self.contig_coverages[v_name],
                                   loc=current_names[v_name] * old_cov_mean,
                                   scale=old_cov_std)
            # old_like += norm.logpdf(self.contig_coverages[v_name] / float(current_names[v_name]),
            #                         loc=initial_mean,
            #                         scale=initial_std)
            new_cov_mean, new_cov_std = self.__get_cov_mean(new_path + [(v_name, v_end)], return_std=True)
            new_like = norm.logpdf(self.contig_coverages[v_name],
                                   loc=(current_names[v_name] + 1) * new_cov_mean,
                                   scale=new_cov_std)
            # new_like += norm.logpdf(self.contig_coverages[v_name] / float(current_names[v_name] + 1),
            #                         loc=initial_mean,
            #                         scale=initial_std)
            old_cov_mean, old_cov_std = new_cov_mean, new_cov_std
            log_like_ratio += (new_like - old_like) * proposed_lengths[v_name]  # weighted by length
            log_like_ratio_list.append(log_like_ratio)
            # logger.trace("      initial_mean: %.4f, old_mean: %.4f (%.4f), proposed_mean: %.4f (%.4f)" % (
            #     initial_mean, old_cov_mean, old_cov_std, new_cov_mean, new_cov_std))
            # logger.trace("      old_like: {},     proposed_like: {}".format(old_like, new_like))
            new_path.append((v_name, v_end))
        # de-weight the log likes
        longest_ex_len = len(proposed_extension)
        v_lengths = [proposed_lengths[_v_] for _v_, _e_ in proposed_extension]
        accumulated_v_lengths = []
        for rev_go in range(len(log_like_ratio_list)):
            accumulated_v_lengths.insert(0, sum(v_lengths[:longest_ex_len - rev_go]))
        log_like_ratio_list = [_llr / accumulated_v_lengths[_go] for _go, _llr in enumerate(log_like_ratio_list)]
        # step-by-step shorten the extension
        for rev_go, log_like_ratio in enumerate(log_like_ratio_list[::-1]):
            proposed_end = longest_ex_len - rev_go
            # de-weight by accumulated length
            if log_like_ratio > log(self.__random.random()):
                return self.__heuristic_extend_path(
                    list(deepcopy(path)) + list(proposed_extension[:proposed_end]),
                    not_do_reverse=not_do_reverse,
                    initial_mean=initial_mean,
                    initial_std=initial_std)
        else:
            if not_do_reverse:
                # logger.trace("    traversal ended to fit {}'s coverage.".format(proposed_extension[0][0]))
                # logger.trace("    checked likes: {}".format(like_ratio_list))
                return list(deepcopy(path))
            else:
                # logger.trace("    traversal reversed to fit {}'s coverage.".format(proposed_extension[0][0]))
                # logger.trace("    checked likes: {}".format(like_ratio_list))
                return self.__heuristic_extend_path(
                    self.graph.reverse_path(list(deepcopy(path))),
                    not_do_reverse=True,
                    initial_mean=initial_mean,
                    initial_std=initial_std)

    def __index_start_subpath(self, subpath, read_id, strand):
        """
        :param subpath: tuple
        :param read_id: int, read id in self.read_paths
        :param strand: bool
        :return:
        """
        if subpath in self.__starting_subpath_to_readpaths:
            self.__starting_subpath_to_readpaths[subpath].add((read_id, strand))
        else:
            self.__starting_subpath_to_readpaths[subpath] = {(read_id, strand)}

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

    def __get_cov_mean(self, path, exclude_path=None, return_std=False):
        assert len(path)
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
            if v_name not in self.graph.vertex_info:
                logger.error(v_name + " not found in the assembly graph!")
            else:
                v_covers.append(self.contig_coverages[v_name] / float(v_names[v_name]))
                v_lengths.append(self.graph.vertex_info[v_name].len * v_names[v_name])
        # logger.trace("        > cal path: {}".format(self.graph.repr_path(path)))
        # logger.trace("        > cover values: {}".format(v_covers))
        # logger.trace("        > cover weights: {}".format(v_lengths))
        mean = np.average(v_covers, weights=v_lengths)
        if return_std:
            std = np.average((np.array(v_covers) - mean) ** 2, weights=v_lengths) ** 0.5
            return mean, std
        else:
            return mean

    def __directed_graph_solver(
            self, ongoing_paths, next_connections, vertices_left, in_all_start_ve):
        if not vertices_left:
            new_paths, new_standardized = self.graph.get_standardized_isomer(ongoing_paths)
            if new_standardized not in self.components_counts:
                self.components_counts[new_standardized] = 1
                self.components.append(new_standardized)
            else:
                self.components_counts[new_standardized] += 1
            return

        find_next = False
        for next_vertex, next_end in next_connections:
            # print("next_vertex", next_vertex, next_end)
            if next_vertex in vertices_left:
                find_next = True
                new_paths = deepcopy(ongoing_paths)
                new_left = deepcopy(vertices_left)
                new_paths[-1].append((next_vertex, not next_end))
                new_left[next_vertex] -= 1
                if not new_left[next_vertex]:
                    del new_left[next_vertex]
                new_connections = sorted(self.graph.vertex_info[next_vertex].connections[not next_end])
                if not new_left:
                    new_paths, new_standardized = self.graph.get_standardized_isomer(new_paths)
                    if new_standardized not in self.components_counts:
                        self.components_counts[new_standardized] = 1
                        self.components.append(new_standardized)
                    else:
                        self.components_counts[new_standardized] += 1
                    return
                else:
                    self.__directed_graph_solver(new_paths, new_connections, new_left, in_all_start_ve)
        if not find_next:
            new_all_start_ve = deepcopy(in_all_start_ve)
            while new_all_start_ve:
                new_start_vertex, new_start_end = new_all_start_ve.pop(0)
                if new_start_vertex in vertices_left:
                    new_paths = deepcopy(ongoing_paths)
                    new_left = deepcopy(vertices_left)
                    new_paths.append([(new_start_vertex, new_start_end)])
                    new_left[new_start_vertex] -= 1
                    if not new_left[new_start_vertex]:
                        del new_left[new_start_vertex]
                    new_connections = sorted(self.graph.vertex_info[new_start_vertex].connections[new_start_end])
                    if not new_left:
                        new_paths, new_standardized = self.graph.get_standardized_isomer(new_paths)
                        if new_standardized not in self.components_counts:
                            self.components_counts[new_standardized] = 1
                            self.components.append(new_standardized)
                        else:
                            self.components_counts[new_standardized] += 1
                    else:
                        self.__directed_graph_solver(new_paths, new_connections, new_left, new_all_start_ve)
                        break
            if not new_all_start_ve:
                return

    def __circular_directed_graph_solver(self,
        ongoing_path,
        next_connections,
        vertices_left,
        check_all_kinds,
        palindromic_repeat_vertices,
        ):
        """
        recursively exhaust all circular paths, deprecated for now
        :param ongoing_path:
        :param next_connections:
        :param vertices_left:
        :param check_all_kinds:
        :param palindromic_repeat_vertices:
        :return:
        """
        if not vertices_left:
            new_path = deepcopy(ongoing_path)
            if palindromic_repeat_vertices:
                new_path = [(this_v, True) if this_v in palindromic_repeat_vertices else (this_v, this_e)
                            for this_v, this_e in new_path]
            if check_all_kinds:
                rev_path = self.graph.reverse_path(new_path)
                this_path_derived = [new_path, rev_path]
                for change_start in range(1, len(new_path)):
                    this_path_derived.append(new_path[change_start:] + new_path[:change_start])
                    this_path_derived.append(rev_path[change_start:] + rev_path[:change_start])
                standardized_path = tuple(sorted(this_path_derived)[0])
                if standardized_path not in self.components_counts:
                    self.components_counts[standardized_path] = 1
                    self.components.append(standardized_path)
                else:
                    self.components_counts[standardized_path] += 1
            else:
                new_path = tuple(new_path)
                if new_path not in self.components_counts:
                    self.components_counts[new_path] = 1
                    self.components.append(new_path)
                else:
                    self.components_counts[new_path] += 1
            return

        for next_vertex, next_end in next_connections:
            # print("next_vertex", next_vertex)
            if next_vertex in vertices_left:
                new_path = deepcopy(ongoing_path)
                new_left = deepcopy(vertices_left)
                new_path.append((next_vertex, not next_end))
                new_left[next_vertex] -= 1
                if not new_left[next_vertex]:
                    del new_left[next_vertex]
                new_connections = self.graph.vertex_info[next_vertex].connections[not next_end]
                if not new_left:
                    if (self.__start_vertex, not self.__start_direction) in new_connections:
                        if palindromic_repeat_vertices:
                            new_path = [
                                (this_v, True) if this_v in palindromic_repeat_vertices else (this_v, this_e)
                                for this_v, this_e in new_path]
                        if check_all_kinds:
                            rev_path = self.graph.reverse_path(new_path)
                            this_path_derived = [new_path, rev_path]
                            for change_start in range(1, len(new_path)):
                                this_path_derived.append(new_path[change_start:] + new_path[:change_start])
                                this_path_derived.append(rev_path[change_start:] + rev_path[:change_start])
                            standardized_path = tuple(sorted(this_path_derived)[0])
                            if standardized_path not in self.components_counts:
                                self.components_counts[standardized_path] = 1
                                self.components.append(standardized_path)
                            else:
                                self.components_counts[standardized_path] += 1
                        else:
                            new_path = tuple(new_path)
                            if new_path not in self.components_counts:
                                self.components_counts[new_path] = 1
                                self.components.append(new_path)
                            else:
                                self.components_counts[new_path] += 1
                        return
                    else:
                        return
                else:
                    new_connections = sorted(new_connections)
                    self.__circular_directed_graph_solver(new_path, new_connections, new_left, check_all_kinds,
                                                          palindromic_repeat_vertices)


