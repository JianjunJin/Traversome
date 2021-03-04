from loguru import logger
from ifragaria.utils import harmony_weights, ProcessingGraphFailed
from copy import deepcopy
from scipy.stats import norm
from math import log, exp
import numpy as np
import random


class GraphAlignmentPathGenerator(object):
    """
    generate heuristic components (isomers & sub-chromosomes) from alignments.
    Here the components are not necessarily identical in contig composition.
    """

    def __init__(self,
                 assembly_graph,
                 graph_alignment,
                 num_search=10000,
                 differ_f=1.,
                 decay_f=20.,
                 cov_inert=1.):
        """
        :param assembly_graph:
        :param graph_alignment:
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
        """
        self.graph = assembly_graph
        self.alignment = graph_alignment
        self.num_search = num_search
        assert 0 <= differ_f
        self.__differ_f = differ_f
        assert 0 <= decay_f
        self.__decay_f = decay_f
        assert 0 <= cov_inert
        self.__cov_inert = cov_inert

        # to be generated
        self.local_max_alignment_len = None
        self.read_paths = list()
        self.__read_paths_counter = dict()
        # self.__vertex_to_readpath = {vertex: set() for vertex in self.graph.vertex_info}
        self.__starting_subpath_to_readpaths = {}
        self.__middle_subpath_to_readpaths = {}
        self.__read_paths_counter_indexed = False
        self.contig_coverages = {}

        self.components = list()
        self.components_counts = dict()


    def get_heuristic_components(self, circular, random_seed):
        logger.warning("This function is under testing .. ")
        random.seed(random_seed)
        if not self.__read_paths_counter_indexed:
            self.index_readpaths_subpaths()
        self.estimate_contig_coverages_from_read_paths()
        self.get_heuristic_paths(force_circular=circular)


    # def get_heuristic_circular_isomers(self, random_seed):
    #     # based on alignments
    #     logger.warning("This function is under testing .. ")
    #     random.seed(random_seed)
    #     if not self.__read_paths_counter_indexed:
    #         self.index_readpaths_subpaths()
    #
    #
    # def get_all_circular_isomers(self):
    #     # based on alignments
    #     self.components = list()
    #     self.components_set = set()
    #     if self.__re_estimate_multiplicity or not self.graph.vertex_to_copy:
    #         self.graph.estimate_multiplicity_by_cov(mode=self.mode)
    #         self.graph.estimate_multiplicity_precisely()
    #
    #     # for palindromic repeats
    #     if self.graph.detect_palindromic_repeats(redo=False):
    #         logger.debug("Palindromic repeats detected. "
    #                      "Different paths generating identical sequence will be merged.")
    #
    #     #
    #     self.graph.update_orf_total_len()
    #     if 1 not in self.graph.copy_to_vertex:
    #         do_check_all_start_kinds = True
    #         self.__start_vertex = sorted(self.graph.vertex_info,
    #                                      key=lambda x: (-self.graph.vertex_info[x].len,
    #                                                     -max(self.graph.vertex_info[x].other_attr["orf"][True][
    #                                                              "sum_len"],
    #                                                          self.graph.vertex_info[x].other_attr["orf"][False][
    #                                                              "sum_len"]),
    #                                                     x))[0]
    #         self.__start_direction = True
    #     else:
    #         # start from a single copy vertex, no need to check all kinds of start vertex
    #         do_check_all_start_kinds = False
    #         self.__start_vertex = sorted(self.graph.copy_to_vertex[1])[0]
    #         self.__start_direction = True
    #
    #     # each contig stored format:
    #     first_path = [(self.__start_vertex, self.__start_direction)]
    #     first_connections = sorted(self.graph.vertex_info[self.__start_vertex].connections[self.__start_direction])
    #     vertex_to_copy = deepcopy(self.graph.vertex_to_copy)
    #     vertex_to_copy[self.__start_vertex] -= 1
    #     if vertex_to_copy[self.__start_vertex] <= 0:
    #         del vertex_to_copy[self.__start_vertex]
    #     self.__circular_directed_graph_solver(first_path, first_connections, vertex_to_copy, do_check_all_start_kinds,
    #                                           self.graph.palindromic_repeats)
    #
    #     if not self.components:
    #         raise ProcessingGraphFailed("Detecting path(s) from remaining graph failed!")


    def index_readpaths_subpaths(self, filter_by_graph=True):
        self.__read_paths_counter = dict()
        alignment_lengths = []
        if filter_by_graph:
            for gaf_record in self.alignment:
                this_read_path = tuple(self.graph.get_standardized_path(gaf_record.path, dc=False))
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
                this_read_path = tuple(self.graph.get_standardized_path(gaf_record.path, dc=False))
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
        self.contig_coverages = {v_name: 0 for v_name in self.graph.vertex_info}
        for read_path in self.read_paths:
            for v_name, v_end in read_path:
                if v_name in self.contig_coverages:
                    self.contig_coverages[v_name] += 1


    def get_heuristic_paths(self, force_circular):
        for count_search in range(1, self.num_search + 1):
            new_path = self.graph.get_standardized_path(self.__heuristic_extend_path([], force_circular=force_circular))
            if new_path in self.components_counts:
                self.components_counts[new_path] += 1
                logger.debug("{} unique paths found in {} trials, {} trials left".format(
                    len(self.components), count_search, self.num_search - count_search))
            else:
                self.components_counts[new_path] = 1
                self.components.append(new_path)
                logger.info("{} unique paths found in {} trials, {} trials left".format(
                    len(self.components), count_search, self.num_search - count_search))


    def __heuristic_extend_path(self, path, not_do_reverse=False, force_circular=False):
        """

        improvement needed

        :param path: empty path like [] or starting path like [("v1", True), ("v2", False)]
        :param not_do_reverse: a mark to stop searching from the reverse end.
        :return: a candidate component. e.g. [("v0", True), ("v1", True), ("v2", False), ("v3", True)]
        """
        if not path:
            # randomly choose the read path and the direction
            read_path = random.choice(self.read_paths)  # change the weight (-~ depth) to flatten the search later
            if random.random() > 0.5:
                read_path = self.graph.reverse_path(read_path)
            path = list(read_path)
            return self.__heuristic_extend_path(path, not_do_reverse=False, force_circular=force_circular)
        else:
            # keep going in a circle util the path length reaches beyond the longest read alignment
            # stay within what data can tell
            repeating_unit = self.graph.roll_path(path)
            if len(path) > len(repeating_unit) and \
                    self.graph.get_path_internal_length(path) >= self.local_max_alignment_len:
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
                        cdd_cov = [self.__get_cov_mean(rev_p, path) for rev_p in candidates]
                        weights = [weights[go_c] * exp(-abs(log(cov/current_ave_coverage)))
                                   for go_c, cov in enumerate(cdd_cov)]
                    read_id, strand = random.choices(candidates, weights=weights)[0]
                    if strand:
                        path = list(self.read_paths[read_id])
                    else:
                        path = self.graph.reverse_path(self.read_paths[read_id])
                    return self.__heuristic_extend_path(path, force_circular=force_circular)
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
                        next_name, next_end = random.choices(candidates_rev, weights=weights)[0]
                    else:
                        next_name, next_end = random.choice(candidates_rev)
                    return self.__heuristic_check_extending_multiplicity(
                        path=path,
                        extend_path=[(next_name, not next_end)],
                        not_do_reverse=not_do_reverse,
                        force_circular=force_circular)
                else:
                    if not_do_reverse:
                        return path
                    else:
                        return self.__heuristic_extend_path(
                            self.graph.reverse_path(path), not_do_reverse=True, force_circular=force_circular)
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
                    same_ov_w *= exp(log(self.__decay_f) * ovl_c_num)
                    weights.extend(same_ov_w)
                if self.__cov_inert:
                    # logger.debug(candidates)
                    # logger.debug(candidates_ovl_n)
                    cdd_cov = [self.__get_cov_mean(self.read_paths[r_id][candidates_ovl_n[go_c]:])
                               for go_c, (r_id, r_strand) in enumerate(candidates)]
                    weights = [weights[go_c] * exp(-abs(log(cov / current_ave_coverage)))
                               for go_c, cov in enumerate(cdd_cov)]
                chosen_cdd_id = random.choices(range(len(candidates)), weights=weights)[0]
                read_id, strand = candidates[chosen_cdd_id]
                if strand:
                    new_extend = list(self.read_paths[read_id][candidates_ovl_n[chosen_cdd_id]:])
                else:
                    new_extend = self.graph.reverse_path(self.read_paths[read_id])[candidates_ovl_n[chosen_cdd_id]:]
                # logger.debug("path: " + str(path))
                # logger.debug("extend: " + str(new_extend))
                # logger.debug("closed_from_start: " + str(closed_from_start))
                return self.__heuristic_check_extending_multiplicity(
                    path=path,
                    extend_path=new_extend,
                    not_do_reverse=not_do_reverse,
                    force_circular=force_circular)


    def __heuristic_check_extending_multiplicity(self, path, extend_path, not_do_reverse, force_circular):
        """
        normal distribution
        :param path:
        :param extend_path:
        :return:
        """
        current_names = [v_n for v_n, v_e in path]
        current_names = {v_n: current_names.count(v_n) for v_n in set(current_names)}
        # extend_names = [v_n for v_n, v_e in path]
        # extend_names = {v_n: extend_names.count(v_n) for v_n in set(extend_names)}
        new_path = deepcopy(path)
        for v_name, v_end in extend_path:
            if v_name in current_names:
                old_cov_mean, old_cov_std = self.__get_cov_mean(new_path, return_std=True)
                new_cov_mean, new_cov_std = self.__get_cov_mean(new_path + [(v_name, v_end)], return_std=True)
                old_like = norm.pdf(self.contig_coverages[v_name],
                                    loc=current_names[v_name] * old_cov_mean,
                                    scale=old_cov_std)
                new_like = norm.pdf(self.contig_coverages[v_name],
                                    loc=(current_names[v_name] + 1) * new_cov_mean,
                                    scale=new_cov_std)
                like_ratio = new_like / old_like
                if like_ratio > 1:
                    new_path.append((v_name, v_end))
                elif force_circular and not self.graph.is_circular_path(self.graph.roll_path(new_path)):
                    new_path.append((v_name, v_end))
                else:
                    if random.random() < like_ratio:
                        new_path.append((v_name, v_end))
                    else:
                        if not_do_reverse:
                            return new_path
                        else:
                            return self.__heuristic_extend_path(
                                self.graph.reverse_path(new_path), not_do_reverse=True, force_circular=force_circular)
            else:
                new_path.append((v_name, v_end))
        return self.__heuristic_extend_path(new_path, not_do_reverse=not_do_reverse, force_circular=force_circular)


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
        recursively exhaust all circular paths
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


