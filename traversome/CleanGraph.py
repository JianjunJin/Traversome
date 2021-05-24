
from loguru import logger
from traversome.utils import generate_clusters_from_connections
from itertools import combinations
from collections import OrderedDict
from copy import deepcopy
from math import log
import os


class CleanGraph(object):
    """
    unfinished
    """
    def __init__(self, traversome):
        self.traversome = traversome
        self.graph = traversome.graph
        self.read_paths = traversome.read_paths
        self.__shuffled = traversome.shuffled
        # to be generated
        self.max_read_path_size = 0
        self.id_to_read_paths = OrderedDict()
        self.v_name_to_read_paths = OrderedDict()            # {v_name: {read_id: {loc: strand(end)}}}
        self.rep_p_to_read_paths = OrderedDict()     # {rep_path: {terminal_pair:
                                                             #               {"path_counts": int,
                                                             #                "read_id": [int],
                                                             #                "pmer_pos": [int],
                                                             #                "rep_strand": [bool]}}}
        self.__solutions = OrderedDict()                     # {rep_path: {terminal_pair_group: True}}
        self.__solved = set()                                # {rep_path}
        self.__cutting_tmp = {}

    def run(self, min_effective_count=10, ignore_ratio=0.001):
        """
        use long reads to prune nonexistent paths from the graph
        :param min_effective_count: INT,
        a solution will not be made unless all supposed new paths are supported by this number of reads
        :param ignore_ratio: FLOAT,
        any conflicting minor path that supported by this ratio of reads will be ignored during solution
        :return:
        TODO add new path to bridge broken areas
        """
        if not self.read_paths:
            self.traversome.generate_read_paths()
        self.__index_read_paths()
        finished = False
        count_r = 0
        while not finished:
            count_r += 1
            logger.debug("========== Iteration {} ==========".format(count_r))
            self.traversome.generate_maximum_read_path_size()
            self.max_read_path_size = self.traversome.max_read_path_size
            logger.debug("Maximum read path size: {}".format(self.max_read_path_size))
            self.__index_read_path_mers()
            logger.debug("Num of candidate repeat paths: {}".format(len(self.rep_p_to_read_paths)))
            self.__generating_solutions(min_effective_count)
            logger.debug("Num of solutions: {}".format(len(self.__solutions)))
            finished = self.__solve_repeats(ignore_ratio)
            if self.traversome.keep_temp:
                self.graph.write_to_gfa(os.path.join(self.traversome.outdir, "cleaned.{}.gfa".format(count_r)))

    def __index_read_paths(self):
        # self.id_to_read_paths:     for reducing memory cost
        # self.v_name_to_read_paths: for speeding up looking up during read_path renaming
        for go_r, read_path in enumerate(self.read_paths):
            self.id_to_read_paths[go_r] = read_path
            for go_v, (v_name, v_end) in enumerate(read_path):
                if v_name not in self.v_name_to_read_paths:
                    self.v_name_to_read_paths[v_name] = OrderedDict()
                if go_r not in self.v_name_to_read_paths[v_name]:
                    self.v_name_to_read_paths[v_name][go_r] = OrderedDict()
                self.v_name_to_read_paths[v_name][go_r][go_v] = v_end

    def __index_read_path_mers(self):
        # Index read_path_mers by pmer (left_v, rep_path, right_v), with depth counted.
        # 1. Only record repeat_candidates that have branching sites inside
        # 2. Only record the non-leaking repeat_candidates (J note 2021-04-29),
        # otherwise neither self.__rename_read_paths() nor re-mapping
        # will work properly for assigning reads to the new graph.
        branching_ends = self.graph.get_branching_ends()
        logger.debug("Num of branching_ends: {}".format(len(branching_ends)))
        logger.debug("                     : {}".format(branching_ends))
        for path_mer_size in range(2, self.max_read_path_size + 1):
            for go_r in self.id_to_read_paths:
                read_path = self.id_to_read_paths[go_r]
                this_r_p_size = len(read_path)
                if this_r_p_size < path_mer_size:
                    continue
                path_counts = len(self.read_paths[read_path])
                # skip read paths that does not have branching sites inside
                # 1. check forward
                for (v_name, v_end) in read_path[:-1]:
                    if (v_name, v_end) in branching_ends:
                        break
                else:
                    # 2. check reverse
                    for (v_name, v_end) in read_path[1:]:
                        if (v_name, not v_end) in branching_ends:
                            break
                    else:
                        continue
                # chopping read_paths to pmer (like kmers)
                # index pmers by rep_path, with depth counted
                for go_mer in range(this_r_p_size - path_mer_size):
                    pmer, p_strand = self.graph.get_standardized_path_with_strand(
                        raw_path=read_path[go_mer: go_mer + path_mer_size], 
                        detect_circular=False)
                    # skip read paths that does not have branching sites inside
                    if len(pmer) == 2:
                        check_offset = (None, None)
                    else:
                        check_offset = (1, -1)
                    for (v_name, v_end) in pmer[check_offset[0]:-1]:
                        if (v_name, v_end) in branching_ends:
                            break
                    else:
                        # 2. check reverse
                        for (v_name, v_end) in pmer[1:check_offset[1]]:
                            if (v_name, not v_end) in branching_ends:
                                break
                        else:
                            continue
                    # logger.debug("         {}:{} {}".format(go_mer, go_mer + path_mer_size, pmer))
                    rep_path, rep_strand = self.graph.get_standardized_path_with_strand(
                        raw_path=pmer[1:-1],
                        detect_circular=False)
                    if rep_path not in self.rep_p_to_read_paths:
                        self.rep_p_to_read_paths[rep_path] = OrderedDict()
                    # record the terminal_pair
                    if rep_strand:
                        terminal_pair = (pmer[0], pmer[-1])
                    else:
                        terminal_pair = self.graph.reverse_path((pmer[0], pmer[-1]))
                    if terminal_pair not in self.rep_p_to_read_paths[rep_path]:
                        self.rep_p_to_read_paths[rep_path][terminal_pair] = \
                            {"path_counts": 0, "read_id": [], "pmer_pos": [], "rep_strand": []} 
                    self.rep_p_to_read_paths[rep_path][terminal_pair]["path_counts"] += path_counts
                    self.rep_p_to_read_paths[rep_path][terminal_pair]["read_id"].append(go_r)
                    self.rep_p_to_read_paths[rep_path][terminal_pair]["pmer_pos"].append(go_mer)
                    # self.rep_p_to_read_paths[rep_path][terminal_pair]["pmer_strand"].append(p_strand)
                    self.rep_p_to_read_paths[rep_path][terminal_pair]["rep_strand"].append(rep_strand == p_strand)
        # delete leaking repeat_candidates
        for rep_path, terminal_pairs_info in list(self.rep_p_to_read_paths.items()):
            if not self.graph.is_no_leaking_path(path=rep_path, terminal_pairs=list(terminal_pairs_info)):
                del self.rep_p_to_read_paths[rep_path]

    def __generating_solutions(self, min_effective_count):
        # stepwise looking for conflicting paths
        # initialize
        self.__solutions = OrderedDict()
        for repeat_candidate, terminal_pairs_info in self.rep_p_to_read_paths.items():
            # if repeat_candidate == (('251439', False),):
            #     logger.debug("analyzing r={}".format(repeat_candidate))
            # skip solved
            if repeat_candidate in self.__solved:
                continue
            self.__solutions[repeat_candidate] = OrderedDict()
            if len(terminal_pairs_info) == 1:
                if terminal_pairs_info[list(terminal_pairs_info)[0]]["path_counts"] >= min_effective_count:
                    self.__solutions[repeat_candidate][tuple(terminal_pairs_info)] = True
            else:
                # use conflicting_terminals to record the conflicting terminal_pairs as edges
                conflicting_terminals = OrderedDict([(_term_p, []) for _term_p in terminal_pairs_info])
                for terminal_pair_1, terminal_pair_2 in combinations(terminal_pairs_info, 2):
                    # if the left terminal matches but the right terminal unmatches, it create a conflict, vice versa.
                    if (terminal_pair_1[0] == terminal_pair_2[0]) != (terminal_pair_1[1] == terminal_pair_2[1]):
                        conflicting_terminals[terminal_pair_1].append(terminal_pair_2)
                        conflicting_terminals[terminal_pair_2].append(terminal_pair_1)

                # if repeat_candidate == (('251439', False),):
                #     logger.debug("terminal_pairs_info={}".format(terminal_pairs_info))
                #     logger.debug("terminal_pairs_list={}".format(list(terminal_pairs_info)))
                #     logger.debug("conflicting_terminals={}".format(conflicting_terminals))
                #
                # group terminal_pairs by conflicting networks
                terminal_clusters = generate_clusters_from_connections(terminal_pairs_info, conflicting_terminals)

                # if repeat_candidate == (('251439', False),):
                #     logger.debug("terminal_clusters={}".format(terminal_clusters))

                # if there was only one group
                if len(terminal_clusters) == 1:
                    # if that group has only one terminal_pair
                    if len(terminal_clusters[0]) == 1:
                        if terminal_pairs_info[list(terminal_clusters[0])[0]]["path_counts"] >= min_effective_count:
                            self.__solutions[repeat_candidate][tuple(terminal_clusters[0])] = True
                # if there were multiple groups:
                # terminal_pairs conflict with each other within each group,
                # but compatible with all terminal_pairs in another group
                else:
                    for terminal_cl in terminal_clusters:
                        self.__solutions[repeat_candidate][tuple(sorted(terminal_cl))] = True
        for repeat_candidate, terminal_pairs_dict in list(self.__solutions.items()):
            if not terminal_pairs_dict:
                del self.__solutions[repeat_candidate]

    def __solve_repeats(self, ignore_ratio):
        all_repeats_solved = True
        renamed_vertices = set()
        for rep_candidate, terminal_pairs_dict in sorted(self.__solutions.items(), key=lambda x: (len(x[0]), x)):
            if not terminal_pairs_dict:
                continue
            terminal_pairs_list = list(terminal_pairs_dict)
            logger.debug("solving {} : {} : {}".format(rep_candidate,
                                                       " ".join([str(len(_g)) for _g in terminal_pairs_list]),
                                                       " ".join([str(_g) for _g in terminal_pairs_list])))
            if len(rep_candidate) == 0:
                #########
                # prune unsupported connections formed by involved ends
                involved_ends = set()
                keep_connections = set()
                for grouped_pairs in terminal_pairs_list:
                    for (keep_lt_n, keep_lt_e), (keep_rt_n, keep_rt_e) in grouped_pairs:
                        involved_ends.add((keep_lt_n, keep_lt_e))
                        involved_ends.add((keep_rt_n, not keep_rt_e))
                        keep_connections.add(((keep_lt_n, keep_lt_e), (keep_rt_n, not keep_rt_e)))
                        keep_connections.add(((keep_rt_n, not keep_rt_e), (keep_lt_n, keep_lt_e)))
                for (n_1, e_1) in sorted(involved_ends):
                    for n_2, e_2 in self.graph.vertex_info[n_1].connections[e_1]:
                        if (n_2, e_2) in involved_ends:
                            if ((n_1, e_1), (n_2, e_2)) not in keep_connections:
                                self.graph.vertex_info[n_1].connections[e_1].pop((n_2, e_2), None)
                                self.graph.vertex_info[n_2].connections[e_2].pop((n_1, e_1), None)
                                logger.debug("        pruning1 {}".format(((n_1, e_1), (n_2, e_2))))
                self.__solved.add(rep_candidate)
                # no reads processed in this case
            else:
                # check whether it is involved (therefore renamed) in current iteration of __solve_repeats
                # if it is renamed, skip it for next iteration
                involved_in_vertex_duplication = False
                for grouped_pairs in terminal_pairs_list:
                    for (lt_n, lt_e), (rt_n, rt_e) in grouped_pairs:
                        if lt_n in renamed_vertices or rt_n in renamed_vertices:
                            involved_in_vertex_duplication = True
                            break
                    if involved_in_vertex_duplication:
                        break
                if involved_in_vertex_duplication:
                    all_repeats_solved = False
                    continue
                for v_n, v_e in rep_candidate:
                    if v_n in renamed_vertices:
                        involved_in_vertex_duplication = True
                        break
                if involved_in_vertex_duplication:
                    all_repeats_solved = False
                    continue
                else:
                    self.__solved.add(rep_candidate)
                    #########
                    # if no internal branching connections to unrelated vertices (leaking),
                    # prune unsupported connections to the rep_candidate (involved ends)
                    if self.graph.is_no_leaking_path(
                            path=rep_candidate,
                            terminal_pairs=[_term_p for _group_p in terminal_pairs_list for _term_p in _group_p]):
                        (i_lt_n, i_lt_e), (i_rt_n, i_rt_e) = rep_candidate[0], rep_candidate[-1]
                        keep_connections = set()
                        logger.debug("        keeping t {}".format(str(terminal_pairs_list)))
                        # keep connections to terminal_pairs_list
                        for grouped_pairs in terminal_pairs_list:
                            logger.debug("        keeping p {}".format(grouped_pairs))
                            for (keep_lt_n, keep_lt_e), (keep_rt_n, keep_rt_e) in grouped_pairs:
                                keep_connections.add(((keep_lt_n, keep_lt_e), (i_lt_n, not i_lt_e)))
                                # keep_connections.add(((i_lt_n, not i_lt_e), (keep_lt_n, keep_lt_e)))  #
                                keep_connections.add(((i_rt_n, i_rt_e), (keep_rt_n, not keep_rt_e)))
                                # keep_connections.add(((keep_rt_n, not keep_rt_e), (i_rt_n, i_rt_e)))  #
                        # keep connections to repeat
                        terminal_vs = {i_lt_n, i_rt_n}
                        for go_v, (rep_n, rep_e) in enumerate(rep_candidate[:-1]):
                            next_n, next_e = rep_candidate[go_v + 1]
                            if rep_n in terminal_vs or next_n in terminal_vs:
                                keep_connections.add(((rep_n, rep_e), (next_n, not next_e)))
                                keep_connections.add(((next_n, not next_e), (rep_n, rep_e)))
                        logger.debug("        keeping c {}".format(keep_connections))
                        # pruning
                        for (n_1, e_1) in list(self.graph.vertex_info[i_lt_n].connections[not i_lt_e]):
                            if ((n_1, e_1), (i_lt_n, not i_lt_e)) not in keep_connections:
                                logger.debug("        pruning2 {} ~ {}".format(
                                    ((n_1, e_1), (i_lt_n, not i_lt_e)), rep_candidate))
                                self.graph.vertex_info[n_1].connections[e_1].pop((i_lt_n, not i_lt_e), None)
                                self.graph.vertex_info[i_lt_n].connections[not i_lt_e].pop((n_1, e_1), None)

                        for (n_2, e_2) in list(self.graph.vertex_info[i_rt_n].connections[i_rt_e]):
                            if ((i_rt_n, i_rt_e), (n_2, e_2)) not in keep_connections:
                                logger.debug("        pruning3 {} ~ {}".format(
                                    ((i_rt_n, i_rt_e), (n_2, e_2)), rep_candidate))
                                self.graph.vertex_info[n_2].connections[e_2].pop((i_rt_n, i_rt_e), None)
                                self.graph.vertex_info[i_rt_n].connections[i_rt_e].pop((n_2, e_2), None)
                    # duplicate rep_candidate and separate them into different paths
                    if len(terminal_pairs_list) == 1:
                        if len(list(terminal_pairs_list)[0]) == 1:
                            this_terminal = list(terminal_pairs_list)[0][0]
                            # is_no_leaking_path was already checked before solving
                            # if self.graph.is_no_leaking_path(path=rep_candidate, terminal_pairs=[this_terminal]):
                            # record renamed vertices
                            # ironing over a bubble if there are sequential repeats
                            renamed_vertices |= self.unfold_graph_along_path(rep_candidate,
                                                                             terminal_pair=this_terminal,
                                                                             unfold_read_paths_accordingly=True,
                                                                             check_leakage=False)
                        else:
                            pass
                    else:
                        # calculate the depth ratios
                        group_counts = []
                        for grouped_pairs in terminal_pairs_list:
                            group_counts.append(0)
                            for terminal_pair in grouped_pairs:
                                group_counts[-1] += \
                                    self.rep_p_to_read_paths[rep_candidate][terminal_pair]["path_counts"]
                        max_count = float(max(group_counts))
                        go_g = 0
                        while go_g < len(group_counts):
                            if ignore_ratio == 0. or abs(log(group_counts[go_g] / max_count)) < abs(log(ignore_ratio)):
                                go_g += 1
                            else:
                                del group_counts[go_g]
                                for term_p in terminal_pairs_list[go_g]:
                                    for read_id in self.rep_p_to_read_paths[rep_candidate][term_p]["read_id"]:
                                        if read_id in self.id_to_read_paths:
                                            # self.__del_read_path(read_id)
                                            for pm_p in self.rep_p_to_read_paths[rep_candidate][term_p]["pmer_pos"]:
                                                self.__add_to_cutting_list(
                                                    read_id=read_id,
                                                    trim_range=[pm_p + 1, pm_p + 1 + len(rep_candidate)])
                                del terminal_pairs_list[go_g]
                        self.__execute_cutting_list()
                        #
                        if len(group_counts) == 1:
                            if len(terminal_pairs_list[0]) == 1:
                                renamed_vertices |= \
                                    self.unfold_graph_along_path(rep_candidate,
                                                                 terminal_pair=terminal_pairs_list[0][0],
                                                                 unfold_read_paths_accordingly=True,
                                                                 check_leakage=False)
                            else:
                                pass
                        else:
                            # multiplicate the rep_candidate
                            self.split_the_repeats(rep_candidate,
                                                   n_groups=len(group_counts),
                                                   terminal_pair_group_list=terminal_pairs_list,
                                                   weights=group_counts,
                                                   distribute_read_paths_accordingly=True)
                            for v_n, v_e in rep_candidate:
                                renamed_vertices.add(v_n)
        return all_repeats_solved

    def unfold_graph_along_path(
            self,
            input_path,
            terminal_pair,
            unfold_read_paths_accordingly=False,
            check_leakage=True):
        """
        :param input_path: rep_candidate without anchors
        :param terminal_pair:
        :param unfold_read_paths_accordingly:
        :param check_leakage:
        :return:
        TODO: check palindromic issue
        TODO: check all attributes in Assembly.__init__() when duplication happened
        """
        logger.debug("unfolding {} -> {} -> {}".format(terminal_pair[0], input_path, terminal_pair[1]))
        renamed_vertices = set()
        v_counts_in_path = {_v_name: 0 for _v_name, _v_end in input_path}
        for v_name, v_end in input_path:
            v_counts_in_path[v_name] += 1
        do_unfold = False
        for v_n in sorted(v_counts_in_path):
            if v_counts_in_path[v_n] > 1:
                renamed_vertices.add(v_n)
                do_unfold = True
        if not do_unfold:
            return renamed_vertices
        # basic checking
        if unfold_read_paths_accordingly:
            assert terminal_pair, "parameter terminal_pair is required for renaming read paths!"
        assert self.graph.contain_path(input_path), str(input_path) + " not in the graph!"
        if check_leakage and not self.graph.is_no_leaking_path(input_path, [terminal_pair]):
            logger.debug("Leaking path detected! Giving up unfolding!")
            return
        # generate renamed_path for following connection making
        renamed_path = []
        v_copy_id = {_v_name: 0 for _v_name, _v_end in input_path}
        for v_name, v_end in input_path:
            v_copy_id[v_name] += 1
            if v_copy_id[v_name] > 1:
                renamed_path.append(("{}__copy{}".format(v_name, v_copy_id[v_name]), v_end))
            else:
                renamed_path.append((v_name, v_end))
        overlap_of_connections = []
        for go_p, (v_name, v_end) in enumerate(input_path[:-1]):
            next_n, next_e = input_path[go_p + 1]
            overlap_of_connections.append(self.graph.vertex_info[v_name].connections[v_end][(next_n, not next_e)])
        # separate the coverage before splitting each vertex
        # TODO: initialize other coverage associated values
        for v_name in v_counts_in_path:
            self.graph.vertex_info[v_name].cov /= float(v_counts_in_path[v_name])
        # follow the path to unfold the vertices in the graph
        go_p = 0
        v_copy_id = {_v_name: 0 for _v_name, _v_end in input_path}
        vertex_info_dict = {_v_name: deepcopy(self.graph.vertex_info[_v_name]) for _v_name in v_copy_id}
        while go_p < len(input_path):
            v_name, v_end = input_path[go_p]
            v_copy_id[v_name] += 1
            if v_copy_id[v_name] > 1:
                v_new_name = "{}__copy{}".format(v_name, v_copy_id[v_name])
                self.graph.vertex_info[v_new_name] = deepcopy(vertex_info_dict[v_name])
            else:
                v_new_name = v_name
            # up stream
            if go_p > 0:
                self.graph.vertex_info[v_new_name].connections[not v_end] = \
                    OrderedDict([(renamed_path[go_p - 1], overlap_of_connections[go_p - 1])])
            # down stream
            if go_p < len(input_path) - 1:
                # logger.debug("create connection for {}{}".format(v_new_name, v_end))
                next_v, next_e = renamed_path[go_p + 1]
                self.graph.vertex_info[v_new_name].connections[v_end] = \
                    OrderedDict([((next_v, not next_e), overlap_of_connections[go_p])])
            go_p += 1
        # clean the start
        # clean inner connections to the left of the start
        old_l_n, old_l_e = input_path[0]
        new_l_n, new_l_e = renamed_path[0]
        for left_anchor_n, left_anchor_e in sorted(self.graph.vertex_info[new_l_n].connections[not new_l_e]):
            if left_anchor_n in v_counts_in_path:
                foo = self.graph.vertex_info[new_l_n].connections[not new_l_e].pop((left_anchor_n, left_anchor_e), None)
        # rename the outer connections (from the anchors) to the left of the start
        if old_l_n != new_l_n:
            for (left_anchor_n, left_anchor_e), ovl in self.graph.vertex_info[new_l_n].connections[not new_l_e].items():
                foo = self.graph.vertex_info[left_anchor_n].connections[left_anchor_e].pop((old_l_n, not old_l_e), None)
                self.graph.vertex_info[left_anchor_n].connections[left_anchor_e][(new_l_n, not new_l_e)] = ovl
        # clean the end
        # clean the inner connections to the right of the end
        old_r_n, old_r_e = input_path[-1]
        new_r_n, new_r_e = renamed_path[-1]
        for right_anchor_n, right_anchor_e in sorted(self.graph.vertex_info[new_r_n].connections[new_r_e]):
            if right_anchor_n in v_counts_in_path:
                foo = self.graph.vertex_info[new_r_n].connections[new_r_e].pop((right_anchor_n, right_anchor_e), None)
        # rename the outer connections (from the anchors) to the right of the end
        if old_r_n != new_r_n:
            for (right_anchor_n, right_anchor_e), ovl in self.graph.vertex_info[new_r_n].connections[new_r_e].items():
                foo = self.graph.vertex_info[right_anchor_n].connections[right_anchor_e].pop((old_r_n, old_r_e), None)
                self.graph.vertex_info[right_anchor_n].connections[right_anchor_e][(new_r_n, new_r_e)] = ovl

        # rename read_paths
        if unfold_read_paths_accordingly \
                and tuple(input_path) in self.rep_p_to_read_paths \
                and tuple(input_path) != tuple(renamed_path):
            self.__rename_read_paths(input_path, renamed_path, terminal_pair)

        return renamed_vertices

    def split_the_repeats(
            self,
            the_repeat_path,
            n_groups,
            terminal_pair_group_list,
            weights=None,
            distribute_read_paths_accordingly=False):
        """
        :param the_repeat_path:
        :param n_groups:
        :param terminal_pair_group_list: the number of input groups must equals n_groups.
        :param weights: the number of input values must equals n_groups. Values will be normalized.
        :param distribute_read_paths_accordingly:
        :return:
        """
        # basic checking
        logger.debug("splitting {}".format(the_repeat_path))
        for grouped_pair in terminal_pair_group_list:
            logger.debug("          {}".format(grouped_pair))
        assert n_groups >= 1
        assert len(terminal_pair_group_list) == n_groups, \
            "length of terminal_pairs_dict MUST equal the separation number!"
        assert self.graph.contain_path(the_repeat_path), str(the_repeat_path) + " not in the graph!"
        # normalize weights
        # TODO the weights should not be simply determined from current split.
        #      Even in the no_leaking scenario, excluding outer connections from the terminal pairs is not workable,
        #      because the outer connections may have their outer connections and so on.
        #      Ideally they should be determined from read_paths went through each vertex after splitting and renaming
        #      Maybe, re-estimate them after a complete cleaning process
        if not weights:
            weights = [1. / n_groups]
        else:
            assert len(weights) == n_groups, "length of weights MUST equal the separation number!"
            if sum(weights) == 0:
                weights = [1. / n_groups] * n_groups
            else:
                sum_counts = float(sum(weights))
                weights = [this_count / sum_counts for this_count in weights]

        vertices = sorted(set([v_name for v_name, v_end in the_repeat_path]))
        duplicated_name_groups = self.graph.duplicate(vertices=vertices, num_dup=n_groups, depth_factors=weights)

        for go_group, grouped_pair in enumerate(terminal_pair_group_list):
            name_translator = duplicated_name_groups[go_group]
            valid_terminal_left = set([terminal_p[0] for terminal_p in grouped_pair])
            check_path_left_n, check_path_left_e = name_translator[the_repeat_path[0][0]], not the_repeat_path[0][1]
            for (next_n, next_e) in list(self.graph.vertex_info[check_path_left_n].connections[check_path_left_e]):
                if (next_n, next_e) not in valid_terminal_left:
                    # delete the connection
                    del self.graph.vertex_info[check_path_left_n].connections[check_path_left_e][(next_n, next_e)]
                    del self.graph.vertex_info[next_n].connections[next_e][(check_path_left_n, check_path_left_e)]
            valid_terminal_right = set([(terminal_p[1][0], not terminal_p[1][1]) for terminal_p in grouped_pair])
            check_path_right_n, check_path_right_e = name_translator[the_repeat_path[-1][0]], the_repeat_path[-1][1]
            for (next_n, next_e) in list(self.graph.vertex_info[check_path_right_n].connections[check_path_right_e]):
                if (next_n, next_e) not in valid_terminal_right:
                    # delete the connection
                    del self.graph.vertex_info[check_path_right_n].connections[check_path_right_e][(next_n, next_e)]
                    del self.graph.vertex_info[next_n].connections[next_e][(check_path_right_n, check_path_right_e)]
        for go_group, grouped_pair in enumerate(terminal_pair_group_list):
            if distribute_read_paths_accordingly \
                    and tuple(the_repeat_path) in self.rep_p_to_read_paths:
                name_translator = duplicated_name_groups[go_group]
                renamed_path = [(name_translator[v_n], v_e) for v_n, v_e in the_repeat_path]
                if tuple(the_repeat_path) != tuple(renamed_path):
                    for terminal_pair in grouped_pair:
                        self.__rename_read_paths(the_repeat_path, renamed_path, terminal_pair)

    def __rename_read_paths(self, inner_path, new_inner_path, terminal_pair):
        """
        assign the read paths to the original inner_path or the renamed path
        which can also be fulfilled by remapping the associated reads to the new graph
        :param inner_path:
        :param new_inner_path:
        :param terminal_pair:
        :return:
        """
        if tuple(inner_path) != tuple(new_inner_path):
            # rand_num = random.random()
            # logger.debug("            renaming {} -> {}".format(inner_path, new_inner_path))
            # Part 1: use rep_p_to_read_paths to rename those have inner_path and terminal_pair
            read_info = self.rep_p_to_read_paths[inner_path][terminal_pair]
            left_v_name = terminal_pair[0][0]
            right_v_name = terminal_pair[1][0]
            for go_p, read_id in enumerate(read_info["read_id"]):
                if read_id not in self.id_to_read_paths:
                    continue
                read_path = list(self.id_to_read_paths[read_id])
                pmer_pos = read_info["pmer_pos"][go_p]
                rep_strand = read_info["rep_strand"][go_p]
                if read_path == [('251439', False), ('235187', True), ('252021', False),
                                 ('235115', True),
                                 ('254127', False), ('254145', True), ('254127', False),
                                 ('254103', True),
                                 ('249729', False), ('252317', True), ('249729', False), ('252317', True),
                                 ('249729', False), ('252317', True), ('249729', False), ('252317', True),
                                 ('249729', False), ('252317', True), ('249729', False), ('252317', True),
                                 ('249729', False), ('254137', True)]:
                    print("check rep_strand 1", rep_strand)
                    print(read_path)
                elif read_path == [('254103', False),  # 178
                                   ('254127', False), ('254145', True), ('254127', False),
                                   ('235115', False),
                                   ('252021', True), ('235187', False), ('251439', True), ('254037', True),
                                   ('254133', False), ('252875', True), ('254121', False)]:
                    print("check rep_strand 2", rep_strand)
                    print(read_path)
                elif read_id == 178:
                    print("check rep_strand 2", rep_strand)
                    print(read_path)
                # if rand_num > 0.9:
                #     logger.debug("   renaming inner {}".format(new_inner_path))
                #     logger.debug("            p_id {}, p_strand {}".format(pmer_pos, rep_strand))
                if not rep_strand:
                    new_inner_path = self.graph.reverse_path(new_inner_path)
                # if rand_num > 0.9:
                #     logger.debug("   renaming inner {}".format(new_inner_path))
                # inner_path_id = pmer_pos + 1, because inner_path = pmer_path[1:-1]
                new_read_path = read_path[:pmer_pos + 1] + \
                                list(new_inner_path) + \
                                read_path[pmer_pos + 1 + len(inner_path):]  # len(inner_path) == len(new_inner_path)
                # logger.debug("            old {}".format(read_path))
                # logger.debug("            new {}".format(new_read_path))
                if read_path == [('251439', False), ('235187', True), ('252021', False), ('235115', True),
                                 ('254127', False), ('254145', True), ('254127', False), ('254103', True),
                                 ('249729', False), ('252317', True), ('249729', False), ('252317', True),
                                 ('249729', False), ('252317', True), ('249729', False), ('252317', True),
                                 ('249729', False), ('252317', True), ('249729', False), ('252317', True),
                                 ('249729', False), ('254137', True)]:
                    print(new_read_path)
                elif read_id == 178:
                    print(new_read_path)
                new_read_path = tuple(new_read_path)
                read_path = tuple(read_path)
                self.__update_read_path_indices(
                    read_id, read_path, new_read_path, from_v=pmer_pos + 1, to_v=pmer_pos + 1 + len(inner_path))
            # Part 2: use read_id to rename the left (without covering the complete terminal_pair)
            old_path_names = set()
            for v_name, v_end in inner_path:
                old_path_names.add(v_name)
            inner_associated_read_ids = set()
            for v_name in sorted(old_path_names):
                for read_id in self.v_name_to_read_paths[v_name]:
                    inner_associated_read_ids.add(read_id)
            left_associated_read_ids = set()
            for read_id in self.v_name_to_read_paths[left_v_name]:
                left_associated_read_ids.add(read_id)
            right_associated_read_ids = set()
            for read_id in self.v_name_to_read_paths[right_v_name]:
                right_associated_read_ids.add(read_id)
            # Part 2 Scenario 1
            # for debug: there should be rare intersect between above three id sets, because of the previous processing
            for read_id in inner_associated_read_ids & left_associated_read_ids & right_associated_read_ids:
                if self.graph.contain_path(self.id_to_read_paths[read_id]):
                    # if the path is already renamed
                    pass
                else:
                    logger.debug("unexpected(1): {} -> {} -> {}: {}: {}".format(
                        terminal_pair[0], inner_path, terminal_pair[1], read_id, self.id_to_read_paths[read_id]))
            # Part 2 Scenario 2
            # for those does not match current terminal
            self.__align_inner_v_and_rename(
                id_set=inner_associated_read_ids - left_associated_read_ids - right_associated_read_ids,
                inner_path=inner_path,
                new_inner_path=new_inner_path,
                terminal_pair=terminal_pair,
                old_path_names=old_path_names)
            # Part 2 Scenario 3
            self.__align_t_v_and_rename(
                id_set=inner_associated_read_ids & left_associated_read_ids,
                start_v_name=left_v_name,
                start_v_end=terminal_pair[0][1],
                is_from_left=True,
                inner_path=inner_path,
                new_inner_path=new_inner_path)
            # Part 2 Scenario 4
            self.__align_t_v_and_rename(
                id_set=inner_associated_read_ids & right_associated_read_ids,
                start_v_name=right_v_name,
                start_v_end=terminal_pair[1][1],
                is_from_left=False,
                inner_path=inner_path,
                new_inner_path=new_inner_path)
            # may need align for sequential repeat
            # only inner_path and partial terminal

    def __align_inner_v_and_rename(
            self,
            id_set,
            inner_path,
            new_inner_path,
            terminal_pair,
            old_path_names):
        # id_set=inner_associated_read_ids - left_associated_read_ids - right_associated_read_ids
        for read_id in id_set:
            for v_name, v_end in self.id_to_read_paths[read_id]:
                if v_name not in old_path_names:
                    # other path
                    break
            else:
                # if the original read_path only contain vertices of the inner_path
                # the length of inner_path should be longer thant the read_path
                len_inner = len(inner_path)
                len_read = len(self.id_to_read_paths[read_id])
                length_dif = len_inner - len_read
                if length_dif < 0:
                    self.__del_read_path(
                        read_id, report=True,
                        extra_report_info="unexpected(2): {} -> {} -> {}: {}".format(
                            terminal_pair[0], inner_path, terminal_pair[1], self.id_to_read_paths[read_id]))
                    continue
                # if the original read_path is a sequential repeat inside the inner_path
                # randomly assign the read_path to where it could be aligned
                aligned = False
                for r_strand in self.__shuffled([True, False]):
                    for try_go in self.__shuffled(list(range(length_dif + 1))):
                        if r_strand:
                            try_path = self.graph.reverse_path(inner_path)
                            try_new_path = self.graph.reverse_path(new_inner_path)
                        else:
                            try_path = inner_path
                            try_new_path = new_inner_path
                        if self.id_to_read_paths[read_id] == try_path[try_go: try_go + len_read]:
                            self.__update_read_path_indices(
                                read_id, self.id_to_read_paths[read_id], tuple(try_new_path[try_go: try_go + len_read]))
                            aligned = True
                            break
                    if aligned:
                        break
                if not aligned:
                    if self.graph.contain_path(self.id_to_read_paths[read_id]):
                        pass
                    else:
                        self.__del_read_path(
                            read_id, report=True,
                            extra_report_info="unexpected(3): {} -> {} -> {}: {}".format(
                                terminal_pair[0], inner_path, terminal_pair[1], self.id_to_read_paths[read_id]))
                        continue

    def __align_t_v_and_rename(
            self,
            id_set,
            start_v_name,
            start_v_end,
            is_from_left,
            inner_path,
            new_inner_path):
        for read_id in id_set:
            aligned = False
            # skip deleted read_paths
            if read_id not in self.id_to_read_paths:
                continue
            read_path = self.id_to_read_paths[read_id]
            for try_go, try_e in self.__shuffled(list(self.v_name_to_read_paths[start_v_name][read_id].items())):
                new_read_path = list(read_path)
                if is_from_left:
                    if try_e == start_v_end:
                        to_change = read_path[try_go + 1:]
                        if to_change == inner_path[:len(to_change)]:
                            new_read_path[try_go + 1:] = new_inner_path[:len(to_change)]
                            self.__update_read_path_indices(read_id, read_path, tuple(new_read_path), from_v=try_go + 1)
                            aligned = True
                    else:
                        to_change = read_path[:try_go]
                        if to_change == self.graph.reverse_path(inner_path)[-len(to_change):]:
                            new_read_path[:try_go] = self.graph.reverse_path(new_inner_path)[-len(to_change):]
                            self.__update_read_path_indices(read_id, read_path, tuple(new_read_path), to_v=try_go)
                            aligned = True
                else:
                    if try_e == start_v_end:
                        to_change = read_path[:try_go]
                        if to_change == inner_path[-len(to_change):]:
                            new_read_path[:try_go] = new_inner_path[-len(to_change):]
                            self.__update_read_path_indices(read_id, read_path, tuple(new_read_path), to_v=try_go)
                            aligned = True
                    else:
                        to_change = read_path[try_go + 1:]
                        if to_change == self.graph.reverse_path(inner_path)[:len(to_change)]:
                            new_read_path[try_go + 1:] = self.graph.reverse_path(new_inner_path)[:len(to_change)]
                            self.__update_read_path_indices(read_id, read_path, tuple(new_read_path), from_v=try_go + 1)
                            aligned = True
                if aligned:
                    break
            if not aligned:
                if self.graph.contain_path(read_path):
                    pass
                else:
                    logger.debug("unexpected(4): start_v_name={}, start_v_end={}, new_ip={}".format(
                        start_v_name, start_v_end, new_inner_path))
                    self.__del_read_path(
                        read_id, report=True,
                        extra_report_info="unexpected(4): {} -> {} -> {}: {}: {}".format(
                            ["*", (start_v_name, start_v_end)][is_from_left],
                            inner_path,
                            [(start_v_name, start_v_end), "*"][is_from_left],
                            read_id,
                            self.id_to_read_paths[read_id]))
                    # self.__add_to_trim_list(
                    #     read_id,
                    #     trim_range=,
                    #     report=True,
                    #     extra_report_info="unexpected(4): {} -> {} -> {}: {}: {}".format(
                    #         ["*", (start_v_name, start_v_end)][is_from_left],
                    #         inner_path,
                    #         [(start_v_name, start_v_end), "*"][is_from_left],
                    #         read_id,
                    #         self.id_to_read_paths[read_id]))

    def __update_read_path_indices(self, read_id, read_path, new_read_path, from_v=0, to_v=None):
        assert len(read_path) == len(new_read_path)
        if to_v is None:
            to_v = len(new_read_path)
        # update self.read_paths
        tmp = self.read_paths[read_path]
        del self.read_paths[read_path]
        self.read_paths[new_read_path] = tmp
        # update self.id_to_read_paths
        self.id_to_read_paths[read_id] = new_read_path
        # update self.v_name_to_read_paths
        for go_del, (del_n, del_e) in enumerate(read_path[from_v:to_v]):
            if read_id in self.v_name_to_read_paths[del_n]:
                self.v_name_to_read_paths[del_n][read_id].pop(from_v + go_del)
                if not self.v_name_to_read_paths[del_n][read_id]:
                    del self.v_name_to_read_paths[del_n][read_id]
        for go_add, (add_n, add_e) in enumerate(new_read_path[from_v:to_v]):
            if add_n not in self.v_name_to_read_paths:
                self.v_name_to_read_paths[add_n] = OrderedDict()
            if read_id not in self.v_name_to_read_paths[add_n]:
                self.v_name_to_read_paths[add_n][read_id] = OrderedDict()
            self.v_name_to_read_paths[add_n][read_id][from_v + go_add] = add_e

    def __add_to_cutting_list(self, read_id, trim_range, report=False, extra_report_info=""):
        if report:
            if extra_report_info:
                logger.debug(extra_report_info)
            logger.debug("delete read path ({}) : {} : {}".format(read_id, trim_range, self.id_to_read_paths[read_id]))
        if read_id not in self.__cutting_tmp:
            self.__cutting_tmp[read_id] = set(range(len(self.id_to_read_paths[read_id])))
        for different_phase_pos in range(*trim_range):
            self.__cutting_tmp[read_id].discard(different_phase_pos)

    def __execute_cutting_list(self, report=False):
        for read_id in sorted(self.__cutting_tmp):
            if not self.__cutting_tmp[read_id]:
                self.__del_read_path(read_id, report, extra_report_info="cutting")
            else:
                # generate remaining ranges
                phased_ranges = []
                _mark_interval_ = False
                max_pos = len(self.id_to_read_paths[read_id])
                for go_pos in range(max_pos):
                    if go_pos in self.__cutting_tmp[read_id]:
                        if _mark_interval_:
                            continue
                        else:
                            phased_ranges.append([go_pos])
                            _mark_interval_ = True
                    else:
                        if _mark_interval_:
                            phased_ranges[-1].append(go_pos)
                            _mark_interval_ = False
                        else:
                            continue
                if _mark_interval_:
                    phased_ranges[-1].append(max_pos)
                for p_range in phased_ranges[::-1]:
                    if p_range[1] == max_pos or p_range[0] == 0:
                        self.__split_read(read_id, p_range[0])
                    else:
                        self.__split_read(read_id, p_range[1])
                        self.__split_read(read_id, p_range[0])
        # self.rep_p_to_read_paths will be updated
        self.__index_read_path_mers()

    def __split_read(self, old_read_id, split_pos):
        if split_pos == 0:
            return
        # make a new read path
        new_read_id = max(self.id_to_read_paths) + 1
        new_read_path, new_strand = self.graph.get_standardized_path_with_strand(
            raw_path=self.id_to_read_paths[old_read_id][split_pos:], detect_circular=False)
        new_len = len(new_read_path)
        if not new_len:
            return
        self.id_to_read_paths[new_read_id] = new_read_path
        for go_change, (change_n, change_e) in enumerate(new_read_path):
            if new_strand:
                del self.v_name_to_read_paths[change_n][old_read_id][split_pos + go_change]
            else:
                del self.v_name_to_read_paths[change_n][old_read_id][split_pos + new_len - 1 - go_change]
            if not self.v_name_to_read_paths[change_n][old_read_id]:
                del self.v_name_to_read_paths[change_n][old_read_id]
            if new_read_id not in self.v_name_to_read_paths[change_n]:
                self.v_name_to_read_paths[change_n][new_read_id] = OrderedDict()
            self.v_name_to_read_paths[change_n][new_read_id][go_change] = change_e
        self.read_paths[new_read_path] = self.read_paths[self.id_to_read_paths[old_read_id]]
        # re-index the truncated original path
        truncated_path, truncated_strand = self.graph.get_standardized_path_with_strand(
            raw_path=self.id_to_read_paths[old_read_id][:split_pos], detect_circular=False)
        self.read_paths[truncated_path] = self.read_paths[self.id_to_read_paths[old_read_id]]
        self.id_to_read_paths[old_read_id] = truncated_path
        if not truncated_strand:
            for go_change, (change_n, change_e) in enumerate(truncated_path):
                self.v_name_to_read_paths[change_n][old_read_id][go_change] = change_e

    def __trim_read_path(self, read_id, this_range):
        read_path = self.id_to_read_paths[read_id]
        for go_del in range(*this_range):
            del_n, del_e = read_path[go_del]
            del self.v_name_to_read_paths[del_n][read_id][go_del]
            if not self.v_name_to_read_paths[del_n][read_id]:
                del self.v_name_to_read_paths[del_n][read_id]

    def __del_read_path(self, read_id, report=False, extra_report_info=""):
        """
        """
        if report:
            if extra_report_info:
                logger.debug(extra_report_info)
            logger.debug("delete read path ({}) {}".format(read_id, self.id_to_read_paths[read_id]))
        read_path = self.id_to_read_paths[read_id]
        for go_del, (del_n, del_e) in enumerate(read_path):
            self.v_name_to_read_paths[del_n].pop(read_id, None)
        del self.read_paths[read_path]
        del self.id_to_read_paths[read_id]


