
from loguru import logger
from ifragaria.utils import generate_clusters_from_connections
from itertools import combinations
from collections import OrderedDict
from copy import deepcopy
from math import log


class CleanGraph(object):
    def __init__(self, ifragaria):
        self.graph = ifragaria.graph
        self.max_read_path_size = ifragaria.max_read_path_size
        self.read_paths = ifragaria.read_paths
        # to be generated
        self.__id_to_read_paths = OrderedDict()
        self.inner_to_read_paths = OrderedDict()     # {inner_path: {terminal_pair:
                                                     #               {"path_counts": int,
                                                     #                "read_id": [int],
                                                     #                "pmer_id": [int],
                                                     #                "pmer_strand": [bool]}}}
        self.__solutions = OrderedDict()     # {inner_path: {terminal_pair_group: True}}
        self.__solved = set()
        # self.terminal_pair_to_inner = OrderedDict()  # {(left_name, left_end, "as_left"):
        #                                              #  {(right_name, right_end, "as_right"): {inner_path: True}}}

    def run(self, min_effective_count=10, ignore_ratio=0.001):
        """
        use long reads to prune nonexistent paths from the graph
        :param min_effective_count: INT,
        a solution will not be made unless all supposed new paths are supported by this number of reads
        :param ignore_ratio: FLOAT,
        any conflicting minor path that supported by this ratio of reads will be ignored during solution
        :return:
        TODO add new path
        """
        finished = False
        while not finished:
            self.__index_read_path_mers()
            self.__generating_candidate_solutions(min_effective_count, ignore_ratio)
            # self.index_terminal_pairs()
            finished = self.__solve_repeats()

    def __index_read_paths(self):
        # for reducing memory cost in self.inner_to_read_paths
        for go_r, read_path in enumerate(self.read_paths):
            self.__id_to_read_paths[go_r] = read_path

    def __index_read_path_mers(self):
        # index read_path_mers by inner_path, with depth counted
        branching_ends = self.graph.get_branching_ends()
        for path_size in (2, self.max_read_path_size + 1):
            for go_r in self.__id_to_read_paths:
                read_path = self.__id_to_read_paths[go_r]
                this_r_p_size = len(read_path)
                path_counts = len(self.read_paths[read_path])
                # skip read paths that does not have branching sites inside
                # with underlying limit: this_r_p_size >= 2
                # 1. check forward
                for (v_name, v_end) in read_path[:-1]:
                    if (v_name, v_end) in branching_ends:
                        break
                else:
                    continue
                # 2. check reverse
                for (v_name, v_end) in read_path[1:]:
                    if (v_name, not v_end) in branching_ends:
                        break
                else:
                    continue
                # if this_r_p_size > 2:
                # chopping read_paths to read_path_mers (like kmers)
                # index read_path_mers by inner_path, with depth counted
                for go_mer in range(this_r_p_size - 2):
                    read_path_mer = self.graph.get_standardized_path(read_path[go_mer: go_mer + path_size])
                    inner_path, keep_strand = self.graph.get_standardized_path_with_strand(
                        raw_path=read_path_mer[1:-1],
                        detect_circular=False)
                    if inner_path not in self.inner_to_read_paths:
                        self.inner_to_read_paths[inner_path] = OrderedDict()
                    # record the terminal_pair
                    if keep_strand:
                        terminal_pair = (read_path_mer[0], read_path_mer[-1])
                    else:
                        terminal_pair = self.graph.reverse_path((read_path_mer[0], read_path_mer[-1]))
                    if terminal_pair not in self.inner_to_read_paths:
                        self.inner_to_read_paths[inner_path][terminal_pair] = \
                            {"path_counts": 0, "read_id": [], "pmer_id": [], "pmer_strand": []}
                    self.inner_to_read_paths[inner_path][terminal_pair]["path_counts"] += path_counts
                    self.inner_to_read_paths[inner_path][terminal_pair]["read_id"].append(go_r)
                    self.inner_to_read_paths[inner_path][terminal_pair]["pmer_id"].append(go_mer)
                    self.inner_to_read_paths[inner_path][terminal_pair]["pmer_strand"].append(keep_strand)

    def __generating_candidate_solutions(self, min_effective_count, ignore_ratio):
        # discard_pair = OrderedDict() # {(inner_path, terminal_pair): True}
        # do not know where to use discard_pair yet, maybe delete it
        # stepwise looking for conflicting paths
        # initialize
        self.__solutions = OrderedDict()
        for inner_path, terminal_pairs_info in self.inner_to_read_paths.items():
            # skip solved inner_paths
            if inner_path in self.__solved:
                continue
            self.__solutions[inner_path] = OrderedDict()
            if len(terminal_pairs_info) == 1:
                if terminal_pairs_info[list(terminal_pairs_info)[0]]["path_counts"] >= min_effective_count:
                    self.__solutions[inner_path][tuple(terminal_pairs_info)] = True
                # else:
                #     discard_pair[(inner_path, list(terminal_pairs_info)[0])] = True
            else:
                # use conflicting_terminals to record the conflicting terminal_pairs as edges
                conflicting_terminals = OrderedDict([(_term_p, []) for _term_p in terminal_pairs_info])
                for terminal_pair_1, terminal_pair_2 in combinations(terminal_pairs_info, 2):
                    # if the left terminal matches but the right terminal unmatches, it create a conflict, vice versa.
                    if (terminal_pair_1[0] == terminal_pair_2[0]) != (terminal_pair_1[1] == terminal_pair_2[1]):
                        this_ratio = self.inner_to_read_paths[inner_path][terminal_pair_1]["path_counts"] / \
                                     self.inner_to_read_paths[inner_path][terminal_pair_2]["path_counts"]
                        # if one terminal pair does NOT have significantly higher support over another,
                        # record the conflict
                        if ignore_ratio == 0 or abs(log(this_ratio)) < abs(log(ignore_ratio)):
                            # if that terminal pair was not removed from conflicting_terminals due to low support
                            if terminal_pair_1 in conflicting_terminals:
                                conflicting_terminals[terminal_pair_1].append(terminal_pair_2)
                            if terminal_pair_2 in conflicting_terminals:
                                conflicting_terminals[terminal_pair_2].append(terminal_pair_1)
                        elif this_ratio > 1:
                            # discard_pair[(inner_path, terminal_pair_2)] = True
                            if terminal_pair_2 in conflicting_terminals:
                                del conflicting_terminals[terminal_pair_2]
                        else:
                            # discard_pair[(inner_path, terminal_pair_1)] = True
                            if terminal_pair_1 in conflicting_terminals:
                                del conflicting_terminals[terminal_pair_1]
                # group terminal_pairs by conflicting networks
                terminal_clusters = generate_clusters_from_connections(terminal_pairs_info, conflicting_terminals)
                # if there was only one group
                if len(terminal_clusters) == 1:
                    # if that group has only one terminal_pair
                    if len(terminal_clusters[0]) == 1:
                        if terminal_pairs_info[list(terminal_clusters[0])[0]]["path_counts"] >= min_effective_count:
                            self.__solutions[inner_path][tuple(terminal_clusters[0])] = True
                    #     else:
                    #         discard_pair[(inner_path, list(terminal_clusters[0])[0])] = True
                    # else:
                    #     for terminal_pair in terminal_clusters[0]:
                    #         discard_pair[(inner_path, terminal_pair)] = True
                # if there were multiple groups:
                # terminal_pairs conflict with each other within each group,
                # but compatible with all terminal_pairs in another group
                else:
                    for terminal_cl in terminal_clusters:
                        self.__solutions[inner_path][tuple(sorted(terminal_cl))] = True

    # def index_terminal_pairs(self):
    #     # index terminal_pairs
    #     for inner_path in self.inner_to_read_paths:
    #         for (left_name, left_end), (right_name, right_end) in self.inner_to_read_paths[inner_path]:
    #             left_item = (left_name, left_end, "as_left")
    #             right_item = (right_name, right_end, "as_right")
    #             if left_item not in self.terminal_pair_to_inner:
    #                 self.terminal_pair_to_inner[left_item] = OrderedDict()
    #             if right_item not in self.terminal_pair_to_inner[left_item]:
    #                 self.terminal_pair_to_inner[left_item][right_item] = OrderedDict()
    #             self.terminal_pair_to_inner[left_item][right_item][inner_path] = True
    #             if right_item not in self.terminal_pair_to_inner:
    #                 self.terminal_pair_to_inner[right_item] = OrderedDict()
    #             if left_item not in self.terminal_pair_to_inner[right_item]:
    #                 self.terminal_pair_to_inner[right_item][left_item] = OrderedDict()
    #             self.terminal_pair_to_inner[right_item][left_item][inner_path] = True

    def __solve_repeats(self):
        all_repeats_solved = True
        renamed_vertices = set()
        for inner_path, terminal_pairs_dict in sorted(self.__solutions.items(), key=lambda x: (len(x[0]), x)):
            terminal_pairs_list = list(terminal_pairs_dict)
            if not inner_path:
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
                self.__solved.add(inner_path)
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
                for v_n, v_e in inner_path:
                    if v_n in renamed_vertices:
                        involved_in_vertex_duplication = True
                        break
                if involved_in_vertex_duplication:
                    all_repeats_solved = False
                    continue
                else:
                    self.__solved.add(inner_path)
                    #########
                    # prune unsupported connections to the inner_path (involved ends)
                    (i_lt_n, i_lt_e), (i_rt_n, i_rt_e) = inner_path[0], inner_path[-1]
                    keep_connections = set()
                    for grouped_pairs in terminal_pairs_list:
                        for (keep_lt_n, keep_lt_e), (keep_rt_n, keep_rt_e) in grouped_pairs:
                            keep_connections.add(((keep_lt_n, keep_lt_e), (i_lt_n, not i_lt_e)))
                            # keep_connections.add(((i_lt_n, not i_lt_e), (keep_lt_n, keep_lt_e)))
                            keep_connections.add(((i_rt_n, i_rt_e), (keep_rt_n, not keep_rt_e)))
                            # keep_connections.add(((keep_rt_n, not keep_rt_e), (i_rt_n, i_rt_e)))
                    for (n_1, e_1) in self.graph.vertex_info[i_lt_n].connections[not i_lt_e]:
                        if ((n_1, e_1), (i_lt_n, not i_lt_e)) not in keep_connections:
                            self.graph.vertex_info[n_1].connections[e_1].pop((i_lt_n, not i_lt_e), None)
                            self.graph.vertex_info[i_lt_n].connections[not i_lt_e].pop((n_1, e_1), None)
                    for (n_2, e_2) in self.graph.vertex_info[i_rt_n].connections[i_rt_e]:
                        if ((i_rt_n, i_rt_e), (n_2, not e_2)) not in keep_connections:
                            self.graph.vertex_info[n_2].connections[e_2].pop((i_rt_n, i_rt_e), None)
                            self.graph.vertex_info[i_rt_n].connections[i_rt_e].pop((n_2, e_2), None)
                    # duplicate inner_path and separate them into different paths
                    if len(terminal_pairs_list) == 1:
                        if len(list(terminal_pairs_list)[0]) == 1:
                            this_terminal = list(terminal_pairs_list)[0].pop()
                            if not self.graph.is_no_leaking_path(path=inner_path, terminal_pair=this_terminal):
                                # ironing over a bubble if there are sequential repeats
                                self.unfold_graph_along_path(inner_path,
                                                             unfold_read_paths_accordingly=True,
                                                             terminal_pair=this_terminal,
                                                             check_leakage=False)
                                # record renamed vertices
                                v_counts_in_path = {_v_name: 0 for _v_name, _v_end in inner_path}
                                for v_name, v_end in inner_path:
                                    v_counts_in_path[v_name] += 1
                                for v_n in sorted(v_counts_in_path):
                                    if v_counts_in_path[v_n] > 1:
                                        renamed_vertices.add(v_n)
                            else:
                                pass
                        else:
                            pass
                    else:
                        # calculate the depth ratios
                        group_counts = []
                        for grouped_pairs in terminal_pairs_list:
                            group_counts.append(0)
                            for terminal_pair in grouped_pairs:
                                group_counts[-1] += self.inner_to_read_paths[inner_path][terminal_pair]["path_counts"]
                        # multiplicate the inner_path
                        self.split_the_repeats(inner_path,
                                               n_groups=len(group_counts),
                                               terminal_pair_group_list=terminal_pairs_list,
                                               weights=group_counts,
                                               distribute_read_paths_accordingly=True)
                        for v_n, v_e in inner_path:
                            renamed_vertices.add(v_n)
        return all_repeats_solved

    def unfold_graph_along_path(
            self,
            input_path,
            unfold_read_paths_accordingly=False,
            terminal_pair=tuple(),
            check_leakage=True):
        """
        :param input_path: inner_path without anchors
        :param unfold_read_paths_accordingly:
        :param terminal_pair:
        :param check_leakage:
        :return:
        TODO: check palindromic issue
        TODO: check all attributes in Assembly.__init__() when duplication happened
        """
        # basic checking
        if unfold_read_paths_accordingly:
            assert terminal_pair, "parameter terminal_pair is required for renaming read paths!"
        assert self.graph.contain_path(input_path), str(input_path) + " not in the graph!"
        if check_leakage and not self.graph.is_no_leaking_path(input_path, terminal_pair):
            logger.debug("Leaking path detected! Giving up unfolding!")
            return
        v_counts_in_path = {_v_name: 0 for _v_name, _v_end in input_path}
        for v_name, v_end in input_path:
            v_counts_in_path[v_name] += 1
        # generate renamed_path for following connection making
        renamed_path = []
        v_copy_id = {_v_name: 0 for _v_name, _v_end in input_path}
        for v_name, v_end in input_path:
            v_copy_id[v_name] += 1
            if v_copy_id[v_name] > 1:
                renamed_path.append(("{}__copy{}".format(v_name, v_copy_id[v_name]), v_end))
            else:
                renamed_path.append((v_name, v_end))
        overlap_vals = []
        for go_p, (v_name, v_end) in enumerate(input_path[:-1]):
            overlap_vals.append(self.graph.vertex_info[v_name].connections[v_end][input_path[go_p + 1]])
        # separate the coverage before splitting each vertex
        # TODO: initialize other coverage associated values
        for v_name, v_end in input_path:
            self.graph.vertex_info[v_name].cov /= float(v_counts_in_path[v_name])
        # follow the path to unfold the vertices in the graph
        go_p = 0
        v_copy_id = {_v_name: 0 for _v_name, _v_end in input_path}
        while go_p < len(input_path):
            v_name, v_end = input_path[go_p]
            v_copy_id[v_name] += 1
            if v_copy_id[v_name] > 1:
                v_new_name = "{}__copy{}".format(v_name, v_copy_id[v_name])
                self.graph.vertex_info[v_new_name] = deepcopy(self.graph.vertex_info[v_name])
            else:
                v_new_name = v_name
            # up stream
            if go_p > 0:
                self.graph.vertex_info[v_new_name].connections[not v_end] = \
                    OrderedDict([(renamed_path[go_p - 1], overlap_vals[go_p - 1])])
            # down stream
            if go_p < len(input_path) - 1:
                next_v, next_e = renamed_path[go_p + 1]
                self.graph.vertex_info[v_new_name].connections[v_end] = \
                    OrderedDict([((next_v, not next_e), overlap_vals[go_p])])
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
        if unfold_read_paths_accordingly and input_path in self.inner_to_read_paths:
            self.__rename_read_paths(input_path, renamed_path, terminal_pair)

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
        assert n_groups >= 1
        assert len(terminal_pair_group_list) == n_groups, \
            "length of terminal_pairs_dict MUST equal the separation number!"
        assert self.graph.contain_path(the_repeat_path), str(the_repeat_path) + " not in the graph!"
        # normalize weights
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
            # if go_group == 0:  # modify the first group in the end
            #     continue
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

            if distribute_read_paths_accordingly and the_repeat_path in self.inner_to_read_paths:
                renamed_path = [(name_translator[v_n], v_e) for v_n, v_e in the_repeat_path]
                for terminal_pair in grouped_pair:
                    self.__rename_read_paths(the_repeat_path, renamed_path, terminal_pair)

    def __rename_read_paths(self, inner_path, new_inner_path, terminal_pair):
        read_info = self.inner_to_read_paths[inner_path][terminal_pair]
        for go_p, read_id in enumerate(read_info["read_id"]):
            read_path = list(self.__id_to_read_paths[read_id])
            pmer_id = read_info["pmer_id"]
            pmer_strand = read_info["pmer_strand"]
            if not pmer_strand:
                new_inner_path = self.graph.reverse_path(new_inner_path)
            read_path[pmer_id:pmer_id + len(inner_path)] = new_inner_path
            new_read_path = tuple(read_path)
            self.read_paths[new_read_path] = self.read_paths[read_path]
            del self.read_paths[read_path]
            self.__id_to_read_paths[read_id] = new_read_path



