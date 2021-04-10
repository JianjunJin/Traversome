
from loguru import logger
from ifragaria.utils import generate_clusters_from_connections
from itertools import combinations
from collections import OrderedDict
from math import log


class CleanGraph(object):
    def __init__(self, ifragaria):
        self.graph = ifragaria.graph
        self.max_read_path_size = ifragaria.max_read_path_size
        self.read_paths = ifragaria.read_paths


    def run(self, min_effective_count=10, ignore_ratio=0.001):
        """
        use long reads to prune nonexistent paths from the graph
        :param min_effective_count: INT,
        a solution will not be made unless all supposed new paths are supported by this number of reads
        :param ignore_ratio: FLOAT,
        any conflicting minor path that supported by this ratio of reads will be ignored during solution
        :return:
        TODO prune crossing (size==2)
        TODO add new path
        """

        # # index read_paths by sizes
        # sizes_to_read_paths = {this_size: [] for this_size in range(3, self.max_read_path_size + 1)}
        # for read_path in self.read_paths:
        #     sizes_to_read_paths[len(read_path)].append(read_path)

        inner_to_read_paths = OrderedDict()
        branching_ends = self.graph.get_branching_ends()
        for path_size in (2, self.max_read_path_size + 1):
            for go_r, read_path in enumerate(self.read_paths):
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
                    if inner_path not in inner_to_read_paths:
                        inner_to_read_paths[inner_path] = OrderedDict()
                    # record the terminal_pair
                    if keep_strand:
                        terminal_pair = (read_path_mer[0], read_path_mer[-1])
                    else:
                        left_name, left_end = read_path_mer[0]
                        right_name, right_end = read_path_mer[-1]
                        terminal_pair = ((right_name, not right_end), (left_name, not left_end))
                    if terminal_pair not in inner_to_read_paths:
                        inner_to_read_paths[inner_path][terminal_pair] = \
                            {"path_counts": 0, "read_id": [], "pmer_id": [], "pmer_strand": []}
                    inner_to_read_paths[inner_path][terminal_pair]["path_counts"] += path_counts
                    inner_to_read_paths[inner_path][terminal_pair]["read_id"].append([go_r])
                    inner_to_read_paths[inner_path][terminal_pair]["pmer_id"].append([go_mer])
                    inner_to_read_paths[inner_path][terminal_pair]["pmer_strand"].append([keep_strand])

        # stepwise looking for conflicting paths
        # {(inner_path, terminal_pair_group): True}
        candidate_solutions = OrderedDict()
        # {(inner_path, terminal_pair): True}
        discard_pair = OrderedDict()
        for inner_path, terminal_pairs_info in inner_to_read_paths.items():
            if len(terminal_pairs_info) == 1:
                if terminal_pairs_info[list(terminal_pairs_info)[0]]["path_counts"] >= min_effective_count:
                    candidate_solutions[(inner_path, tuple(terminal_pairs_info))] = True
                else:
                    discard_pair[(inner_path, list(terminal_pairs_info)[0])] = True
            else:
                # use conflicting_terminals to record the conflicting terminal_pairs as edges
                conflicting_terminals = OrderedDict([(_term_p, []) for _term_p in terminal_pairs_info])
                for terminal_pair_1, terminal_pair_2 in combinations(terminal_pairs_info, 2):
                    # if the left terminal matches but the right terminal unmatches, it create a conflict, vice versa.
                    if (terminal_pair_1[0] == terminal_pair_2[0]) != (terminal_pair_1[1] == terminal_pair_2[1]):
                        this_ratio = inner_to_read_paths[inner_path][terminal_pair_1]["path_counts"] / \
                                     inner_to_read_paths[inner_path][terminal_pair_2]["path_counts"]
                        # if one terminal pair does NOT have significantly higher support over another,
                        # record the conflict
                        if ignore_ratio == 0 or abs(log(this_ratio)) < abs(log(ignore_ratio)):
                            # if that terminal pair was not removed from conflicting_terminals due to low support
                            if terminal_pair_1 in conflicting_terminals:
                                conflicting_terminals[terminal_pair_1].append(terminal_pair_2)
                            if terminal_pair_2 in conflicting_terminals:
                                conflicting_terminals[terminal_pair_2].append(terminal_pair_1)
                        elif this_ratio > 1:
                            discard_pair[(inner_path, terminal_pair_2)] = True
                            if terminal_pair_2 in conflicting_terminals:
                                del conflicting_terminals[terminal_pair_2]
                        else:
                            discard_pair[(inner_path, terminal_pair_1)] = True
                            if terminal_pair_1 in conflicting_terminals:
                                del conflicting_terminals[terminal_pair_1]
                # group terminal_pairs by conflicting networks
                terminal_clusters = generate_clusters_from_connections(terminal_pairs_info, conflicting_terminals)
                # if there was only one group
                if len(terminal_clusters) == 1:
                    # if that group has only one terminal_pair
                    if len(terminal_clusters[0]) == 1:
                        if terminal_pairs_info[list(terminal_clusters[0])[0]]["path_counts"] >= min_effective_count:
                            candidate_solutions[(inner_path, tuple(terminal_clusters[0]))] = True
                        else:
                            discard_pair[(inner_path, list(terminal_clusters[0])[0])] = True
                    else:
                        for terminal_pair in terminal_clusters[0]:
                            discard_pair[(inner_path, terminal_pair)] = True
                # if there were multiple groups:
                # terminal_pairs conflict with each other within each group,
                # but compatible with all terminal_pairs in another group
                else:
                    for terminal_cl in terminal_clusters:
                        candidate_solutions[(inner_path, tuple(sorted(terminal_cl)))] = True

        # index terminal_pairs
        terminal_pair_to_inner = OrderedDict()
        for inner_path in inner_to_read_paths:
            for (left_name, left_end), (right_name, right_end) in inner_to_read_paths[inner_path]:
                left_item = (left_name, left_end, "as_left")
                right_item = (right_name, right_end, "as_right")
                if left_item not in terminal_pair_to_inner:
                    terminal_pair_to_inner[left_item] = OrderedDict()
                if right_item not in terminal_pair_to_inner[left_item]:
                    terminal_pair_to_inner[left_item][right_item] = OrderedDict()
                terminal_pair_to_inner[left_item][right_item][inner_path] = 0
                if right_item not in terminal_pair_to_inner:
                    terminal_pair_to_inner[right_item] = OrderedDict()
                if left_item not in terminal_pair_to_inner[right_item]:
                    terminal_pair_to_inner[right_item][left_item] = OrderedDict()
                terminal_pair_to_inner[right_item][left_item][inner_path] = 0

        # solving repeats
        for inner_path, terminal_pairs in candidate_solutions: