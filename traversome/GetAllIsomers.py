
from loguru import logger
from traversome.utils import ProcessingGraphFailed #, smart_trans_for_sort
from copy import deepcopy
from itertools import combinations
import sys



class GetAllIsomers(object):
    """
    All isomers are defined to have the same contig composition.
    """
    def __init__(self, graph, mode="embplant_pt"):
        self.mode = mode
        self.graph = graph
        
        self.__paths = list()
        self.__paths_set = set()
        self.__start_vertex = None
        self.__start_direction = None


    def get_all_isomers(self):
        self.__paths = list()
        self.__paths_set = set()
        # start from a terminal vertex in an open graph/subgraph
        #         or a single copy vertex in a closed graph/subgraph
        self.graph.update_orf_total_len()

        # 2019-12-28 palindromic repeats
        if self.graph.detect_palindromic_repeats(redo=False):
            logger.warning("Palindromic repeats detected. "
                           "Different paths generating identical sequence will be merged.")

        all_start_v_e = []
        start_vertices = set()
        for go_set, v_set in enumerate(self.graph.vertex_clusters):
            is_closed = True
            for test_vertex_n in sorted(v_set):
                for test_end in (False, True):
                    if not self.graph.vertex_info[test_vertex_n].connections[test_end]:
                        is_closed = False
                        if test_vertex_n not in start_vertices:
                            all_start_v_e.append((test_vertex_n, not test_end))
                            start_vertices.add(test_vertex_n)
            if is_closed:
                if 1 in self.graph.copy_to_vertex[1] and bool(self.graph.copy_to_vertex[1] & v_set):
                    single_copy_v = \
                    sorted(self.graph.copy_to_vertex[1] & v_set, key=lambda x: -self.graph.vertex_info[x].len)[0]
                    all_start_v_e.append((single_copy_v, True))
                else:
                    longest_v = sorted(v_set, key=lambda x: -self.graph.vertex_info[x].len)[0]
                    all_start_v_e.append((longest_v, True))
        all_start_v_e.sort(key=lambda x: (smart_trans_for_sort(x[0]), x[1]))
        # start from a self-loop vertex in an open/closed graph/subgraph
        for go_set, v_set in enumerate(self.graph.vertex_clusters):
            for test_vertex_n in sorted(v_set):
                if self.graph.vertex_info[test_vertex_n].is_self_loop():
                    all_start_v_e.append((test_vertex_n, True))
                    all_start_v_e.append((test_vertex_n, False))

        start_v_e = all_start_v_e.pop(0)
        first_path = [[start_v_e]]
        first_connections = sorted(self.graph.vertex_info[start_v_e[0]].connections[start_v_e[1]])
        vertex_to_copy = deepcopy(self.graph.vertex_to_copy)
        vertex_to_copy[start_v_e[0]] -= 1
        if not vertex_to_copy[start_v_e[0]]:
            del vertex_to_copy[start_v_e[0]]
        self.__directed_graph_solver(first_path, first_connections, vertex_to_copy, all_start_v_e)

        # standardized_path_unique_set = set([this_path_pair[1] for this_path_pair in path_paris])
        # paths = []
        # for raw_path, standardized_path in path_paris:
        #     if standardized_path in standardized_path_unique_set:
        #         paths.append(raw_path)
        #         standardized_path_unique_set.remove(standardized_path)

        if not self.__paths:
            raise ProcessingGraphFailed("Detecting path(s) from remaining graph failed!")
        else:
            sorted_paths = []
            # total_len = len(list(set(paths))[0])
            record_pattern = False
            for original_id, this_path in enumerate(self.__paths):
                acc_dist = 0
                for copy_num in self.graph.copy_to_vertex:
                    if copy_num > 2:
                        for vertex_name in self.graph.copy_to_vertex[copy_num]:
                            for this_p_part in this_path:
                                loc_ids = [go_to_id for go_to_id, (v, e) in enumerate(this_p_part) if v == vertex_name]
                                if len(loc_ids) > 1:
                                    record_pattern = True
                                    if (this_p_part[0][0], not this_p_part[0][1]) \
                                            in self.graph.vertex_info[this_p_part[-1][0]].connections[
                                        this_p_part[-1][1]]:
                                        # circular
                                        part_len = len(this_p_part)
                                        for id_a, id_b in combinations(loc_ids, 2):
                                            acc_dist += min((id_a - id_b) % part_len, (id_b - id_a) % part_len)
                                    else:
                                        for id_a, id_b in combinations(loc_ids, 2):
                                            acc_dist += id_b - id_a
                sorted_paths.append((this_path, acc_dist, original_id))
            if record_pattern:
                sorted_paths.sort(key=lambda x: (-x[1], x[2]))
                pattern_dict = {acc_distance: ad_id + 1
                                for ad_id, acc_distance
                                in enumerate(sorted(set([x[1] for x in sorted_paths]), reverse=True))}
                if len(pattern_dict) > 1:
                    # if self.mode == "embplant_pt":
                    #     logger.warning("Multiple repeat patterns appeared in your data, "
                    #                    "a more balanced pattern (always the repeat_pattern1) would be "
                    #                    "suggested for plastomes with inverted repeats!")
                    # else:
                    #     logger.warning("Multiple repeat patterns appeared in your data.")
                    sorted_paths = [(this_path, ".repeat_pattern" + str(pattern_dict[acc_distance]))
                                    for this_path, acc_distance, foo_id in sorted_paths]
                else:
                    sorted_paths = [(this_path, "") for this_path in sorted(self.__paths)]
            else:
                sorted_paths = [(this_path, "") for this_path in sorted(self.__paths)]

            return sorted_paths


    def __directed_graph_solver(
            self, ongoing_paths, next_connections, vertices_left, in_all_start_ve):
        if not vertices_left:
            new_paths, new_standardized = self.graph.get_standardized_isomer(ongoing_paths)
            if new_standardized not in self.__paths_set:
                self.__paths.append(new_paths)
                self.__paths_set.add(new_standardized)
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
                    if new_standardized not in self.__paths_set:
                        self.__paths.append(new_paths)
                        self.__paths_set.add(new_standardized)
                    return
                else:
                    if self.mode == "embplant_pt" and len(new_connections) == 2 and new_connections[0][0] == \
                            new_connections[1][0]:
                        new_connections.sort(
                            key=lambda x: self.graph.vertex_info[x[0]].other_attr["orf"][x[1]]["sum_len"])
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
                        if new_standardized not in self.__paths_set:
                            self.__paths.append(new_paths)
                            self.__paths_set.add(new_standardized)
                    else:
                        if self.mode == "embplant_pt" and len(new_connections) == 2 and new_connections[0][0] == \
                                new_connections[1][0]:
                            new_connections.sort(
                                key=lambda x: self.graph.vertex_info[x[0]].other_attr["orf"][x[1]]["sum_len"])
                        self.__directed_graph_solver(new_paths, new_connections, new_left, new_all_start_ve)
                        break
            if not new_all_start_ve:
                return
