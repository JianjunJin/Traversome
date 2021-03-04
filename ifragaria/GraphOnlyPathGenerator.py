
from loguru import logger
from ifragaria.utils import ProcessingGraphFailed
from copy import deepcopy
from itertools import combinations



class GraphOnlyPathGenerator(object):
    """
    All isomers are defined to have the same contig composition.
    """

    def __init__(self, graph, mode="embplant_pt", re_estimate_multiplicity=False):
        self.graph = graph
        self.mode = mode
        self.__re_estimate_multiplicity = re_estimate_multiplicity
        self.isomers = []

        # temporary values
        self.components = list()
        self.components_set = set()
        self.__start_vertex = None
        self.__start_direction = None


    def get_all_circular_isomers(self):
        self.components = list()
        self.components_set = set()
        if self.__re_estimate_multiplicity or not self.graph.vertex_to_copy:
            self.graph.estimate_multiplicity_by_cov(mode=self.mode)
            self.graph.estimate_multiplicity_precisely()

        # for palindromic repeats
        if self.graph.detect_palindromic_repeats(redo=False):
            logger.debug("Palindromic repeats detected. "
                         "Different paths generating identical sequence will be merged.")

        #
        self.graph.update_orf_total_len()
        if 1 not in self.graph.copy_to_vertex:
            do_check_all_start_kinds = True
            self.__start_vertex = sorted(self.graph.vertex_info,
                                         key=lambda x: (-self.graph.vertex_info[x].len,
                                                 -max(self.graph.vertex_info[x].other_attr["orf"][True]["sum_len"],
                                                      self.graph.vertex_info[x].other_attr["orf"][False]["sum_len"]),
                                                 x))[0]
            self.__start_direction = True
        else:
            # start from a single copy vertex, no need to check all kinds of start vertex
            do_check_all_start_kinds = False
            self.__start_vertex = sorted(self.graph.copy_to_vertex[1])[0]
            self.__start_direction = True

        # each contig stored format:
        first_path = [(self.__start_vertex, self.__start_direction)]
        first_connections = sorted(self.graph.vertex_info[self.__start_vertex].connections[self.__start_direction])
        vertex_to_copy = deepcopy(self.graph.vertex_to_copy)
        vertex_to_copy[self.__start_vertex] -= 1
        if vertex_to_copy[self.__start_vertex] <= 0:
            del vertex_to_copy[self.__start_vertex]
        self.__circular_directed_graph_solver(first_path, first_connections, vertex_to_copy, do_check_all_start_kinds,
                                              self.graph.palindromic_repeats)

        if not self.components:
            raise ProcessingGraphFailed("Detecting path(s) from remaining graph failed!")
        else:
            # modify start_vertex based on the whole path, if starting from a single copy vertex
            def reseed_a_path(input_path, input_unique_vertex):
                if input_unique_vertex not in input_path:
                    new_path = self.graph.reverse_path(input_path)
                else:
                    new_path = input_path
                # logger.debug(new_path, input_unique_vertex)
                reseed_from = new_path.index(input_unique_vertex)
                return new_path[reseed_from:] + new_path[:reseed_from]

            if 1 in self.graph.copy_to_vertex:
                branching_single_copy_vertices = set()
                if self.mode == "embplant_pt" and 2 in self.graph.copy_to_vertex:
                    # find branching points
                    for candidate_name in self.graph.copy_to_vertex[2]:
                        if not bool(self.graph.is_sequential_repeat(candidate_name)):
                            for neighboring_vertices in self.graph.vertex_info[candidate_name].connections.values():
                                if len(neighboring_vertices) == 2:
                                    (left_v, left_e), (right_v, right_e) = sorted(neighboring_vertices)
                                    if left_v in self.graph.copy_to_vertex[1] and right_v in self.graph.copy_to_vertex[1]:
                                        branching_single_copy_vertices.add(((left_v, not left_e), (right_v, right_e)))
                                        branching_single_copy_vertices.add(((right_v, not right_e), (left_v, left_e)))
                if branching_single_copy_vertices:
                    # more orfs found in the reverse direction of LSC of a typical plastome
                    # different paths may have different LSC
                    # picking the sub-path with the longest length with strand of least orfs as the new start point
                    branching_single_copy_vertices = sorted(branching_single_copy_vertices)
                    for go_p, each_path in enumerate(self.components):
                        reverse_path = self.graph.reverse_path(each_path)
                        sub_paths_for_checking = []
                        for (left_v, left_e), (right_v, right_e) in branching_single_copy_vertices:
                            if (left_v, left_e) in each_path:
                                if (right_v, right_e) in each_path:
                                    left_id = each_path.index((left_v, left_e))
                                    right_id = each_path.index((right_v, right_e))
                                    if left_id <= right_id:
                                        sub_paths_for_checking.append(each_path[left_id: right_id + 1])
                                    else:
                                        sub_paths_for_checking.append(each_path[left_id:] + each_path[:right_id + 1])
                                else:
                                    sub_paths_for_checking.append([])
                            else:
                                if (right_v, right_e) in reverse_path:
                                    left_id = reverse_path.index((left_v, left_e))
                                    right_id = reverse_path.index((right_v, right_e))
                                    if left_id <= right_id:
                                        sub_paths_for_checking.append(reverse_path[left_id: right_id + 1])
                                    else:
                                        sub_paths_for_checking.append(
                                            reverse_path[left_id:] + reverse_path[:right_id + 1])
                                else:
                                    sub_paths_for_checking.append([])
                        # picking the vertex with the longest length with strand of least orfs
                        lsc_pair_id = sorted(range(len(sub_paths_for_checking)),
                                             key=lambda x:
                                             (-sum([self.graph.vertex_info[sub_v].len
                                                    for sub_v, sub_e in sub_paths_for_checking[x]]) +
                                              self.graph.overlap() * (len(sub_paths_for_checking) - 1),
                                              sum([self.graph.vertex_info[sub_v].other_attr["orf"][sub_e]["sum_len"]
                                                   for sub_v, sub_e in sub_paths_for_checking[x]]),
                                              x))[0]
                        self.components[go_p] = reseed_a_path(each_path, branching_single_copy_vertices[lsc_pair_id][0])
                else:
                    candidate_single_copy_vertices = set()
                    for single_v in self.graph.copy_to_vertex[1]:
                        candidate_single_copy_vertices.add((single_v, True))
                        candidate_single_copy_vertices.add((single_v, False))
                    if self.mode == "embplant_pt":
                        # more orfs found in the reverse direction of LSC of a typical plastome
                        # picking the vertex with the longest length with strand of least orfs
                        self.__start_vertex, self.__start_direction = \
                            sorted(candidate_single_copy_vertices,
                                   key=lambda x: (-self.graph.vertex_info[x[0]].len,
                                                  self.graph.vertex_info[x[0]].other_attr["orf"][x[1]]["sum_len"],x))[0]
                        # if self.__reverse_start_direction_for_pt:
                        #     self.__start_direction = not self.__start_direction
                    else:
                        # picking the vertex with the longest length with strand of most orfs
                        self.__start_vertex, self.__start_direction = \
                            sorted(candidate_single_copy_vertices,
                                   key=lambda x: (-self.graph.vertex_info[x[0]].len,
                                                  -self.graph.vertex_info[x[0]].other_attr["orf"][x[1]]["sum_len"], x))[0]
                    for go_p, each_path in enumerate(self.components):
                        self.components[go_p] = reseed_a_path(each_path, (self.__start_vertex, self.__start_direction))
            #
            # return self.isomers


    def get_all_isomers(self):
        self.components = list()
        self.components_set = set()
        if self.__re_estimate_multiplicity or not self.graph.vertex_to_copy:
            self.graph.estimate_multiplicity_by_cov(mode=self.mode)
            self.graph.estimate_multiplicity_precisely()

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
        # all_start_v_e.sort(key=lambda x: (smart_trans_for_sort(x[0]), x[1]))
        all_start_v_e.sort()
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

        if not self.components:
            raise ProcessingGraphFailed("Detecting path(s) from remaining graph failed!")


    def __directed_graph_solver(
            self, ongoing_paths, next_connections, vertices_left, in_all_start_ve):
        if not vertices_left:
            new_paths, new_standardized = self.graph.get_standardized_isomer(ongoing_paths)
            if new_standardized not in self.components_set:
                self.components.append(new_paths)
                self.components_set.add(new_standardized)
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
                    if new_standardized not in self.components_set:
                        self.components.append(new_paths)
                        self.components_set.add(new_standardized)
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
                        if new_standardized not in self.components_set:
                            self.components.append(new_paths)
                            self.components_set.add(new_standardized)
                    else:
                        if self.mode == "embplant_pt" and len(new_connections) == 2 and new_connections[0][0] == \
                                new_connections[1][0]:
                            new_connections.sort(
                                key=lambda x: self.graph.vertex_info[x[0]].other_attr["orf"][x[1]]["sum_len"])
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
                if standardized_path not in self.components_set:
                    self.components_set.add(standardized_path)
                    self.components.append(standardized_path)
            else:
                new_path = tuple(new_path)
                if new_path not in self.components_set:
                    self.components_set.add(new_path)
                    self.components.append(new_path)
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
                            if standardized_path not in self.components_set:
                                self.components_set.add(standardized_path)
                                self.components.append(standardized_path)
                        else:
                            new_path = tuple(new_path)
                            if new_path not in self.components_set:
                                self.components_set.add(new_path)
                                self.components.append(new_path)
                        return
                    else:
                        return
                else:
                    new_connections = sorted(new_connections)
                    # if next_connections is SSC, reorder
                    if self.mode == "embplant_pt" and len(new_connections) == 2 and new_connections[0][0] == \
                            new_connections[1][0]:
                        new_connections.sort(
                            key=lambda x: -self.graph.vertex_info[x[0]].other_attr["orf"][x[1]]["sum_len"])
                    self.__circular_directed_graph_solver(new_path, new_connections, new_left, check_all_kinds,
                                                          palindromic_repeat_vertices)


