



from loguru import logger



class GetAllCircularPaths:

	def __init__(
		self, 
		mode="embplant_pt", 
		library_info=None, 
		reverse_start_direction_for_pt=False,
		):



    def circular_directed_graph_solver(
        ongoing_path, 
        next_connections, 
        vertices_left, 
        check_all_kinds,
        palindromic_repeat_vertices,
        ):
        # print("-----------------------------")
        # print("ongoing_path", ongoing_path)
        # print("next_connect", next_connections)
        # print("vertices_lef", vertices_left)

        if not vertices_left:
            new_path = deepcopy(ongoing_path)
            if palindromic_repeat_vertices:
                new_path = [(this_v, True) if this_v in palindromic_repeat_vertices else (this_v, this_e)
                            for this_v, this_e in new_path]
            if check_all_kinds:
                if palindromic_repeat_vertices:
                    rev_path = [(this_v, True) if this_v in palindromic_repeat_vertices else (this_v, not this_e)
                                for this_v, this_e in new_path[::-1]]
                else:
                    rev_path = [(this_v, not this_e) for this_v, this_e in new_path[::-1]]
                this_path_derived = [new_path, rev_path]
                for change_start in range(1, len(new_path)):
                    this_path_derived.append(new_path[change_start:] + new_path[:change_start])
                    this_path_derived.append(rev_path[change_start:] + rev_path[:change_start])
                standardized_path = tuple(sorted(this_path_derived)[0])
                if standardized_path not in paths_set:
                    paths_set.add(standardized_path)
                    paths.append(standardized_path)
            else:
                new_path = tuple(new_path)
                if new_path not in paths_set:
                    paths_set.add(new_path)
                    paths.append(new_path)
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
                new_connections = self.vertex_info[next_vertex].connections[not next_end]
                if not new_left:
                    if (start_vertex, not start_direction) in new_connections:
                        if palindromic_repeat_vertices:
                            new_path = [
                                (this_v, True) if this_v in palindromic_repeat_vertices else (this_v, this_e)
                                for this_v, this_e in new_path]
                        if check_all_kinds:
                            if palindromic_repeat_vertices:
                                rev_path = [(this_v, True) if this_v in palindromic_repeat_vertices else
                                            (this_v, not this_e)
                                            for this_v, this_e in new_path[::-1]]
                            else:
                                rev_path = [(this_v, not this_e) for this_v, this_e in new_path[::-1]]
                            this_path_derived = [new_path, rev_path]
                            for change_start in range(1, len(new_path)):
                                this_path_derived.append(new_path[change_start:] + new_path[:change_start])
                                this_path_derived.append(rev_path[change_start:] + rev_path[:change_start])
                            standardized_path = tuple(sorted(this_path_derived)[0])
                            if standardized_path not in paths_set:
                                paths_set.add(standardized_path)
                                paths.append(standardized_path)
                        else:
                            new_path = tuple(new_path)
                            if new_path not in paths_set:
                                paths_set.add(new_path)
                                paths.append(new_path)
                        return
                    else:
                        return
                else:
                    new_connections = sorted(new_connections)
                    # if next_connections is SSC, reorder
                    if mode == "embplant_pt" and len(new_connections) == 2 and new_connections[0][0] == \
                            new_connections[1][0]:
                        new_connections.sort(
                            key=lambda x: -self.vertex_info[x[0]].other_attr["orf"][x[1]]["sum_len"])
                    circular_directed_graph_solver(new_path, new_connections, new_left, check_all_kinds,
                                                   palindromic_repeat_vertices)

    # for palindromic repeats
    palindromic_repeats = set()
    log_palindrome = False
    for vertex_n in self.vertex_info:
        if self.vertex_info[vertex_n].seq[True] == self.vertex_info[vertex_n].seq[False]:
            forward_c = deepcopy(self.vertex_info[vertex_n].connections[True])
            reverse_c = deepcopy(self.vertex_info[vertex_n].connections[False])
            # This is heuristic
            # In the rarely-used expression way, a contig connect itself in one end:
            # (vertex_n, True) in forward_c or (vertex_n, False) in reverse_c
            if forward_c and \
                    ((forward_c == reverse_c) or
                     ((vertex_n, True) in forward_c) or
                     ((vertex_n, False) in reverse_c)):
                log_palindrome = True
                if len(forward_c) == len(reverse_c) == 2:  # simple palindromic repeats, prune repeated connections
                    for go_d, (nb_vertex, nb_direction) in enumerate(tuple(forward_c)):
                        del self.vertex_info[nb_vertex].connections[nb_direction][(vertex_n, bool(go_d))]
                        del self.vertex_info[vertex_n].connections[bool(go_d)][(nb_vertex, nb_direction)]
                elif len(forward_c) == len(reverse_c) == 1:  # connect to the same inverted repeat
                    pass
                else:  # complicated, recorded
                    palindromic_repeats.add(vertex_n)
    if log_palindrome:
        log_handler.info("Palindromic repeats detected. "
                         "Different paths generating identical sequence will be merged.")

    #
    self.update_orf_total_len()
    paths = []
    paths_set = set()
    if 1 not in self.copy_to_vertex:
        do_check_all_start_kinds = True
        start_vertex = sorted(self.vertex_info,
                              key=lambda x: (-self.vertex_info[x].len,
                                             -max(self.vertex_info[x].other_attr["orf"][True]["sum_len"],
                                                  self.vertex_info[x].other_attr["orf"][False]["sum_len"]),
                                             x))[0]
        start_direction = True
    else:
        # start from a single copy vertex, no need to check all kinds of start vertex
        do_check_all_start_kinds = False
        start_vertex = sorted(self.copy_to_vertex[1])[0]
        start_direction = True

    # each contig stored format:
    first_path = [(start_vertex, start_direction)]
    first_connections = sorted(self.vertex_info[start_vertex].connections[start_direction])
    vertex_to_copy = deepcopy(self.vertex_to_copy)
    vertex_to_copy[start_vertex] -= 1
    if vertex_to_copy[start_vertex] <= 0:
        del vertex_to_copy[start_vertex]
    circular_directed_graph_solver(first_path, first_connections, vertex_to_copy, do_check_all_start_kinds,
                                   palindromic_repeats)

    if not paths:
        raise ProcessingGraphFailed("Detecting path(s) from remaining graph failed!")
    else:
        

        # modify start_vertex based on the whole path, if starting from a single copy vertex
        def reseed_a_path(input_path, input_unique_vertex):
            if input_unique_vertex not in input_path:
                new_path = [(element_v, not element_e) for (element_v, element_e) in input_path[::-1]]
            else:
                new_path = input_path
            reseed_from = new_path.index(input_unique_vertex)
            return new_path[reseed_from:] + new_path[:reseed_from]
        


        if 1 in self.copy_to_vertex:
            branching_single_copy_vertices = set()
            if mode == "embplant_pt" and 2 in self.copy_to_vertex:
                # find branching points
                for candidate_name in self.copy_to_vertex[2]:
                    if not bool(self.is_sequential_repeat(candidate_name)):
                        for neighboring_vertices in self.vertex_info[candidate_name].connections.values():
                            if len(neighboring_vertices) == 2:
                                (left_v, left_e), (right_v, right_e) = sorted(neighboring_vertices)
                                if left_v in self.copy_to_vertex[1] and right_v in self.copy_to_vertex[1]:
                                    branching_single_copy_vertices.add(((left_v, not left_e), (right_v, right_e)))
                                    branching_single_copy_vertices.add(((right_v, not right_e), (left_v, left_e)))
            if branching_single_copy_vertices:
                # more orfs found in the reverse direction of LSC of a typical plastome
                # different paths may have different LSC
                # picking the sub-path with the longest length with strand of least orfs as the new start point
                branching_single_copy_vertices = sorted(branching_single_copy_vertices)
                for go_p, each_path in enumerate(paths):
                    reverse_path = [(element_v, not element_e) for (element_v, element_e) in each_path[::-1]]
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
                                         (-sum([self.vertex_info[sub_v].len
                                                for sub_v, sub_e in sub_paths_for_checking[x]]) +
                                              self.__overlap * (len(sub_paths_for_checking) - 1),
                                          sum([self.vertex_info[sub_v].other_attr["orf"][sub_e]["sum_len"]
                                               for sub_v, sub_e in sub_paths_for_checking[x]]),
                                          x))[0]
                    paths[go_p] = reseed_a_path(each_path, branching_single_copy_vertices[lsc_pair_id][0])
            else:
                candidate_single_copy_vertices = set()
                for single_v in self.copy_to_vertex[1]:
                    candidate_single_copy_vertices.add((single_v, True))
                    candidate_single_copy_vertices.add((single_v, False))
                if mode == "embplant_pt":
                    # more orfs found in the reverse direction of LSC of a typical plastome
                    # picking the vertex with the longest length with strand of least orfs
                    start_vertex, start_direction = sorted(candidate_single_copy_vertices,
                                                           key=lambda x: (-self.vertex_info[x[0]].len,
                                                                          self.vertex_info[x[0]].other_attr["orf"][
                                                                              x[1]]["sum_len"],
                                                                          x))[0]
                    if reverse_start_direction_for_pt:
                        start_direction = not start_direction
                else:
                    # picking the vertex with the longest length with strand of most orfs
                    start_vertex, start_direction = sorted(candidate_single_copy_vertices,
                                                           key=lambda x: (-self.vertex_info[x[0]].len,
                                                                          -self.vertex_info[x[0]].other_attr["orf"][
                                                                              x[1]]["sum_len"],
                                                                          x))[0]
                for go_p, each_path in enumerate(paths):
                    paths[go_p] = reseed_a_path(each_path, (start_vertex, start_direction))

        # sorting path by average distance among multi-copy loci
        # the highest would be more symmetrical IR, which turns out to be more reasonable
        sorted_paths = []
        total_len = len(list(paths)[0])
        record_pattern = False
        for original_id, this_path in enumerate(paths):
            acc_dist = 0
            for copy_num in self.copy_to_vertex:
                if copy_num > 2:
                    record_pattern = True
                    for vertex_name in self.copy_to_vertex[copy_num]:
                        loc_ids = [go_to_id for go_to_id, (v, e) in enumerate(this_path) if v == vertex_name]
                        for id_a, id_b in combinations(loc_ids, 2):
                            acc_dist += min((id_a - id_b) % total_len, (id_b - id_a) % total_len)
            sorted_paths.append((this_path, acc_dist, original_id))
        if record_pattern:
            sorted_paths.sort(key=lambda x: (-x[1], x[2]))
            pattern_dict = {acc_distance: ad_id + 1
                            for ad_id, acc_distance in enumerate(sorted(set([x[1] for x in sorted_paths]),
                                                                        reverse=True))}
            if len(pattern_dict) > 1:
                if mode == "embplant_pt":
                    if log_handler:
                        log_handler.warning("Multiple repeat patterns appeared in your data, "
                                            "a more balanced pattern (always the repeat_pattern1) "
                                            "would be suggested for plastomes with the canonical IR!")
                    else:
                        sys.stdout.write("Warning: Multiple repeat patterns appeared in your data, "
                                         "a more balanced pattern (always the repeat_pattern1) would be suggested "
                                         "for plastomes with the canonical IR!\n")
                sorted_paths = [(this_path, ".repeat_pattern" + str(pattern_dict[acc_distance]))
                                for this_path, acc_distance, foo_id in sorted_paths]
            else:
                sorted_paths = [(this_path, "") for this_path in paths]
        else:
            sorted_paths = [(this_path, "") for this_path in paths]

        if mode == "embplant_pt":
            if len(sorted_paths) > 2 and not (100000 < len(self.export_path(sorted_paths[0][0]).seq) < 200000):
                if log_handler:
                    log_handler.warning("Multiple circular genome structures with abnormal length produced!")
                    log_handler.warning("Please check the assembly graph and selected graph to confirm.")
                else:
                    sys.stdout.write(
                        "Warning: Multiple circular genome structures with abnormal length produced!\n")
                    sys.stdout.write("Please check the assembly graph and selected graph to confirm.\n")
            elif len(sorted_paths) > 2:
                if log_handler:
                    log_handler.warning("Multiple circular genome structures produced!")
                    log_handler.warning("Please check the existence of those isomers "
                                        "by using reads mapping (library information) or longer reads.")
                else:
                    sys.stdout.write("Warning: Multiple circular genome structures produced!\n")
                    sys.stdout.write("Please check the existence of those isomers by "
                                     "using reads mapping (library information) or longer reads.\n")
            elif len(sorted_paths) > 1:
                if log_handler:
                    log_handler.warning("More than one circular genome structure produced ...")
                    log_handler.warning("Please check the final result to confirm whether they are "
                                        " simply different in SSC direction (two flip-flop configurations)!")
                else:
                    sys.stdout.write("More than one circular genome structure produced ...\n")
                    sys.stdout.write("Please check the final result to confirm whether they are "
                                     "simply different in SSC direction (two flip-flop configurations)!\n")
        return sorted_paths
