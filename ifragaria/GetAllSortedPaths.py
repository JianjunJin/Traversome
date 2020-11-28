




class GetAllPaths(object):
	"""

	"""
	def __init__(self, mode="embplant_pt", log_handler=None):
		self.mode = mode
		self.log_handler = log_handler		



    def standardize_paths(self, raw_paths, undirected_vertices):
    	"""

    	"""

		# ...
        if undirected_vertices:
            corrected_paths = [
	            [(this_v, True) if this_v in undirected_vertices 
		         else (this_v, this_e) for this_v, this_e in path_part]
                for path_part in raw_paths
            ]
        else:
            corrected_paths = deepcopy(raw_paths)

		# ...
        here_standardized_path = []
        for part_path in corrected_paths:

			# ...
            if undirected_vertices:
                rev_part = [(this_v, True) if this_v in undirected_vertices else (this_v, not this_e)
                            for this_v, this_e in part_path[::-1]]
            else:
                rev_part = [(this_v, not this_e) for this_v, this_e in part_path[::-1]]

			# ...
            if (part_path[0][0], not part_path[0][1]) \
                    in self.vertex_info[part_path[-1][0]].connections[part_path[-1][1]]:
                # circular
                this_part_derived = [part_path, rev_part]
                for change_start in range(1, len(part_path)):
                    this_part_derived.append(part_path[change_start:] + part_path[:change_start])
                    this_part_derived.append(rev_part[change_start:] + rev_part[:change_start])
                try:
                    standard_part = tuple(sorted(this_part_derived, key=lambda x: smart_trans_for_sort(x))[0])
                except TypeError:
                    for j in this_part_derived:
                        print(j)
                    exit()
            else:
                standard_part = tuple(sorted([part_path, rev_part], key=lambda x: smart_trans_for_sort(x))[0])
            
			# store this part in the path
            here_standardized_path.append(standard_part)

        return corrected_paths, tuple(sorted(here_standardized_path, key=lambda x: smart_trans_for_sort(x)))



        def directed_graph_solver(ongoing_paths, next_connections, vertices_left, in_all_start_ve, undirected_vertices):
            # print("-----------------------------")
            # print("ongoing_path", ongoing_path)
            # print("next_connect", next_connections)
            # print("vertices_lef", vertices_left)
            # print("vertices_lef", len(vertices_left))
            if not vertices_left:
                new_paths, new_standardized = standardize_paths(ongoing_paths, undirected_vertices)
                if new_standardized not in paths_set:
                    paths.append(new_paths)
                    paths_set.add(new_standardized)
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
                    new_connections = sorted(self.vertex_info[next_vertex].connections[not next_end])
                    if not new_left:
                        new_paths, new_standardized = standardize_paths(new_paths, undirected_vertices)
                        if new_standardized not in paths_set:
                            paths.append(new_paths)
                            paths_set.add(new_standardized)
                        return
                    else:
                        if mode == "embplant_pt" and len(new_connections) == 2 and new_connections[0][0] == \
                                new_connections[1][0]:
                            new_connections.sort(
                                key=lambda x: self.vertex_info[x[0]].other_attr["orf"][x[1]]["sum_len"])
                        directed_graph_solver(new_paths, new_connections, new_left, in_all_start_ve,
                                              undirected_vertices)
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
                        new_connections = sorted(self.vertex_info[new_start_vertex].connections[new_start_end])
                        if not new_left:
                            new_paths, new_standardized = standardize_paths(new_paths, undirected_vertices)
                            if new_standardized not in paths_set:
                                paths.append(new_paths)
                                paths_set.add(new_standardized)
                        else:
                            if mode == "embplant_pt" and len(new_connections) == 2 and new_connections[0][0] == \
                                    new_connections[1][0]:
                                new_connections.sort(
                                    key=lambda x: self.vertex_info[x[0]].other_attr["orf"][x[1]]["sum_len"])
                            directed_graph_solver(new_paths, new_connections, new_left, new_all_start_ve,
                                                  undirected_vertices)
                            break
                if not new_all_start_ve:
                    return

        paths = list()
        paths_set = set()
        # start from a terminal vertex in an open graph/subgraph
        #         or a single copy vertex in a closed graph/subgraph
        self.update_orf_total_len()

        # 2019-12-28 palindromic repeats
        palindromic_repeats = set()
        log_palindrome = False
        for vertex_n in self.vertex_info:
            if self.vertex_info[vertex_n].seq[True] == self.vertex_info[vertex_n].seq[False]:
                temp_f = self.vertex_info[vertex_n].connections[True]
                temp_r = self.vertex_info[vertex_n].conncetions[False]
                if temp_f and temp_f == temp_r:
                    log_palindrome = True
                    if len(temp_f) == len(temp_r) == 2:  # simple palindromic repeats, prune repeated connections
                        for go_d, (nb_vertex, nb_direction) in enumerate(tuple(temp_f)):
                            del self.vertex_info[nb_vertex].connections[nb_direction][(vertex_n, bool(go_d))]
                            del self.vertex_info[vertex_n].connections[bool(go_d)][(nb_vertex, nb_direction)]
                    elif len(temp_f) == len(temp_r) == 1:  # connect to the same inverted repeat
                        pass
                    else:  # complicated, recorded
                        palindromic_repeats.add(vertex_n)
        if log_palindrome:
            log_handler.info("Palindromic repeats detected. "
                             "Different paths generating identical sequence will be merged.")

        all_start_v_e = []
        start_vertices = set()
        for go_set, v_set in enumerate(self.vertex_clusters):
            is_closed = True
            for test_vertex_n in sorted(v_set):
                for test_end in (False, True):
                    if not self.vertex_info[test_vertex_n].connections[test_end]:
                        is_closed = False
                        if test_vertex_n not in start_vertices:
                            all_start_v_e.append((test_vertex_n, not test_end))
                            start_vertices.add(test_vertex_n)
            if is_closed:
                if 1 in self.copy_to_vertex[1] and bool(self.copy_to_vertex[1] & v_set):
                    single_copy_v = sorted(self.copy_to_vertex[1] & v_set, key=lambda x: -self.vertex_info[x].len)[0]
                    all_start_v_e.append((single_copy_v, True))
                else:
                    longest_v = sorted(v_set, key=lambda x: -self.vertex_info[x].len)[0]
                    all_start_v_e.append((longest_v, True))
        all_start_v_e.sort(key=lambda x: (smart_trans_for_sort(x[0]), x[1]))
        # start from a self-loop vertex in an open/closed graph/subgraph
        for go_set, v_set in enumerate(self.vertex_clusters):
            for test_vertex_n in sorted(v_set):
                if self.vertex_info[test_vertex_n].is_self_loop():
                    all_start_v_e.append((test_vertex_n, True))
                    all_start_v_e.append((test_vertex_n, False))

        start_v_e = all_start_v_e.pop(0)
        first_path = [[start_v_e]]
        first_connections = sorted(self.vertex_info[start_v_e[0]].connections[start_v_e[1]])
        vertex_to_copy = deepcopy(self.vertex_to_copy)
        vertex_to_copy[start_v_e[0]] -= 1
        if not vertex_to_copy[start_v_e[0]]:
            del vertex_to_copy[start_v_e[0]]
        directed_graph_solver(first_path, first_connections, vertex_to_copy, all_start_v_e,
                              undirected_vertices=palindromic_repeats)

        # standardized_path_unique_set = set([this_path_pair[1] for this_path_pair in path_paris])
        # paths = []
        # for raw_path, standardized_path in path_paris:
        #     if standardized_path in standardized_path_unique_set:
        #         paths.append(raw_path)
        #         standardized_path_unique_set.remove(standardized_path)

        if not paths:
            raise ProcessingGraphFailed("Detecting path(s) from remaining graph failed!")
        else:
            sorted_paths = []
            # total_len = len(list(set(paths))[0])
            record_pattern = False
            for original_id, this_path in enumerate(paths):
                acc_dist = 0
                for copy_num in self.copy_to_vertex:
                    if copy_num > 2:
                        for vertex_name in self.copy_to_vertex[copy_num]:
                            for this_p_part in this_path:
                                loc_ids = [go_to_id for go_to_id, (v, e) in enumerate(this_p_part) if v == vertex_name]
                                if len(loc_ids) > 1:
                                    record_pattern = True
                                    if (this_p_part[0][0], not this_p_part[0][1]) \
                                            in self.vertex_info[this_p_part[-1][0]].connections[this_p_part[-1][1]]:
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
                    if log_handler:
                        if mode == "embplant_pt":
                            log_handler.warning("Multiple repeat patterns appeared in your data, "
                                                "a more balanced pattern (always the repeat_pattern1) would be "
                                                "suggested for plastomes with inverted repeats!")
                        else:
                            log_handler.warning("Multiple repeat patterns appeared in your data.")
                    else:
                        if mode == "embplant_pt":
                            sys.stdout.write("Warning: Multiple repeat patterns appeared in your data, "
                                             "a more balanced pattern (always the repeat_pattern1) would be suggested "
                                             "for plastomes with inverted repeats!\n")
                        else:
                            sys.stdout.write("Warning: Multiple repeat patterns appeared in your data.\n")
                    sorted_paths = [(this_path, ".repeat_pattern" + str(pattern_dict[acc_distance]))
                                    for this_path, acc_distance, foo_id in sorted_paths]
                else:
                    sorted_paths = [(this_path, "") for this_path in sorted(paths)]
            else:
                sorted_paths = [(this_path, "") for this_path in sorted(paths)]

            if mode == "embplant_pt":
                if len(sorted_paths) > 2 and \
                        not (100000 < sum(
                            [len(self.export_path(part_p).seq) for part_p in sorted_paths[0][0]]) < 200000):
                    if log_handler:
                        log_handler.warning("Multiple structures (gene order) with abnormal plastome length produced!")
                        log_handler.warning("Please check the assembly graph and selected graph to confirm.")
                    else:
                        sys.stdout.write(
                            "Warning: Multiple structures (gene order) with abnormal plastome length produced!\n")
                        sys.stdout.write("Please check the assembly graph and selected graph to confirm.\n")
                elif len(sorted_paths) > 2:
                    if log_handler:
                        log_handler.warning("Multiple structures (gene order) produced!")
                        log_handler.warning("Please check the existence of those isomers "
                                            "by using reads mapping (library information) or longer reads.")
                    else:
                        sys.stdout.write("Warning: Multiple structures (gene order) produced!\n")
                        sys.stdout.write("Please check the existence of those isomers by "
                                         "using reads mapping (library information) or longer reads.\n")
                elif len(sorted_paths) > 1:
                    if log_handler:
                        log_handler.warning("More than one structure (gene order) produced ...")
                        log_handler.warning("Please check the final result to confirm whether they are "
                                            " simply different in SSC direction (two flip-flop configurations)!")
                    else:
                        sys.stdout.write("More than one structure (gene order) produced ...\n")
                        sys.stdout.write("Please check the final result to confirm whether they are "
                                         " simply different in SSC direction (two flip-flop configurations)!\n")
            return sorted_paths