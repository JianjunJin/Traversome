






class FindTargetGraph(object):
	"""

	"""

	def __init__(
        self, 
        tab_file, 
        database_name, 
        mode="embplant_pt", 
        type_factor=3, 
        weight_factor=100.0,
        max_contig_multiplicity=8, 
        min_sigma_factor=0.1, 
        expected_max_size=INF, 
        expected_min_size=0,
        log_hard_cov_threshold=10., 
        contamination_depth=3., 
        contamination_similarity=0.95,
        degenerate=True, 
        degenerate_depth=1.5, 
        degenerate_similarity=0.98, 
        only_keep_max_cov=True,
        min_single_copy_percent=50, 
        meta=False,
        broken_graph_allowed=False, 
        temp_graph=None, 
        verbose=True,
        read_len_for_log=None, 
        kmer_for_log=None,
        log_handler=None, 
        debug=False):
        """
        :param tab_file:
        :param database_name:
        :param mode:
        :param type_factor:
        :param weight_factor:
        :param max_contig_multiplicity:
        :param min_sigma_factor:
        :param expected_max_size:
        :param expected_min_size:
        :param log_hard_cov_threshold:
        :param contamination_depth:
        :param contamination_similarity:
        :param degenerate:
        :param degenerate_depth:
        :param degenerate_similarity:
        :param only_keep_max_cov:
        :param min_single_copy_percent: [0-100]
        :param broken_graph_allowed:
        :param temp_graph:
        :param verbose:
        :param read_len_for_log:
        :param kmer_for_log:
        :param log_handler:
        :param debug:
        :return:
        """







        overlap = self.__overlap if self.__overlap else 0

        def log_target_res(self, final_res_combinations_inside):
        	"""

        	"""

            echo_graph_id = int(bool(len(final_res_combinations_inside) - 1))
            
            for go_res, final_res_one in enumerate(final_res_combinations_inside):
                this_graph = final_res_combinations_inside[go_res]["graph"]
                this_k_cov = round(final_res_combinations_inside[go_res]["cov"], 3)
                if read_len_for_log and kmer_for_log:
                    this_b_cov = round(this_k_cov * read_len_for_log / (read_len_for_log - kmer_for_log + 1), 3)
                else:
                    this_b_cov = None
            
                if log_handler:
                    if echo_graph_id:
                        log_handler.info("Graph " + str(go_res + 1))
                    for vertex_set in sorted(this_graph.vertex_clusters):
                        copies_in_a_set = {this_graph.vertex_to_copy[v_name] for v_name in vertex_set}
                        if copies_in_a_set != {1}:
                            for in_vertex_name in sorted(vertex_set):
                                log_handler.info("Vertex_" + in_vertex_name + " #copy = " +
                                                 str(this_graph.vertex_to_copy.get(in_vertex_name, 1)))
                    cov_str = " kmer-coverage" if bool(overlap) else " coverage"
                    log_handler.info("Average " + mode + cov_str +
                                     ("(" + str(go_res + 1) + ")") * echo_graph_id + " = " + "%.1f" % this_k_cov)
                    if this_b_cov:
                        log_handler.info("Average " + mode + " base-coverage" +
                                         ("(" + str(go_res + 1) + ")") * echo_graph_id + " = " + "%.1f" % this_b_cov)
            
                else:
                    if echo_graph_id:
                        sys.stdout.write("Graph " + str(go_res + 1) + "\n")
                    for vertex_set in sorted(this_graph.vertex_clusters):
                        copies_in_a_set = {this_graph.vertex_to_copy[v_name] for v_name in vertex_set}
                        if copies_in_a_set != {1}:
                            for in_vertex_name in sorted(vertex_set):
                                sys.stdout.write("Vertex_" + in_vertex_name + " #copy = " +
                                                 str(this_graph.vertex_to_copy.get(in_vertex_name, 1)) + "\n")
                    cov_str = " kmer-coverage" if bool(overlap) else " coverage"
                    sys.stdout.write("Average " + mode + cov_str +
                                     ("(" + str(go_res + 1) + ")") * echo_graph_id + " = " + "%.1f" % this_k_cov + "\n")
                    if this_b_cov:
                        sys.stdout.write("Average " + mode + " base-coverage" + ("(" + str(go_res + 1) + ")") *
                                         echo_graph_id + " = " + "%.1f" % this_b_cov + "\n")

        if temp_graph:
            if temp_graph.endswith(".gfa"):
                temp_csv = temp_graph[:-3] + "csv"
            elif temp_graph.endswith(".fastg"):
                temp_csv = temp_graph[:-5] + "csv"
            elif temp_graph.endswith(".fasta"):
                temp_csv = temp_graph[:-5] + "csv"
            else:
                temp_csv = temp_graph + ".csv"
        else:
            temp_csv = None
        count_all_temp = [1]


        def add_temp_id(old_temp_file, extra_str=""):
            if old_temp_file.endswith(".gfa"):
                return old_temp_file[:-4] + extra_str + ".gfa"
            elif old_temp_file.endswith(".csv"):
                return old_temp_file[:-4] + extra_str + ".csv"
            else:
                return old_temp_file + extra_str


        def write_temp_out(_assembly, _database_name, _temp_graph, _temp_csv, go_id):
            if _temp_graph:
                tmp_graph_1 = add_temp_id(_temp_graph, ".%02d.%02d" % (count_all_temp[0], go_id))
                tmp_csv_1 = add_temp_id(_temp_csv, ".%02d.%02d" % (count_all_temp[0], go_id))
                if verbose:
                    if log_handler:
                        log_handler.info("Writing out temp graph (%d): %s" % (go_id, tmp_graph_1))
                    else:
                        sys.stdout.write("Writing out temp graph (%d): %s" % (go_id, tmp_graph_1) + "\n")
                _assembly.write_to_gfa(tmp_graph_1)
                _assembly.write_out_tags([_database_name], tmp_csv_1)
                count_all_temp[0] += 1




        if broken_graph_allowed and not meta:
            weight_factor = 10000.

        if meta:
            try:
                self.parse_tab_file(
                    tab_file, database_name=database_name, type_factor=type_factor, log_handler=log_handler)
            except ProcessingGraphFailed:
                return []
        else:
            self.parse_tab_file(tab_file, database_name=database_name, type_factor=type_factor, log_handler=log_handler)
        new_assembly = deepcopy(self)
        is_reasonable_res = False
        data_contains_outlier = False
        try:
            while not is_reasonable_res:
                is_reasonable_res = True
                if verbose or debug:
                    if log_handler:
                        log_handler.info("tagged vertices: " + str(sorted(new_assembly.tagged_vertices[database_name])))
                        log_handler.info("tagged coverage: " +
                                         str(["%.1f" % new_assembly.vertex_info[log_v].cov
                                              for log_v in sorted(new_assembly.tagged_vertices[database_name])]))
                    else:
                        sys.stdout.write("tagged vertices: " + str(sorted(new_assembly.tagged_vertices[database_name]))
                                         + "\n")
                        sys.stdout.write("tagged coverage: " +
                                         str(["%.1f" % new_assembly.vertex_info[log_v].cov
                                              for log_v in sorted(new_assembly.tagged_vertices[database_name])]) + "\n")
                new_assembly.merge_all_possible_vertices()
                new_assembly.tag_in_between(database_n=database_name)
                # new_assembly.processing_polymorphism(mode=mode, contamination_depth=contamination_depth,
                #                                      contamination_similarity=contamination_similarity,
                #                                      degenerate=False, verbose=verbose, debug=debug,
                #                                      log_handler=log_handler)
                write_temp_out(new_assembly, database_name, temp_graph, temp_csv, 1)
                changed = True
                count_large_round = 0
                while changed:
                    count_large_round += 1
                    if verbose or debug:
                        if log_handler:
                            log_handler.info(
                                "===================== " + str(count_large_round) + " =====================")
                        else:
                            sys.stdout.write(
                                "===================== " + str(count_large_round) + " =====================\n")
                    changed = False
                    cluster_trimmed = True
                    while cluster_trimmed:
                        # remove low coverages
                        first_round = True
                        delete_those_vertices = set()
                        parameters = []
                        this_del = False
                        new_assembly.estimate_copy_and_depth_by_cov(
                            new_assembly.tagged_vertices[database_name], debug=debug, log_handler=log_handler,
                            verbose=verbose, mode=mode)
                        while first_round or delete_those_vertices or this_del:
                            if data_contains_outlier:
                                this_del, parameters = \
                                    new_assembly.filter_by_coverage(database_n=database_name,
                                                                    weight_factor=weight_factor,
                                                                    log_hard_cov_threshold=log_hard_cov_threshold,
                                                                    min_sigma_factor=min_sigma_factor,
                                                                    min_cluster=2, log_handler=log_handler,
                                                                    verbose=verbose, debug=debug)
                                data_contains_outlier = False
                                if not this_del:
                                    raise ProcessingGraphFailed(
                                        "Unable to generate result with single copy vertex percentage < {}%"
                                            .format(min_single_copy_percent))
                            else:
                                this_del, parameters = \
                                    new_assembly.filter_by_coverage(database_n=database_name,
                                                                    weight_factor=weight_factor,
                                                                    log_hard_cov_threshold=log_hard_cov_threshold,
                                                                    min_sigma_factor=min_sigma_factor,
                                                                    log_handler=log_handler, verbose=verbose,
                                                                    debug=debug)
                            if verbose or debug:
                                if log_handler:
                                    log_handler.info("tagged vertices: " +
                                                     str(sorted(new_assembly.tagged_vertices[database_name])))
                                    log_handler.info("tagged coverage: " +
                                                     str(["%.1f" % new_assembly.vertex_info[log_v].cov
                                                          for log_v
                                                          in sorted(new_assembly.tagged_vertices[database_name])]))
                                else:
                                    sys.stdout.write("tagged vertices: " +
                                                     str(sorted(new_assembly.tagged_vertices[database_name])) + "\n")
                                    log_handler.info("tagged coverage: " +
                                                     str(["%.1f" % new_assembly.vertex_info[log_v].cov
                                                          for log_v
                                                          in
                                                          sorted(new_assembly.tagged_vertices[database_name])]) + "\n")
                            new_assembly.estimate_copy_and_depth_by_cov(
                                new_assembly.tagged_vertices[database_name], debug=debug, log_handler=log_handler,
                                verbose=verbose, mode=mode)
                            first_round = False

                        if new_assembly.exclude_other_hits(database_n=database_name):
                            changed = True

                        cluster_trimmed = False

                        if len(new_assembly.vertex_clusters) == 0:
                            raise ProcessingGraphFailed("No available " + mode + " components detected!")
                        elif len(new_assembly.vertex_clusters) == 1:
                            pass
                        else:
                            cluster_weights = [sum([new_assembly.vertex_info[x_v].other_attr["weight"][database_name]
                                                    for x_v in x
                                                    if
                                                    "weight" in new_assembly.vertex_info[x_v].other_attr
                                                    and
                                                    database_name in new_assembly.vertex_info[x_v].other_attr[
                                                        "weight"]])
                                               for x in new_assembly.vertex_clusters]
                            best = max(cluster_weights)
                            best_id = cluster_weights.index(best)
                            if broken_graph_allowed:
                                id_remained = {best_id}
                                for j, w in enumerate(cluster_weights):
                                    if w * weight_factor > best:
                                        id_remained.add(j)
                                    else:
                                        for del_v in new_assembly.vertex_clusters[j]:
                                            if del_v in new_assembly.tagged_vertices[database_name]:
                                                new_cov = new_assembly.vertex_info[del_v].cov
                                                for mu, sigma in parameters:
                                                    if abs(new_cov - mu) < sigma:
                                                        id_remained.add(j)
                                                        break
                                            if j in id_remained:
                                                break
                            else:
                                # chose the target cluster (best rank)
                                id_remained = {best_id}
                                temp_cluster_weights = deepcopy(cluster_weights)
                                del temp_cluster_weights[best_id]
                                second = max(temp_cluster_weights)
                                if best < second * weight_factor:
                                    write_temp_out(new_assembly, database_name, temp_graph, temp_csv, 2)
                                    raise ProcessingGraphFailed("Multiple isolated " + mode + " components detected! "
                                                                                              "Broken or contamination?")
                                for j, w in enumerate(cluster_weights):
                                    if w == second:
                                        for del_v in new_assembly.vertex_clusters[j]:
                                            if del_v in new_assembly.tagged_vertices[database_name]:
                                                new_cov = new_assembly.vertex_info[del_v].cov
                                                # for debug
                                                # print(new_cov)
                                                # print(parameters)
                                                for mu, sigma in parameters:
                                                    if abs(new_cov - mu) < sigma:
                                                        write_temp_out(new_assembly, database_name,
                                                                       temp_graph, temp_csv, 3)
                                                        raise ProcessingGraphFailed(
                                                            "Complicated graph: please check around EDGE_" + del_v + "!"
                                                                                                                     "# tags: " +
                                                            str(new_assembly.vertex_info[del_v].other_attr.
                                                                get("tags", {database_name: ""})[database_name]))

                            # remove other clusters
                            vertices_to_del = set()
                            for go_cl, v_2_del in enumerate(new_assembly.vertex_clusters):
                                if go_cl not in id_remained:
                                    vertices_to_del |= v_2_del
                            if vertices_to_del:
                                if verbose or debug:
                                    if log_handler:
                                        log_handler.info("removing other clusters: " + str(vertices_to_del))
                                    else:
                                        sys.stdout.write("removing other clusters: " + str(vertices_to_del) + "\n")
                                new_assembly.remove_vertex(vertices_to_del)
                                cluster_trimmed = True
                                changed = True

                    # merge vertices
                    new_assembly.merge_all_possible_vertices()
                    new_assembly.tag_in_between(database_n=database_name)

                    # no tip contigs allowed
                    if broken_graph_allowed:
                        pass
                    else:
                        first_round = True
                        delete_those_vertices = set()
                        while first_round or delete_those_vertices:
                            first_round = False
                            delete_those_vertices = set()
                            for vertex_name in new_assembly.vertex_info:
                                # both ends must have edge(s)
                                if sum([bool(len(cn))
                                        for cn in new_assembly.vertex_info[vertex_name].connections.values()]) != 2:
                                    if vertex_name in new_assembly.tagged_vertices[database_name]:
                                        write_temp_out(new_assembly, database_name, temp_graph, temp_csv, 4)
                                        raise ProcessingGraphFailed(
                                            "Incomplete/Complicated graph: please check around EDGE_" + vertex_name + "!")
                                    else:
                                        delete_those_vertices.add(vertex_name)
                            if delete_those_vertices:
                                if verbose or debug:
                                    if log_handler:
                                        log_handler.info("removing terminal contigs: " + str(delete_those_vertices))
                                    else:
                                        sys.stdout.write(
                                            "removing terminal contigs: " + str(delete_those_vertices) + "\n")
                                new_assembly.remove_vertex(delete_those_vertices)
                                changed = True

                    # # merge vertices
                    # new_assembly.merge_all_possible_vertices()
                    # new_assembly.tag_in_between(mode=mode)
                    # break self-connection if necessary
                    # for vertex_name in new_assembly.vertex_info:
                    #     if (vertex_name, True) in
                    # -> not finished!!

                    # merge vertices
                    new_assembly.merge_all_possible_vertices()
                    new_assembly.processing_polymorphism(database_name=database_name,
                                                         contamination_depth=contamination_depth,
                                                         contamination_similarity=contamination_similarity,
                                                         degenerate=False, degenerate_depth=degenerate_depth,
                                                         degenerate_similarity=degenerate_similarity,
                                                         verbose=verbose, debug=debug, log_handler=log_handler)
                    new_assembly.tag_in_between(database_n=database_name)
                    write_temp_out(new_assembly, database_name, temp_graph, temp_csv, 5)

                write_temp_out(new_assembly, database_name, temp_graph, temp_csv, 6)
                new_assembly.processing_polymorphism(database_name=database_name,
                                                     contamination_depth=contamination_depth,
                                                     contamination_similarity=contamination_similarity,
                                                     degenerate=degenerate, degenerate_depth=degenerate_depth,
                                                     degenerate_similarity=degenerate_similarity,
                                                     warning_count=1, only_keep_max_cov=only_keep_max_cov,
                                                     verbose=verbose, debug=debug, log_handler=log_handler)
                new_assembly.merge_all_possible_vertices()
                write_temp_out(new_assembly, database_name, temp_graph, temp_csv, 7)

                # create idealized vertices and edges
                try:
                    new_average_cov = new_assembly.estimate_copy_and_depth_by_cov(log_handler=log_handler,
                                                                                  verbose=verbose,
                                                                                  mode="all", debug=debug)
                    if verbose:
                        if log_handler:
                            log_handler.info("Estimating copy and depth precisely ...")
                        else:
                            sys.stdout.write("Estimating copy and depth precisely ...\n")
                    final_res_combinations = new_assembly.estimate_copy_and_depth_precisely(
                        maximum_copy_num=max_contig_multiplicity, broken_graph_allowed=broken_graph_allowed,
                        return_new_graphs=True, log_handler=log_handler,
                        verbose=verbose, debug=debug)
                    if verbose:
                        if log_handler:
                            log_handler.info(str(len(final_res_combinations)) + " candidate graph(s) generated.")
                        else:
                            sys.stdout.write(str(len(final_res_combinations)) + " candidate graph(s) generated.\n")
                    absurd_copy_nums = True
                    no_single_copy = True
                    while absurd_copy_nums:
                        go_graph = 0
                        while go_graph < len(final_res_combinations):
                            this_assembly_g = final_res_combinations[go_graph]["graph"]
                            this_parallel_v_sets = [v_set for v_set in this_assembly_g.detect_parallel_vertices()]
                            this_parallel_names = set([v_n for v_set in this_parallel_v_sets for v_n, v_e in v_set])
                            if 1 not in this_assembly_g.copy_to_vertex:
                                if verbose or debug:
                                    if log_handler:
                                        for vertex_name in sorted(this_assembly_g.vertex_info):
                                            log_handler.info(
                                                "Vertex_" + vertex_name + " #copy = " +
                                                str(this_assembly_g.vertex_to_copy.get(vertex_name, 1)))
                                        log_handler.info("Removing this graph without single copy contigs.")
                                    else:
                                        for vertex_name in sorted(this_assembly_g.vertex_info):
                                            sys.stdout.write(
                                                "Vertex_" + vertex_name + " #copy = " +
                                                str(this_assembly_g.vertex_to_copy.get(vertex_name, 1)) + "\n")
                                        sys.stdout.write("Removing this graph without single copy contigs.\n")
                                del final_res_combinations[go_graph]
                            else:
                                no_single_copy = False
                                this_absurd = True
                                for single_copy_v in this_assembly_g.copy_to_vertex[1]:
                                    if single_copy_v not in this_parallel_names:
                                        this_absurd = False
                                draft_size_estimates = 0
                                for inside_v in this_assembly_g.vertex_info:
                                    draft_size_estimates += \
                                        (this_assembly_g.vertex_info[inside_v].len - this_assembly_g.overlap()) * \
                                        this_assembly_g.vertex_to_copy[inside_v]
                                if not this_absurd or expected_min_size < draft_size_estimates < expected_max_size:
                                    absurd_copy_nums = False
                                    go_graph += 1
                                else:
                                    if verbose or debug:
                                        if log_handler:
                                            log_handler.info(
                                                "Removing graph with draft size: " + str(draft_size_estimates))
                                        else:
                                            sys.stdout.write(
                                                "Removing graph with draft size: " + str(draft_size_estimates) + "\n")
                                    # add all combinations
                                    for index_set in generate_index_combinations([len(v_set)
                                                                                  for v_set in this_parallel_v_sets]):
                                        new_possible_graph = deepcopy(this_assembly_g)
                                        dropping_names = []
                                        for go_set, this_v_set in enumerate(this_parallel_v_sets):
                                            keep_this = index_set[go_set]
                                            for go_ve, (this_name, this_end) in enumerate(this_v_set):
                                                if go_ve != keep_this:
                                                    dropping_names.append(this_name)
                                        # if log_handler:
                                        #     log_handler.info("Dropping vertices " + " ".join(dropping_names))
                                        # else:
                                        #     log_handler.info("Dropping vertices " + "".join(dropping_names) + "\n")
                                        new_possible_graph.remove_vertex(dropping_names)
                                        new_possible_graph.merge_all_possible_vertices()
                                        new_possible_graph.estimate_copy_and_depth_by_cov(
                                            log_handler=log_handler, verbose=verbose, mode="all", debug=debug)
                                        final_res_combinations.extend(
                                            new_possible_graph.estimate_copy_and_depth_precisely(
                                                maximum_copy_num=max_contig_multiplicity,
                                                broken_graph_allowed=broken_graph_allowed, return_new_graphs=True,
                                                log_handler=log_handler, verbose=verbose, debug=debug))

                                    del final_res_combinations[go_graph]
                        if not final_res_combinations and absurd_copy_nums:
                            # if absurd_copy_nums:
                            #     raise ProcessingGraphFailed("Complicated graph! Detecting path(s) failed!")
                            # else:
                            raise ProcessingGraphFailed("Complicated graph! Detecting path(s) failed!")
                    if no_single_copy:
                        raise ProcessingGraphFailed("No single copy region?! Detecting path(s) failed!")
                except ImportError as e:
                    raise e
                except (RecursionError, Exception) as e:
                    if broken_graph_allowed:
                        unlabelled_contigs = [check_v for check_v in list(new_assembly.vertex_info)
                                              if check_v not in new_assembly.tagged_vertices[database_name]]
                        if unlabelled_contigs:
                            if verbose or debug:
                                if log_handler:
                                    log_handler.info("removing unlabelled contigs: " + str(unlabelled_contigs))
                                else:
                                    sys.stdout.write("removing unlabelled contigs: " + str(unlabelled_contigs) + "\n")
                            new_assembly.remove_vertex(unlabelled_contigs)
                            new_assembly.merge_all_possible_vertices()
                        else:
                            # delete all previous connections if all present contigs are labelled
                            for del_v_connection in new_assembly.vertex_info:
                                new_assembly.vertex_info[del_v_connection].connections = {True: OrderedDict(),
                                                                                          False: OrderedDict()}
                            new_assembly.update_vertex_clusters()
                        new_average_cov = new_assembly.estimate_copy_and_depth_by_cov(
                            re_initialize=True, log_handler=log_handler, verbose=verbose, mode="all", debug=debug)
                        outer_continue = False
                        for remove_all_connections in (False, True):
                            if remove_all_connections:  # delete all previous connections
                                for del_v_connection in new_assembly.vertex_info:
                                    new_assembly.vertex_info[del_v_connection].connections = {True: OrderedDict(),
                                                                                              False: OrderedDict()}
                            new_assembly.update_vertex_clusters()
                            try:
                                here_max_copy = 1 if remove_all_connections else max_contig_multiplicity
                                final_res_combinations = new_assembly.estimate_copy_and_depth_precisely(
                                    maximum_copy_num=here_max_copy, broken_graph_allowed=True, return_new_graphs=True,
                                    log_handler=log_handler, verbose=verbose, debug=debug)
                            except ImportError as e:
                                raise e
                            except Exception as e:
                                if verbose or debug:
                                    if log_handler:
                                        log_handler.info(str(e))
                                    else:
                                        sys.stdout.write(str(e) + "\n")
                                continue
                            test_first_g = final_res_combinations[0]["graph"]
                            if 1 in test_first_g.copy_to_vertex:
                                single_copy_percent = sum([test_first_g.vertex_info[s_v].len
                                                           for s_v in test_first_g.copy_to_vertex[1]]) \
                                                      / float(sum([test_first_g.vertex_info[a_v].len
                                                                   for a_v in test_first_g.vertex_info]))
                                if single_copy_percent < 0.5:
                                    if verbose:
                                        if log_handler:
                                            log_handler.warning(
                                                "Result with single copy vertex percentage < 50% is "
                                                "unacceptable, continue dropping suspicious vertices ...")
                                        else:
                                            sys.stdout.write(
                                                "Warning: Result with single copy vertex percentage < 50% is "
                                                "unacceptable, continue dropping suspicious vertices ...")
                                    data_contains_outlier = True
                                    is_reasonable_res = False
                                    outer_continue = True
                                    break
                                else:
                                    log_target_res(final_res_combinations)
                                    return final_res_combinations
                            else:
                                if verbose:
                                    if log_handler:
                                        log_handler.warning("Result with single copy vertex percentage < 50% is "
                                                            "unacceptable, continue dropping suspicious vertices ...")
                                    else:
                                        sys.stdout.write("Warning: Result with single copy vertex percentage < 50% is "
                                                         "unacceptable, continue dropping suspicious vertices ...")
                                data_contains_outlier = True
                                is_reasonable_res = False
                                outer_continue = True
                                break
                        if outer_continue:
                            continue
                    elif temp_graph:
                        write_temp_out(new_assembly, database_name, temp_graph, temp_csv, 8)
                        raise ProcessingGraphFailed("Complicated " + mode + " graph! Detecting path(s) failed!")
                    else:
                        if verbose and log_handler:
                            log_handler.exception("")
                        raise e
                else:
                    test_first_g = final_res_combinations[0]["graph"]
                    if 1 in test_first_g.copy_to_vertex or min_single_copy_percent == 0:
                        single_copy_percent = sum([test_first_g.vertex_info[s_v].len
                                                   for s_v in test_first_g.copy_to_vertex[1]]) \
                                              / float(sum([test_first_g.vertex_info[a_v].len
                                                           for a_v in test_first_g.vertex_info]))
                        if single_copy_percent < min_single_copy_percent / 100.:
                            if verbose:
                                if log_handler:
                                    log_handler.warning("Result with single copy vertex percentage < {}% is "
                                                        "unacceptable, continue dropping suspicious vertices ..."
                                                        .format(min_single_copy_percent))
                                else:
                                    sys.stdout.write("Warning: Result with single copy vertex percentage < {}% is "
                                                     "unacceptable, continue dropping suspicious vertices ..."
                                                     .format(min_single_copy_percent))
                            data_contains_outlier = True
                            is_reasonable_res = False
                            continue
                        else:
                            log_target_res(final_res_combinations)
                            return final_res_combinations
                    else:
                        if verbose:
                            if log_handler:
                                log_handler.warning("Result with single copy vertex percentage < {}% is "
                                                    "unacceptable, continue dropping suspicious vertices ..."
                                                    .format(min_single_copy_percent))
                            else:
                                sys.stdout.write("Warning: Result with single copy vertex percentage < {}% is "
                                                 "unacceptable, continue dropping suspicious vertices ..."
                                                 .format(min_single_copy_percent))
                        data_contains_outlier = True
                        is_reasonable_res = False
                        continue
        except KeyboardInterrupt as e:
            write_temp_out(new_assembly, database_name, temp_graph, temp_csv, 9)
            if log_handler:
                log_handler.exception("")
                raise e
            else:
                raise e