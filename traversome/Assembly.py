#!/usr/bin/env python

"""
Assembly class object and associated class objects
"""

import os
import sys
from copy import deepcopy
from collections import OrderedDict
from loguru import logger
from traversome.AssemblySimple import AssemblySimple #, VertexMergingHistory, VertexEditHistory
from traversome.PathGeneratorGraphOnly import PathGeneratorGraphOnly
from traversome.PathGeneratorGraphAlignment import PathGeneratorGraphAlignment
from traversome.EstMultiplicityFromCov import EstMultiplicityFromCov
from traversome.EstMultiplicityPrecise import EstMultiplicityPrecise
from traversome.utils import (
    Sequence, SequenceList, 
    ProcessingGraphFailed, 
    INF, 
    get_orf_lengths, 
    generate_clusters_from_connections,
    # smart_trans_for_sort,
)


class Assembly(AssemblySimple):
    """
    Main class object for storing and 
    """
    def __init__(
            self,
            graph_file=None,
            min_cov=0.,
            max_cov=INF,
            overlap=None,
            record_reversed_paths=True):
        """
        :param graph_file:
        :param min_cov:
        :param max_cov:
        """
        # inherit values from base class
        super().__init__(graph_file, min_cov, max_cov, overlap)
        self.__overlap = self.overlap()
        # super(Assembly, self).__init__(graph_file=graph_file, min_cov=min_cov, max_cov=max_cov, overlap=overlap)
        # self.__overlap = super(Assembly, self).overlap()

        # get an initial set of clusters of connected vertices
        self.vertex_clusters = []
        self.update_vertex_clusters()

        # destinations to fill, what are these?
        self.tagged_vertices = {}
        self.vertex_to_copy = {}
        self.vertex_to_float_copy = {}
        self.copy_to_vertex = {}
        self.__inverted_repeat_vertex = {}
        # self.__vertex_max_copy_id = {}
        self.ave_depth = None

        # optional
        self.palindromic_repeats = None
        self.__palindromic_repeat_detected = False
        self.__record_reversed_paths_to_mem = record_reversed_paths
        self.__reverse_paths = {}

        # summarize init
        logger.debug("init graph: self.vertex_clusters={}".format(self.vertex_clusters))

    def new_graph_with_vertices_renamed(
            self,
            name_translate_dict,
            those_vertices=None,
            fill_fastg_form_name=False):
            # edit_history=None):
        """
        :param name_translate_dict:
        :param those_vertices: transfer all vertices by default
        :param fill_fastg_form_name:
        :return:
        """
        new_graph = Assembly(overlap=self.__overlap)
        if not those_vertices:
            those_vertices = sorted(self.vertex_info)
        for old_name in sorted(those_vertices):
            this_v_info = deepcopy(self.vertex_info[old_name])
            if old_name not in name_translate_dict:
                new_graph.vertex_info[old_name] = this_v_info
            else:
                new_name = name_translate_dict[old_name]
                this_v_info.label = new_name
                this_v_info.connections = {True: OrderedDict(), False: OrderedDict()}
                for this_end in self.vertex_info[old_name].connections:
                    for next_name, next_end in self.vertex_info[old_name].connections[this_end]:
                        this_v_info.connections[this_end][(name_translate_dict.get(next_name, next_name), next_end)] = \
                            self.vertex_info[old_name].connections[this_end][(next_name, next_end)]
                if fill_fastg_form_name:
                    this_v_info.fill_fastg_form_name()
                # if edit_history and new_name in edit_history:
                #     this_v_info.merging_history = VertexMergingHistory([(edit_history[new_name], True)])
                new_graph.vertex_info[new_name] = this_v_info
        return new_graph

    def new_graph_with_vertices_reseeded(self, start_from=1):
        """
        A function called from within .write_to_fastg()
        """
        name_trans = {self.vertex_info[go - start_from]: str(go)
                      for go in range(start_from, start_from + len(self.vertex_info))}
        new_graph = self.new_graph_with_vertices_renamed(name_trans, fill_fastg_form_name=True)
        return new_graph, name_trans

    def write_to_fastg(
            self,
            out_file,
            check_postfix=True,
            rename_if_needed=False,
            out_renaming_table=None,
            echo_rename_warning=False,
            log_handler=None):
        """
        Write the graph to fastg format ...
        """

        # 
        if check_postfix and not out_file.endswith(".fastg"):
            out_file += ".fastg"
        
        # 
        try:
            out_matrix = SequenceList()
            for vertex_name in self.vertex_info:
                this_name = self.vertex_info[vertex_name].fastg_form_name
                for this_end in (False, True):
                    seq_name = [this_name, ("", "'")[not this_end]]
                    if self.vertex_info[vertex_name].connections[this_end]:
                        seq_name.append(":")
                        connect_str = ",".join([self.vertex_info[n_v].fastg_form_name + ("", "'")[n_e]
                                                for n_v, n_e in self.vertex_info[vertex_name].connections[this_end]])
                        seq_name.append(connect_str)
                    seq_name.append(";")
                    out_matrix.append(Sequence("".join(seq_name), self.vertex_info[vertex_name].seq[this_end]))
            out_matrix.interleaved = 70
            out_matrix.write_fasta(out_file)

        except TypeError:
            if rename_if_needed:
                if echo_rename_warning:
                    if log_handler:
                        log_handler.info("Graph converted to new fastg with original Vertex names lost.")
                    else:
                        sys.stdout.write("Graph converted to new fastg with original Vertex names lost.\n")
                new_graph, name_trans = self.new_graph_with_vertices_reseeded()
                new_graph.write_to_fastg(out_file, check_postfix=False)
                if out_renaming_table:
                    with open(out_renaming_table + ".Temp", "w") as out_table:
                        for old_name in sorted(name_trans):
                            out_table.write(old_name + "\t" + name_trans[old_name] + "\n")
                    os.rename(out_renaming_table + ".Temp", out_renaming_table)
                    if echo_rename_warning:
                        if log_handler:
                            log_handler.info("Table (original Vertex names -> new Vertex names) written to " +
                                             out_renaming_table + ".")
                        else:
                            sys.stdout.write("Table (original Vertex names -> new Vertex names) written to " +
                                             out_renaming_table + ".\n")
            else:
                raise ProcessingGraphFailed(
                    "Merged graph cannot be written as fastg format file, please try gfa format!")

    def update_orf_total_len(self, limited_vertices=None):
        if not limited_vertices:
            limited_vertices = sorted(self.vertex_info)
        else:
            limited_vertices = sorted(limited_vertices)
        for vertex_name in limited_vertices:
            self.vertex_info[vertex_name].other_attr["orf"] = {}
            for direction in (True, False):
                this_orf_lens = get_orf_lengths(self.vertex_info[vertex_name].seq[direction])
                self.vertex_info[vertex_name].other_attr["orf"][direction] = {"lengths": this_orf_lens,
                                                                              "sum_len": sum(this_orf_lens)}

    def update_vertex_clusters(self):
        """
        Find connected vertices and store clusters in .vertex_clusters        
        Called during Assembly.__init__(), and can be called again at other times
        such as after removing a vertex.
        """
        self.vertex_clusters = \
            generate_clusters_from_connections(self.vertex_info,
                                               {this_v:
                                                    (next_v
                                                     for connected_set in self.vertex_info[this_v].connections.values()
                                                     for next_v in connected_set)
                                                for this_v in self.vertex_info})

    def remove_vertex(self, vertices, update_cluster=True):
        """
        ...
        """
        for vertex_name in vertices:
            
            # ...
            for this_end, connected_dict in list(self.vertex_info[vertex_name].connections.items()):
                for next_v, next_e in list(connected_dict):
                    del self.vertex_info[next_v].connections[next_e][(vertex_name, this_end)]
            del self.vertex_info[vertex_name]
            
            # ...
            for tag in self.tagged_vertices:
                if vertex_name in self.tagged_vertices[tag]:
                    self.tagged_vertices[tag].remove(vertex_name)
            
            # ...
            if vertex_name in self.vertex_to_copy:
                this_copy = self.vertex_to_copy[vertex_name]
                self.copy_to_vertex[this_copy].remove(vertex_name)
                if not self.copy_to_vertex[this_copy]:
                    del self.copy_to_vertex[this_copy]
                del self.vertex_to_copy[vertex_name]
                del self.vertex_to_float_copy[vertex_name]

        # recalculate clusters (connected vertices) now that a vertices were removed
        if update_cluster:
            self.update_vertex_clusters()
        
        # reset irv to empty dict
        self.__inverted_repeat_vertex = {}

    def detect_parallel_vertices(self, limited_vertices=None):
        """
        called inside find_target_graph()
        """
        if not limited_vertices:
            limiting = False
            limited_vertices = sorted(self.vertex_info)
        else:
            limiting = True
            limited_vertices = sorted(limited_vertices)
        all_both_ends = {}
        for vertex_name in limited_vertices:
            this_cons = self.vertex_info[vertex_name].connections
            connect_1 = this_cons[True]
            connect_2 = this_cons[False]
            if connect_1 and connect_2:
                this_ends_raw = [tuple(sorted(connect_1)), tuple(sorted(connect_2))]
                this_ends = sorted(this_ends_raw)
                direction_remained = this_ends_raw == this_ends
                this_ends = tuple(this_ends)
                if this_ends not in all_both_ends:
                    all_both_ends[this_ends] = set()
                all_both_ends[this_ends].add((vertex_name, direction_remained))
        if limiting:
            limited_vertex_set = set(limited_vertices)
            for each_vertex in self.vertex_info:
                if each_vertex not in limited_vertex_set:
                    this_cons = self.vertex_info[each_vertex].connections
                    connect_1 = this_cons[True]
                    connect_2 = this_cons[False]
                    if connect_1 and connect_2:
                        this_ends_raw = [tuple(sorted(connect_1)), tuple(sorted(connect_2))]
                        this_ends = sorted(this_ends_raw)
                        direction_remained = this_ends_raw == this_ends
                        this_ends = tuple(this_ends)
                        if this_ends in all_both_ends:
                            all_both_ends[this_ends].add((each_vertex, direction_remained))
        return [vertices for vertices in all_both_ends.values() if len(vertices) > 1]

    def duplicate(self, vertices, num_dup=2, depth_factors=None):
        """
        :param vertices:
        :param num_dup:
        :param depth_factors:
            The ratio of new depth over previous depth.
            By default the depth will be equally assigned to new vertices.
            The list length of input values must equals num_dup.
            For example, when num=2, depth_factors will be [0.5, 0.5] by default.
            User input values will NOT be normalized.
        :return: duplicated_vertices_groups = [{label_1__copy2: label_1, label_1: label_1__copy2},
                                               {label_1__copy4: label_1, label_1: label_1__copy4}]
        """
        assert num_dup >= 1
        if not depth_factors:
            depth_factors = [1. / num_dup]
        assert len(depth_factors) == num_dup, "The list length of depth_factors must equals num_dup!"
        # deepcopy *
        duplicated_vertices_groups = []
        go_copy_postfix_offset = {__v_name: 0 for __v_name in vertices}
        for dup_id in range(1, num_dup):
            # create a new_sub_graph, rename all vertices inside this new_sub_graph,
            # transfer vertices in the new_sub_graph back to the original graph.
            # This design makes duplication/renaming easier.
            # 1. create a new_sub_graph and assign the depths to vertices
            new_sub_graph = Assembly(overlap=self.__overlap)
            for v_name in vertices:
                assert v_name in self.vertex_info, "Vertex {} not found in the graph!".format(v_name)
                new_sub_graph.vertex_info[v_name] = deepcopy(self.vertex_info[v_name])
                new_sub_graph.vertex_info[v_name].cov *= depth_factors[dup_id]
            # 2. rename all vertices and connections
            name_translator = {}
            reverse_translator = {}
            for v_name in vertices:
                proposed_name = "{}__copy{}".format(v_name, 1 + dup_id + go_copy_postfix_offset[v_name])
                while proposed_name in self.vertex_info:
                    go_copy_postfix_offset[v_name] += 1
                    proposed_name = "{}__copy{}".format(v_name, 1 + dup_id + go_copy_postfix_offset[v_name])
                name_translator[v_name] = proposed_name
                reverse_translator[proposed_name] = v_name
            new_sub_graph = new_sub_graph.new_graph_with_vertices_renamed(name_translate_dict=name_translator)
            # 3. transfer the renamed graph component back to the original graph
            for new_name in new_sub_graph.vertex_info:
                self.vertex_info[new_name] = new_sub_graph.vertex_info[new_name]
                # create connections between the original graph and the renamed sub graph
                for this_end in (True, False):
                    for next_n, next_e in self.vertex_info[new_name].connections[this_end]:
                        if next_n not in new_sub_graph.vertex_info:
                            self.vertex_info[next_n].connections[next_e][(new_name, this_end)] = \
                                self.vertex_info[next_n].connections[next_e][(reverse_translator[new_name], this_end)]
            del new_sub_graph
            name_translator.update(reverse_translator)
            duplicated_vertices_groups.append(name_translator)
        # modify the depth of the original copy
        raw_translator = {}
        for v_name in vertices:
            self.vertex_info[v_name].cov *= depth_factors[0]
            raw_translator[v_name] = v_name
        duplicated_vertices_groups.insert(0, raw_translator)
        return duplicated_vertices_groups

    def is_no_leaking_path(self, path, terminal_pairs):
        path_set = set([v_n for v_n, v_e in path])
        left_v_e_set = set()
        right_v_e_set = set()
        for terminal_pair in terminal_pairs:
            if terminal_pair[0][0] in path_set:
                return False
            if terminal_pair[1][0] in path_set:
                return False
            left_v_e_set.add(terminal_pair[0])
            right_v_e_set.add((terminal_pair[1][0], not terminal_pair[1][1]))
        for v_name, v_end in path[:-1]:
            for next_name, next_end in self.vertex_info[v_name].connections[v_end]:
                if (next_name, next_end) not in right_v_e_set and next_name not in path_set:
                    return False
        for v_name, v_end in path[1:]:
            for next_name, next_end in self.vertex_info[v_name].connections[not v_end]:
                if (next_name, next_end) not in left_v_e_set and next_name not in path_set:
                    return False
        return True

    def find_the_path_containing_pair(self, left_v_e, terminating_end_set, starting_end_set):
        start_v, start_e = left_v_e
        in_pipe_leak = False
        circle_in_between = []
        checked_vertex_ends = set()
        checked_vertex_ends.add((start_v, start_e))
        searching_con = [(start_v, not start_e)]
        while searching_con:
            in_search_v, in_search_e = searching_con.pop(0)
            if (in_search_v, in_search_e) in terminating_end_set:
                # start from the same (next_t_v, next_t_e), merging to two different ends of connection_set_f
                if circle_in_between:
                    in_pipe_leak = True
                    break
                else:
                    circle_in_between.append(((start_v, start_e), (in_search_v, in_search_e)))
            elif (in_search_v, in_search_e) in starting_end_set:
                in_pipe_leak = True
                break
            else:
                for n_in_search_v, n_in_search_e in self.vertex_info[in_search_v].connections[in_search_e]:
                    if (n_in_search_v, n_in_search_e) in checked_vertex_ends:
                        pass
                    else:
                        checked_vertex_ends.add((n_in_search_v, n_in_search_e))
                        searching_con.append((n_in_search_v, not n_in_search_e))
        if not in_pipe_leak:
            return circle_in_between
        else:
            return []

    def is_sequential_repeat(self, search_vertex_name, return_pair_in_the_trunk_path=True):

        if search_vertex_name not in self.vertex_info:
            raise ProcessingGraphFailed("Vertex name " + search_vertex_name + " not found!")
        connection_set_t = self.vertex_info[search_vertex_name].connections[True]
        connection_set_f = self.vertex_info[search_vertex_name].connections[False]
        all_pairs_of_inner_circles = []

        # branching ends
        if len(connection_set_t) == len(connection_set_f) == 2:
            for next_t_v_e in list(connection_set_t):
                this_inner_circle = self.find_the_path_containing_pair(next_t_v_e, connection_set_f, connection_set_t)
                if this_inner_circle:
                    # check leakage in reverse direction
                    reverse_v_e = this_inner_circle[0][1]
                    not_leak = self.find_the_path_containing_pair(reverse_v_e, connection_set_t, connection_set_f)
                    if not_leak:
                        all_pairs_of_inner_circles.extend(this_inner_circle)
            # sort pairs by average depths(?)
            all_pairs_of_inner_circles.sort(
                key=lambda x: (self.vertex_info[x[0][0]].cov + self.vertex_info[x[1][0]].cov))
            if all_pairs_of_inner_circles and return_pair_in_the_trunk_path:
                # switch nearby vertices
                # keep those prone to be located in the "trunk road" of the repeat
                single_pair_in_main_path = []
                if len(all_pairs_of_inner_circles) == 1:
                    for next_v, next_e in list(connection_set_t) + list(connection_set_f):
                        if (next_v, next_e) not in all_pairs_of_inner_circles[0]:
                            single_pair_in_main_path.append((next_v, next_e))
                    single_pair_in_main_path = tuple(single_pair_in_main_path)
                else:
                    # two circles share this sequential repeat,
                    # return the one with a smaller average depth(?)
                    single_pair_in_main_path = tuple(all_pairs_of_inner_circles[0])
                return single_pair_in_main_path
            return all_pairs_of_inner_circles
        else:
            return all_pairs_of_inner_circles

    def merge_all_possible_vertices(self, limited_vertices=None, copy_tags=True):
        """
        Merges all or a subset of vertices...
        Called in several places...
        """
        # select all or a subset of vertices and sort
        if not limited_vertices:
            limited_vertices = sorted(self.vertex_info)
        else:
            limited_vertices = sorted(limited_vertices)

        # initially merged is False and overlap in True or False
        merged = False
        overlap = (self.__overlap if self.__overlap else 0)

        # iterate over the sorted list of vertices popping items 
        while limited_vertices:

            # select and remove this vertex from list
            this_vertex = limited_vertices.pop()

            # iterate over both ends
            for this_end in (True, False):

                # get the connections to this end of the vertex
                connected_dict = self.vertex_info[this_vertex].connections[this_end]

                # if 1 connection do this, ... currently no operation for 0 or >1...
                if len(connected_dict) == 1:
                    
                    # select first connected vertex
                    next_vertex, next_end = list(connected_dict)[0]

                    # ...
                    if len(self.vertex_info[next_vertex].connections[next_end]) == 1 and this_vertex != next_vertex:
                        # reverse the names
                        merged = True
                        self.vertex_info[this_vertex].merging_history.add((next_vertex, not next_end),
                                                                          add_new_to_front=this_end,
                                                                          reverse_the_latter=not next_end)
                        new_vertex = str(self.vertex_info[this_vertex].merging_history)

                        limited_vertices.remove(next_vertex)
                        limited_vertices.append(new_vertex)
                        
                        # initialization
                        self.vertex_info[new_vertex] = deepcopy(self.vertex_info[this_vertex])
                        self.vertex_info[new_vertex].label = new_vertex
                        self.vertex_info[new_vertex].fastg_form_name = None
                        
                        # modify connections
                        self.vertex_info[new_vertex].connections[this_end] = (
                            deepcopy(self.vertex_info[next_vertex].connections[not next_end]))

                        if (this_vertex, not this_end) in self.vertex_info[new_vertex].connections[this_end]:
                            # forms a circle
                            del self.vertex_info[new_vertex].connections[this_end][(this_vertex, not this_end)]
                            self.vertex_info[new_vertex].connections[this_end][(new_vertex, not this_end)] = None
                        for new_end in (True, False):
                            for n_n_v, n_n_e in self.vertex_info[new_vertex].connections[new_end]:
                                self.vertex_info[n_n_v].connections[n_n_e][(new_vertex, new_end)] = None
                        
                        # len & cov
                        this_len = self.vertex_info[this_vertex].len
                        next_len = self.vertex_info[next_vertex].len
                        this_cov = self.vertex_info[this_vertex].cov
                        next_cov = self.vertex_info[next_vertex].cov
                        self.vertex_info[new_vertex].len = this_len + next_len - overlap
                        self.vertex_info[new_vertex].cov = \
                            ((this_len - overlap + 1) * this_cov + (next_len - overlap + 1) * next_cov) \
                            / ((this_len - overlap + 1) + (next_len - overlap + 1))
                        self.vertex_info[new_vertex].seq[this_end] \
                            += self.vertex_info[next_vertex].seq[not next_end][overlap:]
                        self.vertex_info[new_vertex].seq[not this_end] \
                            = self.vertex_info[next_vertex].seq[next_end][:next_len - overlap] \
                              + self.vertex_info[this_vertex].seq[not this_end]
                        
                        # tags
                        if copy_tags:
                            if "tags" in self.vertex_info[next_vertex].other_attr:
                                if "tags" not in self.vertex_info[new_vertex].other_attr:
                                    self.vertex_info[new_vertex].other_attr["tags"] = \
                                        deepcopy(self.vertex_info[next_vertex].other_attr["tags"])
                                else:
                                    for db_n in self.vertex_info[next_vertex].other_attr["tags"]:
                                        if db_n not in self.vertex_info[new_vertex].other_attr["tags"]:
                                            self.vertex_info[new_vertex].other_attr["tags"][db_n] \
                                                = deepcopy(self.vertex_info[next_vertex].other_attr["tags"][db_n])
                                        else:
                                            self.vertex_info[new_vertex].other_attr["tags"][db_n] \
                                                |= self.vertex_info[next_vertex].other_attr["tags"][db_n]
                            if "weight" in self.vertex_info[next_vertex].other_attr:
                                if "weight" not in self.vertex_info[new_vertex].other_attr:
                                    self.vertex_info[new_vertex].other_attr["weight"] \
                                        = deepcopy(self.vertex_info[next_vertex].other_attr["weight"])
                                else:
                                    for db_n in self.vertex_info[next_vertex].other_attr["weight"]:
                                        if db_n not in self.vertex_info[new_vertex].other_attr["weight"]:
                                            self.vertex_info[new_vertex].other_attr["weight"][db_n] \
                                                = self.vertex_info[next_vertex].other_attr["weight"][db_n]
                                        else:
                                            self.vertex_info[new_vertex].other_attr["weight"][db_n] \
                                                += self.vertex_info[next_vertex].other_attr["weight"][db_n]
                            for db_n in self.tagged_vertices:
                                if this_vertex in self.tagged_vertices[db_n]:
                                    self.tagged_vertices[db_n].add(new_vertex)
                                    self.tagged_vertices[db_n].remove(this_vertex)
                                if next_vertex in self.tagged_vertices[db_n]:
                                    self.tagged_vertices[db_n].add(new_vertex)
                                    self.tagged_vertices[db_n].remove(next_vertex)
                        
                        # ...
                        self.remove_vertex([this_vertex, next_vertex], update_cluster=False)
                        break

        # update the clusters now that some vertices have been merged.
        self.update_vertex_clusters()

        # return boolean of whether anything was merged.
        return merged

    def detect_palindromic_repeats(self, redo=True):
        if not redo and self.palindromic_repeats:
            self.__palindromic_repeat_detected = True
            return True
        else:
            self.palindromic_repeats = set()
            find_palindromic_repeats = False
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
                        find_palindromic_repeats = True
                        if len(forward_c) == len(reverse_c) == 2:  # simple palindromic repeats, prune repeated connections
                            for go_d, (nb_vertex, nb_direction) in enumerate(tuple(forward_c)):
                                del self.vertex_info[nb_vertex].connections[nb_direction][(vertex_n, bool(go_d))]
                                del self.vertex_info[vertex_n].connections[bool(go_d)][(nb_vertex, nb_direction)]
                        elif len(forward_c) == len(reverse_c) == 1:  # connect to the same inverted repeat
                            pass
                        else:  # complicated, recorded
                            self.palindromic_repeats.add(vertex_n)
            self.__palindromic_repeat_detected = True
            return find_palindromic_repeats

    def correct_path_with_palindromic_repeats(self, input_path):
        # detect palindromic_repeats if not detected yet
        if not self.__palindromic_repeat_detected:
            self.detect_palindromic_repeats()
        # if there are palindromic repeats, correct the palindromic node direction into True
        if self.palindromic_repeats:
            corrected_path = tuple([(this_v, True) if this_v in self.palindromic_repeats else (this_v, this_e)
                                    for this_v, this_e in input_path])
        else:
            corrected_path = tuple(input_path)
        return corrected_path

    def estimate_multiplicity_by_cov(
            self,
            limited_vertices=None,
            given_average_cov=None,
            mode="embplant_pt",
            re_initialize=False):
        """
        Use seq coverage data to estimate copy and depth.
        :param limited_vertices: vertex scope, default: all vertices
        :param given_average_cov: user-defined average depth
        :param mode: genome type
        :param re_initialize: reinitialize
        """
        EstMultiplicityFromCov(graph=self,
                               verts=limited_vertices,
                               avgcov=given_average_cov,
                               mode=mode,
                               reinit=re_initialize).run()

    def estimate_multiplicity_precisely(
            self,
            # ave_depth,
            maximum_copy_num=8,
            broken_graph_allowed=False,
            return_new_graphs=False,
            target_name_for_log="target",
            debug=False):
        """

        :param maximum_copy_num:
        :param broken_graph_allowed:
        :param return_new_graphs: return result if True else record res in current graph obj
        :param target_name_for_log: str
        :param debug: pass to scipy.optimize.minimize etc.
        :return:
        """
        res = EstMultiplicityPrecise(
            graph=self,
            # ave_depth=ave_depth,
            maximum_copy_num=maximum_copy_num,
            broken_graph_allowed=broken_graph_allowed,
            return_new_graphs=return_new_graphs,
            label=target_name_for_log,
            debug=debug).run()
        if return_new_graphs:
            return res

    def tag_in_between(self, database_n):
        # add those in between the tagged vertices to tagged_vertices, which offered the only connection
        updated = True
        candidate_vertices = list(self.vertex_info)
        while updated:
            updated = False
            go_to_v = 0
            while go_to_v < len(candidate_vertices):
                can_v = candidate_vertices[go_to_v]
                if can_v in self.tagged_vertices[database_n]:
                    del candidate_vertices[go_to_v]
                    continue
                else:
                    if sum([bool(c_c) for c_c in self.vertex_info[can_v].connections.values()]) != 2:
                        del candidate_vertices[go_to_v]
                        continue
                    count_nearby_tagged = []
                    for can_end in (True, False):
                        for next_v, next_e in self.vertex_info[can_v].connections[can_end]:
                            # candidate_v is the only output vertex to next_v
                            if next_v in self.tagged_vertices[database_n] and \
                                    len(self.vertex_info[next_v].connections[next_e]) == 1:
                                count_nearby_tagged.append((next_v, next_e))
                                break
                    if len(count_nearby_tagged) == 2:
                        del candidate_vertices[go_to_v]
                        # add in between
                        self.tagged_vertices[database_n].add(can_v)
                        if "weight" not in self.vertex_info[can_v].other_attr:
                            self.vertex_info[can_v].other_attr["weight"] = {}
                        if database_n not in self.vertex_info[can_v].other_attr["weight"]:
                            self.vertex_info[can_v].other_attr["weight"][database_n] = 0
                        self.vertex_info[can_v].other_attr["weight"][database_n] += 1 * self.vertex_info[can_v].cov
                        if database_n != "embplant_mt":
                            # Adding extra circle - the contig in-between the sequential repeats
                            # To avoid risk of tagging mt as pt by mistake,
                            # the repeated contig must be at least 2 folds of the nearby tagged contigs
                            near_by_pairs = self.is_sequential_repeat(can_v, return_pair_in_the_trunk_path=False)
                            if near_by_pairs:
                                checking_new = []
                                coverage_folds = []
                                for near_by_p in near_by_pairs:
                                    for (near_v, near_e) in near_by_p:
                                        if (near_v, near_e) not in count_nearby_tagged:
                                            checking_new.append(near_v)
                                            # comment out for improper design: if the untagged is mt
                                            # coverage_folds.append(
                                            #     round(self.vertex_info[can_v].cov /
                                            #           self.vertex_info[near_v].cov, 0))
                                for near_v, near_e in count_nearby_tagged:
                                    coverage_folds.append(
                                        round(self.vertex_info[can_v].cov /
                                              self.vertex_info[near_v].cov, 0))
                                # if coverage folds is
                                if max(coverage_folds) >= 2:
                                    for extra_v_to_add in set(checking_new):
                                        self.tagged_vertices[database_n].add(extra_v_to_add)
                                        try:
                                            candidate_vertices.remove(extra_v_to_add)
                                        except ValueError:
                                            pass
                                        # when a contig has no weights
                                        if "weight" not in self.vertex_info[extra_v_to_add].other_attr:
                                            self.vertex_info[extra_v_to_add].other_attr["weight"] = {database_n: 0}
                                        # when a contig has weights of other database
                                        if database_n not in self.vertex_info[extra_v_to_add].other_attr["weight"]:
                                            self.vertex_info[extra_v_to_add].other_attr["weight"][database_n] = 0
                                        self.vertex_info[extra_v_to_add].other_attr["weight"][database_n] \
                                            += 1 * self.vertex_info[extra_v_to_add].cov
                        updated = True
                        break
                    else:
                        go_to_v += 1

    def parse_tab_file(self, tab_file, database_name, type_factor, log_handler=None):
        # parse_csv, every locus only occur in one vertex (removing locations with smaller weight)
        tag_loci = {}
        tab_matrix = [line.strip("\n").split("\t") for line in open(tab_file)][1:]
        for node_record in tab_matrix:
            vertex_name = node_record[0]
            if vertex_name in self.vertex_info:
                matched = node_record[5].split(">>")
                for locus in matched:
                    if "(" in locus:
                        locus_spl = locus.split("(")
                        locus_type = locus_spl[-1].split(",")[1][:-1]
                        if locus_type not in tag_loci:
                            tag_loci[locus_type] = {}
                        locus_name = "(".join(locus_spl[:-1])
                        locus_start, locus_end = locus_spl[-1].split(",")[0].split("-")
                        locus_start, locus_end = int(locus_start), int(locus_end)
                        locus_len = locus_end - locus_start + 1
                        # skip those tags concerning only the overlapping sites
                        if (locus_start == 1 or locus_end == self.vertex_info[vertex_name].len) \
                                and locus_len == self.__overlap:
                            continue
                        if locus_name in tag_loci[locus_type]:
                            new_weight = locus_len * self.vertex_info[vertex_name].cov
                            if new_weight > tag_loci[locus_type][locus_name]["weight"]:
                                tag_loci[locus_type][locus_name] = {"vertex": vertex_name, "len": locus_len,
                                                                    "weight": new_weight}
                        else:
                            tag_loci[locus_type][locus_name] = {"vertex": vertex_name, "len": locus_len,
                                                                "weight": locus_len * self.vertex_info[vertex_name].cov}

        for locus_type in tag_loci:
            self.tagged_vertices[locus_type] = set()
            for locus_name in tag_loci[locus_type]:
                vertex_name = tag_loci[locus_type][locus_name]["vertex"]
                loci_weight = tag_loci[locus_type][locus_name]["weight"]
                # tags
                if "tags" not in self.vertex_info[vertex_name].other_attr:
                    self.vertex_info[vertex_name].other_attr["tags"] = {}
                if locus_type in self.vertex_info[vertex_name].other_attr["tags"]:
                    self.vertex_info[vertex_name].other_attr["tags"][locus_type].add(locus_name)
                else:
                    self.vertex_info[vertex_name].other_attr["tags"][locus_type] = {locus_name}
                # weight
                if "weight" not in self.vertex_info[vertex_name].other_attr:
                    self.vertex_info[vertex_name].other_attr["weight"] = {}
                if locus_type in self.vertex_info[vertex_name].other_attr["weight"]:
                    self.vertex_info[vertex_name].other_attr["weight"][locus_type] += loci_weight
                else:
                    self.vertex_info[vertex_name].other_attr["weight"][locus_type] = loci_weight
                self.tagged_vertices[locus_type].add(vertex_name)

        for vertex_name in self.vertex_info:
            if "weight" in self.vertex_info[vertex_name].other_attr:
                if len(self.vertex_info[vertex_name].other_attr["weight"]) > 1:
                    all_weights = sorted([(loc_type, self.vertex_info[vertex_name].other_attr["weight"][loc_type])
                                          for loc_type in self.vertex_info[vertex_name].other_attr["weight"]],
                                         key=lambda x: -x[1])
                    best_t, best_w = all_weights[0]
                    for next_t, next_w in all_weights[1:]:
                        if next_w * type_factor < best_w:
                            self.tagged_vertices[next_t].remove(vertex_name)

        if database_name not in self.tagged_vertices or len(self.tagged_vertices[database_name]) == 0:
            raise ProcessingGraphFailed("No available " + database_name + " information found in " + tab_file)

    def exclude_other_hits(self, database_n):
        vertices_to_exclude = []
        for vertex_name in self.vertex_info:
            if "tags" in self.vertex_info[vertex_name].other_attr:
                if database_n in self.vertex_info[vertex_name].other_attr["tags"]:
                    pass
                elif self.vertex_info[vertex_name].other_attr["tags"]:
                    vertices_to_exclude.append(vertex_name)
        if vertices_to_exclude:
            self.remove_vertex(vertices_to_exclude)
            return True
        else:
            return False

    # TODO problematic
    def reduce_to_subgraph(self, bait_vertices, bait_offsets=None,  limit_extending_len=None,  extending_len_weighted_by_depth=False):
        """
        :param bait_vertices:
        :param bait_offsets:
        :param limit_extending_len:
        :param limit_offset_current_vertex:
        :param extending_len_weighted_by_depth:
        :return:
        """
        if bait_offsets is None:
            bait_offsets = {}
        rm_contigs = set()
        rm_sub_ids = []
        overlap = self.__overlap if self.__overlap else 0
        for go_sub, vertices in enumerate(self.vertex_clusters):
            for vertex in sorted(vertices):
                if vertex in bait_vertices:
                    break
            else:
                rm_sub_ids.append(go_sub)
                rm_contigs.update(vertices)
        # rm vertices
        self.remove_vertex(rm_contigs, update_cluster=False)
        # rm clusters
        for sub_id in rm_sub_ids[::-1]:
            del self.vertex_clusters[sub_id]
        # searching within a certain length scope
        if limit_extending_len not in (None, INF):
            if extending_len_weighted_by_depth:
                explorers = {(v_n, v_e): (limit_extending_len - bait_offsets.get((v_n, v_e), 0),
                                          self.vertex_info[v_n].cov)
                             for v_n in set(bait_vertices)
                             for v_e in (True, False)}
                best_explored_record = {}
                # explore all minimum distances starting from the bait_vertices
                while True:
                    changed = False
                    for (this_v, this_e), (quota_len, base_cov) in sorted(explorers.items()):
                        # if there's any this_v active: quota_len>0 AND (not_recorded OR recorded_changed))
                        if quota_len > 0 and \
                                (quota_len, base_cov) != best_explored_record.get((this_v, this_e), 0):
                            changed = True
                            best_explored_record[(this_v, this_e)] = (quota_len, base_cov)
                            for next_v, next_e in self.vertex_info[this_v].connections[this_e]:
                                # not the starting vertices
                                if next_v not in bait_vertices:
                                    new_quota_len = quota_len - (self.vertex_info[next_v].len - overlap) * \
                                                    max(1, self.vertex_info[next_v].cov / base_cov)
                                    # if next_v is active: quota_len>0 AND (not_explored OR larger_len))
                                    next_p = (next_v, not next_e)
                                    if new_quota_len > 0 and \
                                            (next_p not in explorers or
                                             # follow the bait contigs with higher coverage:
                                             # replace new_quota_len > explorers[next_p][0]): with
                                             new_quota_len * base_cov > explorers[next_p][0] * explorers[next_p][1]):
                                        explorers[next_p] = (new_quota_len, base_cov)
                    if not changed:
                        break  # if no this_v active, stop the exploring
            else:
                explorers = {(v_n, v_e): limit_extending_len - bait_offsets.get((v_n, v_e), 0)
                             for v_n in set(bait_vertices)
                             for v_e in (True, False)}
                best_explored_record = {}
                # explore all minimum distances starting from the bait_vertices
                while True:
                    changed = False
                    for (this_v, this_e), quota_len in sorted(explorers.items()):
                        # if there's any this_v active: quota_len>0 AND (not_recorded OR recorded_changed))
                        if quota_len > 0 and quota_len != best_explored_record.get((this_v, this_e), None):
                            changed = True
                            best_explored_record[(this_v, this_e)] = quota_len
                            # for this_direction in (True, False):
                            for next_v, next_e in self.vertex_info[this_v].connections[this_e]:
                                # not the starting vertices
                                if next_v not in bait_vertices:
                                    new_quota_len = quota_len - (self.vertex_info[next_v].len - overlap)
                                    # if next_v is active: quota_len>0 AND (not_explored OR larger_len))
                                    next_p = (next_v, not next_e)
                                    if new_quota_len > explorers.get(next_p, 0):
                                        explorers[next_p] = new_quota_len
                    if not changed:
                        break  # if no this_v active, stop the exploring
            accepted = {candidate_v for (candidate_v, candidate_e) in explorers}
            rm_contigs = {candidate_v for candidate_v in self.vertex_info if candidate_v not in accepted}
            self.remove_vertex(rm_contigs, update_cluster=True)

    def generate_heuristic_components(
            self,
            graph_alignment,
            random_obj,
            num_search,
            num_processes=1,
            force_circular=True,
            hetero_chromosome=True):
        """
        :param graph_alignment:
        :param random_obj: random
            passed from traversome.random [or from import random]
        :param num_search
        :param num_processes
        :param force_circular
        :param hetero_chromosome
        """
        generator = PathGeneratorGraphAlignment(
            assembly_graph=self,
            graph_alignment=graph_alignment,
            num_search=num_search,
            num_processes=num_processes,
            force_circular=force_circular,
            hetero_chromosome=hetero_chromosome,
            random_obj=random_obj)
        generator.generate_heuristic_components()
        return generator.components

    def find_all_circular_isomers(self, mode="embplant_pt", re_estimate_multiplicity=False):
        """
        :param mode:
        :param library_info: not used currently
        :return: sorted_paths
        """
        generator = PathGeneratorGraphOnly(
            graph=self,
            mode=mode,
            re_estimate_multiplicity=re_estimate_multiplicity)
        generator.find_all_circular_isomers()
        return generator.components

    def find_all_isomers(self, mode="embplant_pt"):
        """
        :param mode:
        :return: sorted_paths
        """
        generator = PathGeneratorGraphOnly(graph=self, mode=mode)
        generator.find_all_isomers()
        return generator.components

    def is_circular_path(self, input_path):
        return (input_path[0][0], not input_path[0][1]) in \
               self.vertex_info[input_path[-1][0]].connections[input_path[-1][1]]

    def is_fully_covered_by(self, input_path):
        graph_set = set(self.vertex_info)
        path_set = set([_n_ for _n_, _e_ in input_path])
        logger.trace("{}vs - {}vs = {}".format(
            len(graph_set), len(path_set), sorted(set(self.vertex_info) - set([_n_ for _n_, _e_ in input_path]))))
        return graph_set == path_set

    def get_path_length(self, input_path):
        overlap = self.__overlap if self.__overlap else 0
        circular_len = sum([self.vertex_info[name].len - overlap for name, strand in input_path])
        return circular_len + overlap * int(self.is_circular_path(input_path))

    def get_path_internal_length(self, input_path):
        assert len(input_path) > 1, f"input path len cannot be <= 1, this path is {input_path}"
        overlap = self.__overlap if self.__overlap else 0
        # internal_len is allowed to be negative when this_overlap > 0 and len(the_repeat_path) == 2
        internal_len = -overlap
        for seg_name, seg_strand in input_path[1:-1]:
            internal_len += self.vertex_info[seg_name].len - overlap
        return internal_len

    def get_path_len_without_terminal_overlaps(self, input_path):
        assert len(input_path) > 1
        overlap = self.__overlap if self.__overlap else 0
        path_len = -overlap
        for seg_name, seg_strand in input_path:
            path_len += self.vertex_info[seg_name].len - overlap
        return path_len

    def repr_path(self, in_path):
        # Bandage style
        seq_names = []
        for this_vertex, this_end in in_path:
            seq_names.append(this_vertex + ("-", "+")[this_end])
        if self.is_circular_path(in_path):
            seq_names[-1] += "(circular)"
        return ",".join(seq_names)

    def export_path(self, in_path):
        overlap = self.__overlap if self.__overlap else 0
        seq_segments = []
        for this_vertex, this_end in in_path:
            seq_segments.append(self.vertex_info[this_vertex].seq[this_end][overlap:])
        if not self.is_circular_path(in_path):
            seq_segments[0] = self.vertex_info[in_path[0][0]].seq[in_path[0][1]][:overlap] + seq_segments[0]
        return Sequence(self.repr_path(in_path), "".join(seq_segments))

    def reverse_path(self, raw_path):
        tuple_path = tuple(raw_path)
        if tuple_path in self.__reverse_paths:
            return self.__reverse_paths[tuple_path]
        else:
            # if there are palindromic repeats, correct the palindromic node direction into True
            if not self.__palindromic_repeat_detected:
                self.detect_palindromic_repeats()
            if self.__record_reversed_paths_to_mem:
                if self.palindromic_repeats:
                    reverse_path = [(this_v, True) if this_v in self.palindromic_repeats else (this_v, not this_e)
                                    for this_v, this_e in raw_path[::-1]]
                else:
                    reverse_path = [(this_v, not this_e) for this_v, this_e in raw_path[::-1]]
                tuple_rev = tuple(reverse_path)
                self.__reverse_paths[tuple_path] = tuple_rev
                self.__reverse_paths[tuple_rev] = tuple_path
                return tuple_rev
            else:
                if self.palindromic_repeats:
                    return tuple([(this_v, True) if this_v in self.palindromic_repeats else (this_v, not this_e)
                                  for this_v, this_e in raw_path[::-1]])
                else:
                    return tuple([(this_v, not this_e) for this_v, this_e in raw_path[::-1]])

    def contain_path(self, input_path):
        assert len(input_path) > 0
        go_v = 0
        v_name, v_end = input_path[go_v]
        # initialize last_connection
        last_connection = {(v_name, not v_end)}
        while go_v < len(input_path):
            v_name, v_end = input_path[go_v]
            if v_name not in self.vertex_info:
                return False
            elif (v_name, not v_end) not in last_connection:
                return False
            else:
                last_connection = self.vertex_info[v_name].connections[v_end]
                go_v += 1
        return True

    def roll_path(self, input_path):
        # detect if the_repeat_path could be rolled into repeat_unit
        # e.g.
        # (2, 3, 4, 1, 2) can be rolled into (1, 2, 3, 4)
        # (1, 2, 1, 2, 1) can be rolled into (1, 2)
        corrected_path = self.correct_path_with_palindromic_repeats(input_path)
        go_v = 1
        start_v_e = corrected_path[0]
        len_path = len(corrected_path)
        while go_v < len_path:
            if corrected_path[go_v:].count(start_v_e) > 0:
                next_go = corrected_path.index(start_v_e, go_v)
                repeat_unit = corrected_path[:next_go]
                hypothesized_path = repeat_unit * int(len_path / next_go) + repeat_unit[:(len_path % next_go)]
                if tuple(hypothesized_path) == tuple(corrected_path):
                    return corrected_path[:next_go]
                else:
                    go_v = next_go + 1
            else:
                break
        return deepcopy(corrected_path)

    # separate get_standardized_path and get_standardized_circular_path
    # because get_standardized_path will may be called by many thousands even millions of times
    def get_standardized_path(self, raw_path):
        """
        standardized for comparing and identify unique path
        :param raw_path: path=[(name1:str, direction1:bool), (name2:str, direction2:bool), ..]
        :return: standardized_path
        """
        forward_path = self.correct_path_with_palindromic_repeats(raw_path)
        reverse_path = self.reverse_path(forward_path)
        return sorted([forward_path, reverse_path])[0]

    def get_standardized_circular_path(self, raw_path):
        """
        standardized for comparing and identify unique path
        :param raw_path: path=[(name1:str, direction1:bool), (name2:str, direction2:bool), ..]
        :return: standardized_path
        """
        forward_path = list(self.correct_path_with_palindromic_repeats(raw_path))
        reverse_path = list(self.reverse_path(forward_path))

        if self.is_circular_path(forward_path):
            # if path is circular, try all start points
            iso_paths = [forward_path, reverse_path]
            for change_start in range(1, len(forward_path)):
                iso_paths.append(forward_path[change_start:] + forward_path[:change_start])
                iso_paths.append(reverse_path[change_start:] + reverse_path[:change_start])
            return tuple(sorted(iso_paths)[0])
        else:
            return tuple(sorted([forward_path, reverse_path])[0])

    def get_standardized_path_with_strand(self, raw_path, detect_circular):
        """
        standardized for comparing and identify unique path
        :param raw_path: path=[(name1:str, direction1:bool), (name2:str, direction2:bool), ..]
        :param detect_circular: treat circular path as a special case
        :return: standardized_path, strand_of_the_new_path
        """
        forward_path = list(self.correct_path_with_palindromic_repeats(raw_path))
        reverse_path = list(self.reverse_path(forward_path))

        if detect_circular and self.is_circular_path(forward_path):
            # if path is circular, try all start points
            iso_paths = [forward_path, reverse_path]
            for change_start in range(1, len(forward_path)):
                iso_paths.append(forward_path[change_start:] + forward_path[:change_start])
                iso_paths.append(reverse_path[change_start:] + reverse_path[:change_start])
            standard_id = sorted(range(len(iso_paths)), key=lambda x: iso_paths[x])[0]
            return tuple(iso_paths[standard_id]), standard_id % 2 == 0
        else:
            standard_id = sorted([0, 1], key=lambda x: [forward_path, reverse_path][x])[0]
            return tuple([forward_path, reverse_path][standard_id]), standard_id == 0

    def get_standardized_isomer(self, isomer_raw_paths):
        """
        standardized for comparing and identify unique isomer, similar to self.get_standardized_path()
        :param raw_path: isomer=[path, path, path]
        :return: isomer_with_only_palindromic_corrected, standardized_isomer
        """

        # detect palindromic_repeats if not detected yet
        if not self.__palindromic_repeat_detected:
            self.detect_palindromic_repeats()
        # if there are palindromic repeats, correct the palindromic node direction into True
        if self.palindromic_repeats:
            corrected_isomer = [
                [(this_v, True) if this_v in self.palindromic_repeats
                 else (this_v, this_e) for this_v, this_e in path_part]
                for path_part in isomer_raw_paths
            ]
        else:
            corrected_isomer = deepcopy(isomer_raw_paths)

        # ...
        here_standardized_isomer = []
        for part_path in corrected_isomer:

            # ...
            rev_part = self.reverse_path(part_path)

            # ...
            if self.is_circular_path(part_path):
                # circular
                this_part_derived = [part_path, rev_part]
                for change_start in range(1, len(part_path)):
                    this_part_derived.append(part_path[change_start:] + part_path[:change_start])
                    this_part_derived.append(rev_part[change_start:] + rev_part[:change_start])
                standard_part = tuple(sorted(this_part_derived)[0])
            else:
                standard_part = tuple(sorted([part_path, rev_part])[0])

            # store this part in the path
            here_standardized_isomer.append(standard_part)

        return corrected_isomer, tuple(sorted(here_standardized_isomer))  # , key=lambda x: smart_trans_for_sort(x)

    def get_num_of_possible_alignment_start_points(self, read_len, align_to_path, path_internal_len):
        """
        If a read with certain length (i.e. median length of all candidate reads) could be aligned to a path,
        calculate how many possible start points could this alignment happen.

        Example:
        ----------------------------------------
        |      \                               |
        |     b \          e          / a      |
        |        \___________________/         |
        |        /                   \         |
        |     c /                     \ d      |
        |      /                       \       |
        |     /                         \      |
        |                                \     |
        ----------------------------------------
        for graph(a=2,b=3,c=4,d=5,e=6), if read has length of 11 and be aligned to b->e->d,
        then there could be 3 possible alignment start points

        :param read_len: we use median read length to approximate
        :param align_to_path:
        :param path_internal_len:
        :return:
        """
        overlap = self.__overlap if self.__overlap else 0
        maximum_num_cat = read_len - path_internal_len - 2
        left_trim = max(maximum_num_cat - self.vertex_info[align_to_path[0][0]].len - overlap, 0)
        right_trim = max(maximum_num_cat - self.vertex_info[align_to_path[-1][0]].len - overlap, 0)
        return maximum_num_cat - left_trim - right_trim

    def get_branching_ends(self):
        branching_ends = set()
        for v_name in self.vertex_info:
            for v_end in (True, False):
                if len(self.vertex_info[v_name].connections[v_end]) >= 2:
                    branching_ends.add((v_name, v_end))
        return branching_ends






