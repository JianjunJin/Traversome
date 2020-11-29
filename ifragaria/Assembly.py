#!/usr/bin/env python

"""
Assembly class object and associated class objects
"""

from loguru import logger
from .SimpleAssembly import SimpleAssembly
from .GetAllCircularPaths import GetAllCircularPaths
from .GetAllSortedPaths import GetAllPaths
from .utils import Sequence, SequenceList, ProcessingGraphFailed, INF, smart_trans_for_sort, get_orf_lengths
from copy import deepcopy
from collections import OrderedDict
from itertools import combinations, product
import os
import sys



class Assembly(SimpleAssembly):
    """
    Main class object for storing and 
    """
    def __init__(self, graph_file=None, min_cov=0., max_cov=INF, overlap=None):
        """
        :param graph_file:
        :param min_cov:
        :param max_cov:
        """
        # inherit values from base class
        super(Assembly, self).__init__(graph_file=graph_file, min_cov=min_cov, max_cov=max_cov, overlap=overlap)
        self.__overlap = super(Assembly, self).overlap()

        # get an initial set of clusters of connected vertices
        self.vertex_clusters = []
        self.update_vertex_clusters()

        # destinations to fill, what are these?
        self.tagged_vertices = {}
        self.vertex_to_copy = {}
        self.vertex_to_float_copy = {}
        self.copy_to_vertex = {}
        self.__inverted_repeat_vertex = {}
        self.merging_history = {}

        # summarize init
        logger.debug("init graph: self.vertex_clusters={}".format(self.vertex_clusters))


    def new_graph_with_vertex_reseeded(self, start_from=1):
        """
        A function called from within .write_to_fastg()
        """
        those_vertices = sorted(self.vertex_info)
        new_graph = Assembly(overlap=self.__overlap)
        name_trans = {those_vertices[go - start_from]: str(go)
                      for go in range(start_from, start_from + len(those_vertices))}
        for old_name in those_vertices:
            new_name = name_trans[old_name]
            this_v_info = deepcopy(self.vertex_info[old_name])
            this_v_info.name = new_name
            this_v_info.connections = {True: OrderedDict(), False: OrderedDict()}
            for this_end in self.vertex_info[old_name].connections:
                for next_name, next_end in self.vertex_info[old_name].connections[this_end]:
                    this_v_info.connections[this_end][(name_trans[next_name], next_end)] = None
            this_v_info.fill_fastg_form_name()
            new_graph.vertex_info[new_name] = this_v_info
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
                new_graph, name_trans = self.new_graph_with_vertex_reseeded()
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



    def write_out_tags(self, db_names, out_file):
        """
        Called from within major func find_target_graph()
        """
        # build a UNION of tagged vertices in db_names and sort them.
        tagged_vertices = set()
        for db_n in db_names:
            tagged_vertices |= self.tagged_vertices[db_n]
        tagged_vertices = sorted(tagged_vertices)

        # column headers for tags to be written to file
        lines = [["EDGE", "database", "database_weight", "loci"]]
        
        # iterate over tagged vertices set to fill 'lines' as a dataframe like list
        for this_vertex in tagged_vertices:
            if "tags" in self.vertex_info[this_vertex].other_attr:
                all_tags = self.vertex_info[this_vertex].other_attr["tags"]
                all_tag_list = sorted(all_tags)
                all_weights = self.vertex_info[this_vertex].other_attr["weight"]
                lines.append([this_vertex,
                              ";".join(all_tag_list),
                              ";".join([tag_n + "(" + str(all_weights[tag_n]) + ")" for tag_n in all_tag_list]),
                              ";".join([",".join(sorted(all_tags[tag_n])) for tag_n in all_tag_list])])
            else:
                here_tags = {tag_n for tag_n in db_names if this_vertex in self.tagged_vertices[tag_n]}
                lines.append([this_vertex,
                              ";".join(sorted(here_tags)),
                              "", ""])

        # write items tab-delimited for each line to out_file handle
        with open(out_file, "w") as out:
            out.writelines(["\t".join(line) + "\n" for line in lines])


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

        # reset to empty list. Each cluster is a connected set of vertices.
        self.vertex_clusters = []

        # get sorted list of vertices
        vertices = sorted(self.vertex_info)

        # iterate over vertices 
        for this_vertex in vertices:

            # build a set of connections (edges) from this vertex to others
            connecting_those = set()
            for connected_set in self.vertex_info[this_vertex].connections.values():
                for next_v, next_d in connected_set:
                    for go_to_set, cluster in enumerate(self.vertex_clusters):
                        if next_v in cluster:
                            connecting_those.add(go_to_set)

            # if no edges then store just this one
            if not connecting_those:
                self.vertex_clusters.append({this_vertex})

            # if 1 then store just just this one.
            elif len(connecting_those) == 1:
                self.vertex_clusters[connecting_those.pop()].add(this_vertex)

            # if many then ...
            else:
                sorted_those = sorted(connecting_those, reverse=True)
                self.vertex_clusters[sorted_those[-1]].add(this_vertex)
                for go_to_set in sorted_those[:-1]:
                    for that_vertex in self.vertex_clusters[go_to_set]:
                        self.vertex_clusters[sorted_those[-1]].add(that_vertex)
                    del self.vertex_clusters[go_to_set]



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
            
            # ...
            if vertex_name in self.merging_history:
                del self.merging_history[vertex_name]
        
        # recalculate clusters (connected vertices) now that a vertices were removed
        if update_cluster:
            self.update_vertex_clusters()
        
        # reset irv to empty dict
        self.__inverted_repeat_vertex = {}



    def rename_vertex(self, old_vertex, new_vertex, update_cluster=True):
        """
        Not currently called (deprecate?)
        """
        assert old_vertex != new_vertex
        assert new_vertex not in self.vertex_info, new_vertex + " exists!"
        self.vertex_info[new_vertex] = deepcopy(self.vertex_info[old_vertex])
        self.vertex_info[new_vertex].name = new_vertex
        for this_end in (True, False):
            for next_v, next_e in list(self.vertex_info[new_vertex].connections[this_end]):
                self.vertex_info[next_v].connections[next_e][(new_vertex, this_end)] = \
                    self.vertex_info[next_v].connections[next_e][(old_vertex, this_end)]
                del self.vertex_info[next_v].connections[next_e][(old_vertex, this_end)]
        for tag in self.tagged_vertices:
            if old_vertex in self.tagged_vertices[tag]:
                self.tagged_vertices[tag].add(new_vertex)
                self.tagged_vertices[tag].remove(old_vertex)
        if old_vertex in self.vertex_to_copy:
            this_copy = self.vertex_to_copy[old_vertex]
            self.copy_to_vertex[this_copy].remove(old_vertex)
            self.copy_to_vertex[this_copy].add(new_vertex)
            self.vertex_to_copy[new_vertex] = self.vertex_to_copy[old_vertex]
            del self.vertex_to_copy[old_vertex]
            self.vertex_to_float_copy[new_vertex] = self.vertex_to_float_copy[old_vertex]
            del self.vertex_to_float_copy[old_vertex]
        if self.vertex_info[old_vertex].fastg_form_name:
            split_long_name = self.vertex_info[old_vertex].fastg_form_name.split("_")
            self.vertex_info[new_vertex].fastg_form_name = \
                "_".join([split_long_name[0], new_vertex] + split_long_name[2:])
        del self.vertex_info[old_vertex]
        if update_cluster:
            for go_c, v_cluster in enumerate(self.vertex_clusters):
                if old_vertex in v_cluster:
                    self.vertex_clusters[go_c].remove(old_vertex)
                    self.vertex_clusters[go_c].add(new_vertex)
        if old_vertex in self.merging_history:
            self.merging_history[new_vertex] = self.merging_history[old_vertex]
            del self.merging_history[old_vertex]



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


    def is_sequential_repeat(self, search_vertex_name, return_pair_in_the_trunk_path=True):
        if search_vertex_name not in self.vertex_info:
            raise ProcessingGraphFailed("Vertex name " + search_vertex_name + " not found!")
        connection_set_t = self.vertex_info[search_vertex_name].connections[True]
        connection_set_f = self.vertex_info[search_vertex_name].connections[False]
        all_pairs_of_inner_circles = []

        def path_without_leakage(start_v, start_e, terminating_end_set):
            in_pipe_leak = False
            circle_in_between = []
            in_vertex_ends = set()
            in_vertex_ends.add((start_v, start_e))
            in_searching_con = [(start_v, not start_e)]
            while in_searching_con:
                in_search_v, in_search_e = in_searching_con.pop(0)
                if (in_search_v, in_search_e) in terminating_end_set:
                    # start from the same (next_t_v, next_t_e), merging to two different ends of connection_set_f
                    if circle_in_between:
                        in_pipe_leak = True
                        break
                    else:
                        circle_in_between.append(((start_v, start_e), (in_search_v, in_search_e)))
                elif (in_search_v, in_search_e) in connection_set_t:
                    in_pipe_leak = True
                    break
                else:
                    for n_in_search_v, n_in_search_e in self.vertex_info[in_search_v].connections[in_search_e]:
                        if (n_in_search_v, n_in_search_e) in in_vertex_ends:
                            pass
                        else:
                            in_vertex_ends.add((n_in_search_v, n_in_search_e))
                            in_searching_con.append((n_in_search_v, not n_in_search_e))
            if not in_pipe_leak:
                return circle_in_between
            else:
                return []

        # branching ends
        if len(connection_set_t) == len(connection_set_f) == 2:
            for next_t_v, next_t_e in list(connection_set_t):
                this_inner_circle = path_without_leakage(next_t_v, next_t_e, connection_set_f)
                if this_inner_circle:
                    # check leakage in reverse direction
                    reverse_v, reverse_e = this_inner_circle[0][1]
                    not_leak = path_without_leakage(reverse_v, reverse_e, connection_set_t)
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
                        if this_end:
                            if next_end:
                                new_vertex = this_vertex + "_" + "_".join(next_vertex.split("_")[::-1])
                            else:
                                new_vertex = this_vertex + "_" + next_vertex
                        else:
                            if next_end:
                                new_vertex = next_vertex + "_" + this_vertex
                            else:
                                new_vertex = "_".join(next_vertex.split("_")[::-1]) + "_" + this_vertex

                        # record merging history
                        self.merging_history[new_vertex] = (
                            self.merging_history.get(this_vertex, {this_vertex}) | 
                            self.merging_history.get(next_vertex, {next_vertex})
                        )
                        if this_vertex in self.merging_history:
                            del self.merging_history[this_vertex]
                        if next_vertex in self.merging_history:
                            del self.merging_history[next_vertex]

                        limited_vertices.remove(next_vertex)
                        limited_vertices.append(new_vertex)
                        
                        # initialization
                        self.vertex_info[new_vertex] = deepcopy(self.vertex_info[this_vertex])
                        self.vertex_info[new_vertex].name = new_vertex
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



    # def estimate_copy_and_depth_by_cov(
    #     self, 
    #     limited_vertices=None, 
    #     given_average_cov=None, 
    #     mode="embplant_pt", 
    #     re_initialize=False, 
    #     log_handler=None, 
    #     verbose=True, 
    #     debug=False):
    #     """
    #     Use seq coverage data to estimate copy and depth.
    #     """
    #     tool = EstimateCopyDepthFromCov(
    #         self, 
    #         limited_vertices, 
    #         given_average_cov, 
    #         mode, 
    #         re_initialize,
    #         log_handler, 
    #         verbose, 
    #         debug)

    #     # do we want to store a returned avg cov value?
    #     tool.run()



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


    def get_all_circular_paths(self, mode="embplant_pt", reverse_start_direction_for_pt=False):
        """
        :param mode:
        :param library_info: not used currently
        :param reverse_start_direction_for_pt:
        :return: sorted_paths
        """
        GetAllCircularPaths(self, mode=mode, reverse_start_direction_for_pt=reverse_start_direction_for_pt)


    # BRANCH OUT TO CLASS, FEWER ARGS
    def get_all_paths(self, mode="embplant_pt", log_handler=None):
        """
        :param mode:
        :param log_handler:
        :return: sorted_paths
        """
        pass
        #GetAllPaths()

        def standardize_paths(raw_paths, undirected_vertices):
            if undirected_vertices:
                corrected_paths = [[(this_v, True) if this_v in undirected_vertices else (this_v, this_e)
                                    for this_v, this_e in path_part]
                                   for path_part in raw_paths]
            else:
                corrected_paths = deepcopy(raw_paths)
            here_standardized_path = []
            for part_path in corrected_paths:
                if undirected_vertices:
                    rev_part = [(this_v, True) if this_v in undirected_vertices else (this_v, not this_e)
                                for this_v, this_e in part_path[::-1]]
                else:
                    rev_part = [(this_v, not this_e) for this_v, this_e in part_path[::-1]]
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
                here_standardized_path.append(standard_part)
            return corrected_paths, tuple(sorted(here_standardized_path, key=lambda x: smart_trans_for_sort(x)))



        def directed_graph_solver(ongoing_paths, next_connections, vertices_left, _all_start_ve, undirected_vertices):
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
                        directed_graph_solver(new_paths, new_connections, new_left, _all_start_ve,
                                              undirected_vertices)
            if not find_next:
                new_all_start_ve = deepcopy(_all_start_ve)
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

        # palindromic repeats
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
                    sorted_paths = [(this_path, ".repeat_pattern" + str(pattern_dict[acc_distance]))
                                    for this_path, acc_distance, foo_id in sorted_paths]
                else:
                    sorted_paths = [(this_path, "") for this_path in sorted(paths)]
            else:
                sorted_paths = [(this_path, "") for this_path in sorted(paths)]

            return sorted_paths




    def export_path(self, in_path):
        overlap = self.__overlap if self.__overlap else 0
        seq_names = []
        seq_segments = []
        for this_vertex, this_end in in_path:
            seq_segments.append(self.vertex_info[this_vertex].seq[this_end][overlap:])
            seq_names.append(this_vertex + ("-", "+")[this_end])
        # if not circular
        if (in_path[0][0], not in_path[0][1]) not in self.vertex_info[in_path[-1][0]].connections[in_path[-1][1]]:
            seq_segments[0] = self.vertex_info[in_path[0][0]].seq[in_path[0][1]][:overlap] + seq_segments[0]
        else:
            seq_names[-1] += "(circular)"
        return Sequence(",".join(seq_names), "".join(seq_segments))








