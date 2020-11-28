#!/usr/bin/env python

"""
Assembly class object and associated class objects
"""

from loguru import logger
from .SimpleAssembly import SimpleAssembly
from .utils import Sequence, SequenceList, Vertex, VertexInfo  # fasta funcs


INF = float("inf")



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
        """
        Get ORF lengths and sum in both directions from all or a subset of vertices        
        This is called in .get_all_paths() and .get_all_circular_paths()
        """
        # either select all vertices or a subset of vertices, and sort.
        if not limited_vertices:
            limited_vertices = sorted(self.vertex_info)
        else:
            limited_vertices = sorted(limited_vertices)

        # iterate over vertices and store ORF lens in both directions 
        for vertex_name in limited_vertices:

            # set 'orf' attr for this vertex to an empty dict
            self.vertex_info[vertex_name].other_attr["orf"] = {}

            # calculate and store ORF lens for both directions
            for direction in (True, False):

                # get list of lengths and sum for this ORF
                this_orf_lens = get_orf_lengths(self.vertex_info[vertex_name].seq[direction])
                orf_dict = {"lengths": this_orf_lens, "sum_len": sum(this_orf_lens)}
                self.vertex_info[vertex_name].other_attr["orf"][direction] = orf_dict



    def get_orf_lengths(
        self,
        sequence_string, 
        threshold=200, 
        which_frame=None,
        here_stop_codons=None, 
        here_start_codons=None):
        """
        :param sequence_string:
        :param threshold: default: 200
        :param which_frame: 1, 2, 3, or None
        :param here_stop_codons: default: CLASSIC_STOP_CODONS
        :param here_start_codons: default: CLASSIC_START_CODONS
        :return: [len_orf1, len_orf2, len_orf3 ...] # longest accumulated orfs among all frame choices
        """
        assert which_frame in {0, 1, 2, None}
        if which_frame is None:
            test_frames = [0, 1, 2]
        else:
            test_frames = [which_frame]
        if here_start_codons is None:
            here_start_codons = CLASSIC_START_CODONS
        if here_stop_codons is None:
            here_stop_codons = CLASSIC_STOP_CODONS
        orf_lengths = {}
        
        # 
        for try_frame in test_frames:
            orf_lengths[try_frame] = []
            this_start = False
            for go in range(try_frame, len(sequence_string), 3):
                if this_start:
                    if sequence_string[go:go + 3] not in here_stop_codons:
                        orf_lengths[try_frame][-1] += 3
                    else:
                        if orf_lengths[try_frame][-1] < threshold:
                            del orf_lengths[try_frame][-1]
                        this_start = False
                else:
                    if sequence_string[go:go + 3] in here_start_codons:
                        orf_lengths[try_frame].append(3)
                        this_start = True
                    else:
                        pass
        return sorted(orf_lengths.values(), key=lambda x: -sum(x))[0]



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


    def filter_by_coverage(
        self, drop_num=1, database_n="embplant_pt", log_hard_cov_threshold=10., weight_factor=100., min_sigma_factor=0.1, min_cluster=1, terminal_extra_weight=5.,                        verbose=False, log_handler=None, debug=False):
        changed = False
        overlap = self.__overlap if self.__overlap else 0
        log_hard_cov_threshold = abs(log(log_hard_cov_threshold))
        vertices = sorted(self.vertex_info)
        v_coverages = {this_v: self.vertex_info[this_v].cov / self.vertex_to_copy.get(this_v, 1) for this_v in vertices}
        try:
            max_tagged_cov = max([v_coverages[tagged_v] for tagged_v in self.tagged_vertices[database_n]])
        except ValueError as e:
            if log_handler:
                log_handler.info("tagged vertices: " + str(self.tagged_vertices))
            else:
                sys.stdout.write("tagged vertices: " + str(self.tagged_vertices) + "\n")
            raise e
        # removing coverage with 10 times lower/greater than tagged_cov
        removing_low_cov = [candidate_v
                            for candidate_v in vertices
                            if abs(log(self.vertex_info[candidate_v].cov / max_tagged_cov)) > log_hard_cov_threshold]
        if removing_low_cov:
            if log_handler and (debug or verbose):
                log_handler.info("removing extremely outlying coverage contigs: " + str(removing_low_cov))
            elif verbose or debug:
                sys.stdout.write("removing extremely outlying coverage contigs: " + str(removing_low_cov) + "\n")
            self.remove_vertex(removing_low_cov)
            changed = True
        merged = self.merge_all_possible_vertices()
        if merged:
            changed = True
        vertices = sorted(self.vertex_info)
        v_coverages = {this_v: self.vertex_info[this_v].cov / self.vertex_to_copy.get(this_v, 1)
                       for this_v in vertices}

        coverages = np.array([v_coverages[this_v] for this_v in vertices])
        cover_weights = np.array([(self.vertex_info[this_v].len - overlap)
                                  # multiply by copy number
                                  * self.vertex_to_copy.get(this_v, 1)
                                  # extra weight to short non-target
                                  * (terminal_extra_weight if self.vertex_info[this_v].is_terminal() else 1)
                                  for this_v in vertices])
        tag_kinds = [tag_kind for tag_kind in self.tagged_vertices if self.tagged_vertices[tag_kind]]
        tag_kinds.sort(key=lambda x: x != database_n)
        set_cluster = {}
        for v_id, vertex_name in enumerate(vertices):
            for go_tag, this_tag in enumerate(tag_kinds):
                if vertex_name in self.tagged_vertices[this_tag]:
                    if v_id not in set_cluster:
                        set_cluster[v_id] = set()
                    set_cluster[v_id].add(go_tag)
        min_tag_kind = {0}
        for v_id in set_cluster:
            if 0 not in set_cluster[v_id]:
                min_tag_kind |= set_cluster[v_id]
        min_cluster = max(min_cluster, len(min_tag_kind))

        # old way:
        # set_cluster = {v_coverages[tagged_v]: 0 for tagged_v in self.tagged_vertices[mode]}

        # gmm_scheme = gmm_with_em_aic(coverages, maximum_cluster=6, cluster_limited=set_cluster,
        #                              min_sigma_factor=min_sigma_factor)
        if log_handler and (debug or verbose):
            log_handler.info("Vertices: " + str(vertices))
            log_handler.info("Coverages: " + str([float("%.1f" % cov_x) for cov_x in coverages]))
        elif verbose or debug:
            sys.stdout.write("Vertices: " + str(vertices) + "\n")
            sys.stdout.write("Coverages: " + str([float("%.1f" % cov_x) for cov_x in coverages]) + "\n")
        gmm_scheme = weighted_gmm_with_em_aic(coverages, data_weights=cover_weights,
                                              minimum_cluster=min_cluster, maximum_cluster=6,
                                              cluster_limited=set_cluster, min_sigma_factor=min_sigma_factor,
                                              log_handler=log_handler, verbose_log=verbose)
        cluster_num = gmm_scheme["cluster_num"]
        parameters = gmm_scheme["parameters"]
        # for debug
        # print('testing', end="\n")
        # for temp in parameters:
        #     print("  ", temp, end="\n")
        labels = gmm_scheme["labels"]
        if log_handler and (debug or verbose):
            log_handler.info("Labels: " + str(labels))
        elif verbose or debug:
            sys.stdout.write("Labels: " + str(labels) + "\n")

        # 1
        selected_label_type = list(
            set([lb for go, lb in enumerate(labels) if vertices[go] in self.tagged_vertices[database_n]]))
        if len(selected_label_type) > 1:
            label_weights = {}
            # for lb in selected_label_type:
            #     this_add_up = 0
            #     for go in np.where(labels == lb)[0]:
            #         this_add_up += self.vertex_info[vertices[go]].get("weight", {}).get(mode, 0)
            #     label_weights[lb] = this_add_up
            label_weights = {lb: sum([self.vertex_info[vertices[go]].other_attr.get("weight", {}).get(database_n, 0)
                                      for go in np.where(labels == lb)[0]])
                             for lb in selected_label_type}
            selected_label_type.sort(key=lambda x: -label_weights[x])
            remained_label_type = {selected_label_type[0]}
            for candidate_lb_type in selected_label_type[1:]:
                if label_weights[candidate_lb_type] * weight_factor >= selected_label_type[0]:
                    remained_label_type.add(candidate_lb_type)
                else:
                    break
            extra_kept = set()
            for candidate_lb_type in selected_label_type:
                if candidate_lb_type not in remained_label_type:
                    can_mu = parameters[candidate_lb_type]["mu"]
                    for remained_l in remained_label_type:
                        if abs(can_mu - parameters[remained_l]["mu"]) < 2 * parameters[remained_l]["sigma"]:
                            extra_kept.add(candidate_lb_type)
                            break
            remained_label_type |= extra_kept
        else:
            remained_label_type = {selected_label_type[0]}
        if debug or verbose:
            if log_handler:
                log_handler.info("\t".join(["Mu" + str(go) + ":" + str(parameters[lab_tp]["mu"]) +
                                            " Sigma" + str(go) + ":" + str(parameters[lab_tp]["sigma"])
                                            for go, lab_tp in enumerate(remained_label_type)]))
            else:
                sys.stdout.write("\t".join(["Mu" + str(go) + ":" + str(parameters[lab_tp]["mu"]) +
                                            " Sigma" + str(go) + ":" + str(parameters[lab_tp]["sigma"])
                                            for go, lab_tp in enumerate(remained_label_type)]) + "\n")

        # 2
        # exclude_label_type = set()
        # if len(tag_kinds) > 1:
        #     for go_l, this_label in enumerate(labels):
        #         for this_tag in tag_kinds[1:]:
        #             if vertices[go_l] in self.tagged_vertices[this_tag]:
        #                 exclude_label_type.add(this_label)
        #                 break
        # exclude_label_type = sorted(exclude_label_type)
        # if exclude_label_type:
        #     check_ex = 0
        #     while check_ex < len(exclude_label_type):
        #         if exclude_label_type[check_ex] in remained_label_type:
        #             if debug or verbose:
        #                 if log_handler:
        #                     log_handler.info("label " + str(exclude_label_type[check_ex]) + " kept")
        #                 else:
        #                     sys.stdout.write("label " + str(exclude_label_type[check_ex]) + " kept\n")
        #             del exclude_label_type[check_ex]
        #         else:
        #             check_ex += 1

        candidate_dropping_label_type = {l_t: inf for l_t in set(range(cluster_num)) - remained_label_type}
        for lab_tp in candidate_dropping_label_type:
            check_mu = parameters[lab_tp]["mu"]
            check_sigma = parameters[lab_tp]["sigma"]
            for remained_l in remained_label_type:
                rem_mu = parameters[remained_l]["mu"]
                rem_sigma = parameters[remained_l]["sigma"]
                this_dist = abs(rem_mu - check_mu) - 2 * (check_sigma + rem_sigma)
                candidate_dropping_label_type[lab_tp] = min(candidate_dropping_label_type[lab_tp], this_dist)
        dropping_type = sorted(candidate_dropping_label_type, key=lambda x: -candidate_dropping_label_type[x])
        drop_num = max(len(tag_kinds) - 1, drop_num)
        dropping_type = dropping_type[:drop_num]
        if debug or verbose:
            if log_handler:
                for lab_tp in dropping_type:
                    if candidate_dropping_label_type[lab_tp] < 0:
                        log_handler.warning("Indistinguishable vertices "
                                            + str([vertices[go] for go in np.where(labels == lab_tp)[0]])
                                            + " removed!")
            else:
                for lab_tp in dropping_type:
                    if candidate_dropping_label_type[lab_tp] < 0:
                        sys.stdout.write("Warning: indistinguishable vertices "
                                         + str([vertices[go] for go in np.where(labels == lab_tp)[0]])
                                         + " removed!\n")
        vertices_to_del = {vertices[go] for go, lb in enumerate(labels) if lb in set(dropping_type)}
        if vertices_to_del:
            changed = True
            if verbose or debug:
                if log_handler:
                    log_handler.info("removing outlying coverage contigs: " + str(vertices_to_del))
                else:
                    sys.stdout.write("removing outlying coverage contigs: " + str(vertices_to_del) + "\n")
            self.remove_vertex(vertices_to_del)
        return changed, [(parameters[lab_tp]["mu"], parameters[lab_tp]["sigma"]) for lab_tp in remained_label_type]


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
        if limit_extending_len not in (None, inf):
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


    def generate_consensus_vertex(self, vertices, directions, copy_tags=True, check_parallel_vertices=True, log_handler=None):
        if check_parallel_vertices:
            connection_type = None
            seq_len = None
            if not len(vertices) == len(set(vertices)) == len(directions):
                raise ProcessingGraphFailed("Cannot generate consensus (1)!")
            for go_v, this_v in enumerate(vertices):
                if seq_len:
                    if seq_len != len(self.vertex_info[this_v].seq[True]):
                        raise ProcessingGraphFailed("Cannot generate consensus (2)!")
                else:
                    seq_len = len(self.vertex_info[this_v].seq[True])
                this_cons = self.vertex_info[this_v].connections
                this_ends = tuple([tuple(sorted(this_cons[[directions[go_v]]])),
                                   tuple(sorted(this_cons[not [directions[go_v]]]))])
                if connection_type:
                    if connection_type != this_ends:
                        raise ProcessingGraphFailed("Cannot generate consensus (3)!")
                else:
                    connection_type = this_ends

        if len(vertices) > 1:
            new_vertex = "(" + "|".join(vertices) + ")"
            self.vertex_info[new_vertex] = deepcopy(self.vertex_info[vertices[0]])
            self.vertex_info[new_vertex].name = new_vertex
            self.vertex_info[new_vertex].cov = sum([self.vertex_info[v].cov for v in vertices])
            self.vertex_info[new_vertex].fastg_form_name = None
            # if "long" in self.vertex_info[new_vertex]:
            #     del self.vertex_info[new_vertex]["long"]

            self.merging_history[new_vertex] = set()
            for candidate_v in vertices:
                if candidate_v in self.merging_history:
                    for sub_v_n in self.merging_history[candidate_v]:
                        self.merging_history[new_vertex].add(sub_v_n)
                else:
                    self.merging_history[new_vertex].add(candidate_v)
            for candidate_v in vertices:
                if candidate_v in self.merging_history:
                    del self.merging_history[candidate_v]

            for new_end in (True, False):
                for n_n_v, n_n_e in self.vertex_info[new_vertex].connections[new_end]:
                    self.vertex_info[n_n_v].connections[n_n_e][(new_vertex, new_end)] = None

            consensus_s = generate_consensus(
                *[self.vertex_info[v].seq[directions[go]] for go, v in enumerate(vertices)])
            self.vertex_info[new_vertex].seq[directions[0]] = consensus_s
            self.vertex_info[new_vertex].seq[not directions[0]] = complementary_seq(consensus_s)
            if copy_tags:
                for db_n in self.tagged_vertices:
                    if vertices[0] in self.tagged_vertices[db_n]:
                        self.tagged_vertices[db_n].add(new_vertex)
                        self.tagged_vertices[db_n].remove(vertices[0])

            # tags
            if copy_tags:
                for other_vertex in vertices[1:]:
                    if "tags" in self.vertex_info[other_vertex].other_attr:
                        if "tags" not in self.vertex_info[new_vertex].other_attr:
                            self.vertex_info[new_vertex].other_attr["tags"] = \
                                deepcopy(self.vertex_info[other_vertex].other_attr["tags"])
                        else:
                            for db_n in self.vertex_info[other_vertex].other_attr["tags"]:
                                if db_n not in self.vertex_info[new_vertex].other_attr["tags"]:
                                    self.vertex_info[new_vertex].other_attr["tags"][db_n] \
                                        = deepcopy(self.vertex_info[other_vertex].other_attr["tags"][db_n])
                                else:
                                    self.vertex_info[new_vertex].other_attr["tags"][db_n] \
                                        |= self.vertex_info[other_vertex].other_attr["tags"][db_n]
                    if "weight" in self.vertex_info[other_vertex].other_attr:
                        if "weight" not in self.vertex_info[new_vertex].other_attr:
                            self.vertex_info[new_vertex].other_attr["weight"] \
                                = deepcopy(self.vertex_info[other_vertex].other_attr["weight"])
                        else:
                            for db_n in self.vertex_info[other_vertex].other_attr["weight"]:
                                if db_n not in self.vertex_info[new_vertex].other_attr["weight"]:
                                    self.vertex_info[new_vertex].other_attr["weight"][db_n] \
                                        = self.vertex_info[other_vertex].other_attr["weight"][db_n]
                                else:
                                    self.vertex_info[new_vertex].other_attr["weight"][db_n] \
                                        += self.vertex_info[other_vertex].other_attr["weight"][db_n]
                    for db_n in self.tagged_vertices:
                        if other_vertex in self.tagged_vertices[db_n]:
                            self.tagged_vertices[db_n].add(new_vertex)
                            self.tagged_vertices[db_n].remove(other_vertex)
            self.remove_vertex(vertices)
            if log_handler:
                log_handler.info("Consensus made: " + new_vertex)
            else:
                log_handler.info("Consensus made: " + new_vertex + "\n")


    def find_target_graph(self):
        """

        """
        pass #FindTargetGraph().run()


    def peel_subgraph(self, subgraph, mode="", subgraph_was_merged=False, log_handler=None, verbose=False):
        """
        Not sure yet where this is called...?
        """
        assert isinstance(subgraph, Assembly)
        if subgraph_was_merged:
            subgraph_vertices = set()
            for merged_v_name in subgraph.vertex_info:
                if merged_v_name in subgraph.merging_history:
                    subgraph_vertices |= subgraph.merging_history[merged_v_name]
                else:
                    subgraph_vertices.add(merged_v_name)
        else:
            subgraph_vertices = set(subgraph.vertex_info)
        limited_vertices = set(self.vertex_info) & set(subgraph_vertices)
        if not limited_vertices:
            if log_handler:
                log_handler.warning("No overlapped vertices found for peeling!")
            else:
                sys.stdout.write("No overlapped vertices found for peeling!\n")
            if verbose:
                if log_handler:
                    log_handler.warning("graph vertices: " + str(sorted(self.vertex_info)))
                    log_handler.warning("subgraph vertices: " + str(sorted(subgraph.vertex_info)))
                else:
                    sys.stdout.write("graph vertices: " + str(sorted(self.vertex_info)))
                    sys.stdout.write("subgraph vertices: " + str(sorted(subgraph.vertex_info)))
        average_cov = self.estimate_copy_and_depth_by_cov(
            limited_vertices, mode=mode, re_initialize=True, verbose=verbose)
        vertices_peeling_ratios = {}
        checked = set()
        for peel_name in sorted(limited_vertices):
            for peel_end, peel_connection_set in self.vertex_info[peel_name].connections.items():
                if (peel_name, not peel_end) in checked:
                    continue
                else:
                    checked.add((peel_name, not peel_end))
                for (external_v_n, external_v_e) in sorted(peel_connection_set):
                    if external_v_n in subgraph_vertices:
                        continue
                    if self.vertex_to_float_copy[peel_name] > self.vertex_to_copy[peel_name]:
                        # only peel the average part
                        vertices_peeling_ratios[peel_name] = \
                            1 - self.vertex_to_copy[peel_name] / self.vertex_to_float_copy[peel_name]
                        forward_peeling = [(next_n, not next_e)
                                           for next_n, next_e in self.vertex_info[peel_name].connections[not peel_end]
                                           if next_n in limited_vertices and (next_n, not next_e) not in checked]
                        while forward_peeling:
                            next_name, next_end = forward_peeling.pop(0)
                            if self.vertex_to_float_copy[next_name] > self.vertex_to_copy[next_name]:
                                vertices_peeling_ratios[next_name] = \
                                    1 - self.vertex_to_copy[next_name] / self.vertex_to_float_copy[next_name]
                                checked.add((next_name, next_end))
                                forward_peeling.extend(
                                    [(nx_nx_n, not nx_nx_e)
                                     for nx_nx_n, nx_nx_e in self.vertex_info[next_name].connections[next_end]
                                     if nx_nx_n in limited_vertices and (nx_nx_n, not nx_nx_e) not in checked])
        remove_vertices = {del_v for del_v in limited_vertices if del_v not in vertices_peeling_ratios}
        self.remove_vertex(remove_vertices)
        for peel_this_n in sorted(vertices_peeling_ratios):
            self.vertex_info[peel_this_n].cov *= vertices_peeling_ratios[peel_this_n]
            if "weight" in self.vertex_info[peel_this_n].other_attr and \
                    mode in self.vertex_info[peel_this_n].other_attr["weight"]:
                self.vertex_info[peel_this_n].other_attr["weight"][mode] *= vertices_peeling_ratios[peel_this_n]



    # BRANCH OUT TO CLASS, FEWER ARGS
    def get_all_paths(self, mode="embplant_pt", log_handler=None):
        """
        
        Returns
        --------
        sorted_paths
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








