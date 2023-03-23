#!/usr/bin/env python

"""
Base Assembly class for parsing input graph files.
"""

from loguru import logger
from collections import OrderedDict
from traversome.utils import Sequence, SequenceList, ProcessingGraphFailed, complementary_seq
from hashlib import sha256
from typing import OrderedDict as typingODict
import os



#######################################################
###   GLOBALS
#######################################################
INF = float("inf")
DEFAULT_COV = 1
VERTEX_DIRECTION_BOOL_TO_STR = {True: "+", False: "-"}

#######################################################
###   CLASSES
#######################################################


class Vertex(object):
    def __init__(self,
                 v_name,
                 length=None,
                 coverage=None,
                 forward_seq=None,
                 reverse_seq=None,
                 tail_connections: typingODict = None,
                 head_connections: typingODict = None,
                 fastg_form_long_name=None):
        """
        :param v_name: str
        :param length: int
        :param coverage: float
        :param forward_seq: str
        :param reverse_seq: str
        :param tail_connections: OrderedDict()
        :param head_connections: OrderedDict()
        :param fastg_form_long_name: str
        self.seq={True: FORWARD_SEQ, False: REVERSE_SEQ}
        self.connections={True: tail_connection_set, False: head_connection_set}
        """
        self.name = v_name
        self.len = length
        self.cov = coverage

        """ True: forward, False: reverse """
        if forward_seq and reverse_seq:
            assert forward_seq == complementary_seq(reverse_seq), "forward_seq != complementary_seq(reverse_seq)"
            self.seq = {True: forward_seq, False: reverse_seq}
        elif forward_seq:
            self.seq = {True: forward_seq, False: complementary_seq(forward_seq)}
        elif reverse_seq:
            self.seq = {True: complementary_seq(reverse_seq), False: reverse_seq}
        else:
            self.seq = {True: None, False: None}

        # True: tail, False: head
        self.connections = {True: OrderedDict(), False: OrderedDict()}
        assert tail_connections is None or isinstance(tail_connections, OrderedDict), \
            "tail_connections must be an OrderedDict()"
        assert head_connections is None or isinstance(head_connections, OrderedDict), \
            "head_connections must be an OrderedDict()"
        if tail_connections:
            self.connections[True] = tail_connections
        if head_connections:
            self.connections[False] = head_connections
        self.fastg_form_name = fastg_form_long_name
        self.merging_history = VertexMergingHistory([(v_name, True)])
        self.other_attr = {}

    def __repr__(self):
        return self.name

    def fill_fastg_form_name(self, check_valid=False):
        """
        ensures vertex (contig) names are valid, i.e., avoids ints.
        """
        if check_valid:
            if not str(self.name).isdigit():
                raise ValueError("Invalid vertex name for fastg format!")
            if not isinstance(self.len, int):
                raise ValueError("Invalid vertex length for fastg format!")
            if not (isinstance(self.cov, int) or isinstance(self.cov, float)):
                raise ValueError("Invalid vertex coverage for fastg format!")
        self.fastg_form_name = (
            "EDGE_{}_length_{}_cov_{}"
                .format(
                str(self.name),
                str(self.len),
                str(round(self.cov, 5)),
            )
        )

    def is_terminal(self):
        return not (self.connections[True] and self.connections[False])

    def is_self_loop(self):
        return (self.name, False) in self.connections[True]


class VertexInfo(dict):
    """
    Superclass of dict that requires values to be Vertices
    """

    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            if not isinstance(val, Vertex):
                raise ValueError("Value must be a Vertex type! Current: " + str(type(val)))
        dict.__init__(kwargs)

    def __setitem__(self, key, val):
        if not isinstance(val, Vertex):
            raise ValueError("Value must be a Vertex type! Current: " + str(type(val)))
        val.name = key
        dict.__setitem__(self, key, val)


class VertexMergingHistory(object):
    def __init__(self, history_or_path=None):
        self.__list = []
        if history_or_path:
            for each_item in history_or_path:
                is_vertex = isinstance(each_item, tuple) and len(each_item) == 2 and isinstance(each_item[1], bool)
                is_hist = isinstance(each_item, VertexMergingHistory)
                assert is_vertex or is_hist
                if is_vertex:
                    self.__list.append(each_item)
                else:
                    self.__list.extend(each_item.list())

    def add(self, new_history_or_vertex, add_new_to_front=False, reverse_the_new=False):
        is_vertex = isinstance(new_history_or_vertex, tuple) and len(new_history_or_vertex) == 2
        is_hist = isinstance(new_history_or_vertex, VertexMergingHistory)
        assert is_vertex or is_hist
        if add_new_to_front:
            if reverse_the_new:
                if is_vertex:
                    self.__list.insert(0, (new_history_or_vertex[0], not new_history_or_vertex[1]))
                else:
                    self.__list = list(-new_history_or_vertex) + self.__list
            else:
                if is_vertex:
                    self.__list.insert(0, new_history_or_vertex)
                else:
                    self.__list = list(new_history_or_vertex) + self.__list
        else:
            if reverse_the_new:
                if is_vertex:
                    self.__list.append((new_history_or_vertex[0], not new_history_or_vertex[1]))
                else:
                    self.__list.extend(list(-new_history_or_vertex))
            else:
                if is_vertex:
                    self.__list.append(new_history_or_vertex)
                else:
                    self.__list.extend(list(new_history_or_vertex))

    def __neg__(self):
        return VertexMergingHistory([(each_vertex[0], not each_vertex[1]) for each_vertex in self.__list[::-1]])

    def __iter__(self):
        for item in self.__list:
            yield item

    def __str__(self):
        return "_".join([str(each_vertex[0]) if isinstance(each_vertex, tuple) else str(each_vertex)
                         for each_vertex in self.__list])

    def reverse(self):
        self.__list = [(each_vertex[0], not each_vertex[1]) for each_vertex in self.__list[::-1]]

    def path_list(self):
        return list(self.__list)
        # return [each_vertex.path_list() if isinstance(each_vertex, MergingHistory) else each_vertex
        #         for each_vertex in self.__list]

    def vertex_set(self):
        v_set = set()
        for each_item in self.__list:
            if isinstance(each_item[0], VertexEditHistory):
                v_set.update(each_item[0].vertex_set())
            else:
                v_set.add(each_item[0])
        return v_set


class VertexEditHistory(object):
    def __init__(self, raw_item):
        """
        :param raw_item: (name1 or VertexMergingHistory(), label)
        """
        assert isinstance(raw_item, tuple) and len(raw_item) == 2
        self.__item = raw_item

    def __str__(self):
        return str(self.__item[0]) + "__" + self.__item[1]

    def vertex_set(self):
        v_set = set()
        if isinstance(self.__item[0], VertexMergingHistory):
            v_set.update(self.__item[0].vertex_set())
        else:
            v_set.add(self.__item[0])
        return v_set


class AssemblySimple(object):
    """
    Base class for Assembly class objects used to parse input graph files.

    Attributes:
    -----------
    vertex_info (VertexInfo):
        Class object for storing Vertex info.

    __uni_overlap (bool):
        None
  

    Functions:
    ----------
    parse_gfa: parse GFA files to fill vertex_info and __uni_overlap.
    parse_fastg: parse fastg to fill vertex_info and __uni_overlap.
    uni_overlap: return __uni_overlap
    write_to_fasta: writes
    """
    def __init__(self, graph_file=None, min_cov=0., max_cov=INF, uni_overlap=None):

        # base attributes       
        self.graph_file = graph_file
        self.min_cov = min_cov
        self.max_cov = max_cov
        self.__uni_overlap = uni_overlap

        # destination to be filled with parsed GFA data.
        self.vertex_info = VertexInfo()

        # parse the 
        if self.graph_file:
            if self.graph_file.endswith(".gfa"):
                self.parse_gfa()
            else:
                self.parse_fastg()

    def __repr__(self):
        """
        Human readable desciptive output of the Assembly object
        """
        res = []
        for v in sorted(self.vertex_info):
            res.append(">" + v + "__" + str(self.vertex_info[v].len) + "__" + str(self.vertex_info[v].cov))
            for e in (False, True):
                if len(self.vertex_info[v].connections[e]):
                    res.append("(" + ["head", "tail"][e] + ":")
                    res.append(",".join([next_v + "_" + ["head", "tail"][next_e]
                                         for next_v, next_e in self.vertex_info[v].connections[e]]))
                    res.append(")")
            res.append("\n")
        return "".join(res)

    def __bool__(self):
        return bool(self.vertex_info)

    def __iter__(self):
        "allow iteration of Assembly objects to return ordered vertex info."
        for vertex in sorted(self.vertex_info):
            yield self.vertex_info[vertex]

    def __getitem__(self, item: str):
        return self.vertex_info[item]

    def parse_gfa(self):
        """
        Parse a GFA format file and fill information to self.vertex_info.

        Note: This doesn't seem currently to have a way to identify v.2.0
        """
        logger.info("Parsing graph (GFA)")
        with open(self.graph_file) as gfa_open:

            # read first line to get version number
            line = gfa_open.readline()
            gfa_version_number = "1.0"

            # parse elements from first line
            if line.startswith("H\t"):
                for element in line.strip().split("\t")[1:]:
                    element_tag, element_type, element_description = element.split(":", maxsplit=2)
                    # element_tag, element_type, element_description = element[0], element[1], ":".join(element[2:])
                    if element_tag == "VN":
                        gfa_version_number = element_description

            # return to start of file
            gfa_open.seek(0)

            # parse differently dependong on version. 
            # Fills .vertex_info list, and self.__uni_overlap int
            if gfa_version_number == "1.0":
                self.parse_gfa_v1(gfa_open)
            elif gfa_version_number == "2.0":
                self.parse_gfa_v2(gfa_open)
            else:
                raise ProcessingGraphFailed("Unrecognized GFA version number: " + gfa_version_number)

    def parse_gfa_v1(self, gfa_open):
        """
        Fills .vertex_info with the sequence tag lines and 
        and .__uni_overlap with the link tag lines.
        """
        # set for storing kmer results
        overlap_values = set()

        # iterate over lines in gfa
        for line in gfa_open:

            # if the line contains a sequence tag 
            if line.startswith("S\t"):
                elements = line.strip().split("\t")
                elements.pop(0)  # record_type
                vertex_name = elements.pop(0)  # segment name
                sequence = elements.pop(0)
                seq_len_tag = None
                kmer_count = None
                seq_depth_tag = None
                sh_256_val = None
                other_attributes = {}

                # split each into element_tag, element_type, element_description
                for element in elements:
                    element = element.split(":", maxsplit=2)
                    # skip RC/FC

                    # get the sequence length
                    if element[0].upper() == "LN":
                        seq_len_tag = int(element[-1])
                    
                    # ...
                    elif element[0].upper() == "KC":
                        kmer_count = int(element[-1])

                    # get read counts (as kmer counts)
                    elif element[0].upper() == "RC":
                        kmer_count = int(element[-1])

                    # seqdepth tag ...
                    elif element[0].upper() == "DP":
                        seq_depth_tag = float(element[-1])

                    # get sequence checksum 
                    elif element[0].upper() == "SH":
                        sh_256_val = ":".join(element[2:])

                    # url or local file path of sequence
                    elif element[0].upper() == "UR":
                        seq_file_path = element[-1]
                        if os.path.isfile(seq_file_path):
                            if sequence == "*":
                                sequence = "".join([sub_seq.strip() for sub_seq in open(seq_file_path)])
                            else:
                                tag_seq = "".join([sub_seq.strip() for sub_seq in open(seq_file_path)])
                                if tag_seq != sequence:
                                    raise ProcessingGraphFailed(
                                        vertex_name + " sequences from different sources!")
                        else:
                            raise ProcessingGraphFailed(
                                seq_file_path + " for " + vertex_name + " does not exist!")

                    else:
                        other_attributes[element[0].upper()] = element[-1]

                # store the relevant information
                seq_len = len(sequence)
                if (seq_len_tag is not None) and (seq_len != seq_len_tag):
                    raise ProcessingGraphFailed(
                        vertex_name + " has unmatched sequence length as noted!")

                # complain if bad SHA 
                if (sh_256_val is not None) and (sh_256_val != sha256(sequence)):
                    raise ProcessingGraphFailed(
                        vertex_name + " has unmatched sha256 value as noted!")

                # count data is present, so store it in vertex_info
                if (kmer_count is not None) or (seq_depth_tag is not None):
                    
                    # normalize kmer count to be per-bp
                    if kmer_count is not None:
                        seq_depth = kmer_count / float(seq_len)
                    else:  # seq_depth_tag is not None:
                        seq_depth = seq_depth_tag
                    
                    # if seqdepth is in suitable range then save data
                    if self.min_cov <= seq_depth <= self.max_cov:
                        vert = Vertex(vertex_name, seq_len, seq_depth, sequence)
                        vert.other_attr = other_attributes
                        self.vertex_info[vertex_name] = vert

                        # convert name from integer to contig name
                        if vertex_name.isdigit():
                            self.vertex_info[vertex_name].fill_fastg_form_name()

                # no count data, just store vertex w/ default coverage
                else:
                    vert = Vertex(vertex_name, seq_len, DEFAULT_COV, sequence)
                    vert.other_attr = other_attributes
                    self.vertex_info[vertex_name] = vert

        # return to beginning of file.
        gfa_open.seek(0)

        # iterate over lines in GFA 
        for line in gfa_open:

            # if the line contains a link tag
            if line.startswith("L\t"):

                # parse link info
                elements = line.strip().split("\t")
                elements.pop(0)  # flag
                vertex_1 = elements.pop(0)
                end_1 = elements.pop(0)
                vertex_2 = elements.pop(0)
                end_2 = elements.pop(0)
                alignment_cigar = elements.pop(0)

                # "head"~False, "tail"~True
                if (vertex_1 in self.vertex_info) and (vertex_2 in self.vertex_info):

                    # recode end1 and end2 into a boolean
                    end_1 = {"+": True, "-": False}[end_1]
                    end_2 = {"+": False, "-": True}[end_2]

                    # store the kmer values
                    this_overlap = alignment_cigar.strip("M")
                    try:
                        this_overlap = int(this_overlap)
                    except:
                        raise ValueError(
                            "Contig uni_overlap cigar contains characters other than M: " + alignment_cigar)
                    overlap_values.add(this_overlap)

                    # store the connection (link) between these verts
                    self.vertex_info[vertex_1].connections[end_1][(vertex_2, end_2)] = this_overlap
                    self.vertex_info[vertex_2].connections[end_2][(vertex_1, end_1)] = this_overlap

        # Record the uni_overlap of READS with the contigs.
        # if no kmers data was found then uni_overlap is zero
        if len(overlap_values) == 0:
            self.__uni_overlap = None

        # # if multiple overlaps counts are present then its an error
        # elif len(overlap_values) > 1:
        #     raise ProcessingGraphFailed(
        #         "Multiple uni_overlap values: {}".format(
        #             ",".join([str(ol) for ol in sorted(overlap_values)]))
        #     )

        # only one value is present so let's store it as an int
        else:
            self.__uni_overlap = overlap_values.pop()

    def parse_gfa_v2(self, gfa_open):
        "GFA VERSION 2 PARSING"

        # set for storing kmer results
        overlap_values = set()

        # iterate over lines in gfa
        for line in gfa_open:
            if line.startswith("S\t"):
                elements = line.strip().split("\t")
                elements.pop(0)  # record_type
                vertex_name = elements.pop(0)  # segment name
                int(elements.pop(0))  # seq_len_tag
                sequence = elements.pop(0)
                seq_len_tag = None
                kmer_count = None
                seq_depth_tag = None
                sh_256_val = None
                other_attributes = {}
                for element in elements:
                    element = element.split(":")  # element_tag, element_type, element_description
                    # skip RC/FC
                    if element[0].upper() == "KC":
                        kmer_count = int(element[-1])
                    elif element[0].upper() == "RC":  # took read counts as kmer counts
                        kmer_count = int(element[-1])
                    elif element[0].upper() == "DP":
                        seq_depth_tag = float(element[-1])
                    elif element[0].upper() == "SH":
                        sh_256_val = ":".join(element[2:])
                    elif element[0].upper() == "UR":
                        seq_file_path = element[-1]
                        if os.path.isfile(seq_file_path):
                            if sequence == "*":
                                sequence = "".join([sub_seq.strip() for sub_seq in open(seq_file_path)])
                            else:
                                tag_seq = "".join([sub_seq.strip() for sub_seq in open(seq_file_path)])
                                if tag_seq != sequence:
                                    raise ProcessingGraphFailed(
                                        vertex_name + " sequences from different sources!")
                        else:
                            raise ProcessingGraphFailed(
                                seq_file_path + " for " + vertex_name + " does not exist!")
                    else:
                        other_attributes[element[0].upper()] = element[-1]
                seq_len = len(sequence)
                if seq_len_tag is not None and seq_len != seq_len_tag:
                    raise ProcessingGraphFailed(vertex_name + " has unmatched sequence length as noted!")
                if sh_256_val is not None and sh_256_val != sha256(sequence):
                    raise ProcessingGraphFailed(vertex_name + " has unmatched sha256 value as noted!")
                if kmer_count is not None or seq_depth_tag is not None:
                    if kmer_count is not None:
                        seq_depth = kmer_count / float(seq_len)
                    else:  # seq_depth_tag is not None:
                        seq_depth = seq_depth_tag
                    if self.min_cov <= seq_depth <= self.max_cov:
                        self.vertex_info[vertex_name] = Vertex(vertex_name, seq_len, seq_depth, sequence)
                        self.vertex_info[vertex_name].other_attr = other_attributes
                        if vertex_name.isdigit():
                            self.vertex_info[vertex_name].fill_fastg_form_name()
                else:
                    self.vertex_info[vertex_name] = Vertex(vertex_name, seq_len, DEFAULT_COV, sequence)
                    self.vertex_info[vertex_name].other_attr = other_attributes

        # return to start of file
        gfa_open.seek(0)
        for line in gfa_open:
            if line.startswith("E\t"):  # gfa2 uses E
                elements = line.strip().split("\t")
                elements.pop(0)  # flag
                vertex_1 = elements.pop(0)
                end_1 = elements.pop(0)
                vertex_2 = elements.pop(0)
                end_2 = elements.pop(0)
                alignment_cigar = elements.pop(0)
                # "head"~False, "tail"~True
                if vertex_1 in self.vertex_info and vertex_2 in self.vertex_info:
                    end_1 = {"+": True, "-": False}[end_1]
                    end_2 = {"+": False, "-": True}[end_2]
                    this_overlap = alignment_cigar.strip("M")
                    try:
                        this_overlap = int(this_overlap)
                    except:
                        raise ValueError(
                            "Contig uni_overlap cigar contains characters other than M: " + alignment_cigar)
                    overlap_values.add(this_overlap)
                    self.vertex_info[vertex_1].connections[end_1][(vertex_2, end_2)] = this_overlap
                    self.vertex_info[vertex_2].connections[end_2][(vertex_1, end_1)] = this_overlap

        # store uni_overlap score as either None or an int
        if len(overlap_values) == 0:
            self.__uni_overlap = None
        # elif len(overlap_values) > 1:
        #     raise ProcessingGraphFailed(
        #         "Multiple uni_overlap values: " + ",".join([str(ol) for ol in sorted(overlap_values)]))
        else:
            self.__uni_overlap = overlap_values.pop()

    def parse_fastg(self, min_cov=0., max_cov=INF):
        """
        Parse alternative graph format in FASTG format. Store results in self.vertex_info.
        """
        logger.info("Parsing graph (FASTG)")
        fastg_matrix = SequenceList(self.graph_file)
        # initialize names; only accept vertex that are formally stored, skip those that are only mentioned after ":"
        for i, seq in enumerate(fastg_matrix):
            if ":" in seq.label:
                this_vertex_str, next_vertices_str = seq.label.strip(";").split(":")
            else:
                this_vertex_str, next_vertices_str = seq.label.strip(";"), ""
            v_tag, vertex_name, l_tag, vertex_len, c_tag, vertex_cov = this_vertex_str.strip("'").split("_")
            # skip vertices with cov out of bounds
            vertex_cov = float(vertex_cov)
            if not (min_cov <= vertex_cov <= max_cov):
                continue
            if vertex_name not in self.vertex_info:
                self.vertex_info[vertex_name] = Vertex(vertex_name, int(vertex_len), vertex_cov,
                                                       fastg_form_long_name=this_vertex_str.strip("'"))
        # adding other info based on existed names
        for i, seq in enumerate(fastg_matrix):
            if ":" in seq.label:
                this_vertex_str, next_vertices_str = seq.label.strip(";").split(":")
            else:
                this_vertex_str, next_vertices_str = seq.label.strip(";"), ""
            v_tag, vertex_name, l_tag, vertex_len, c_tag, vertex_cov = this_vertex_str.strip("'").split("_")
            # skip vertices that not in self.vertex_info: 1. with cov out of bounds
            if vertex_name in self.vertex_info:
                # connections
                this_end = not this_vertex_str.endswith("'")
                if next_vertices_str:
                    for next_vertex_str in next_vertices_str.split(","):
                        next_name = next_vertex_str.strip("'").split("_")[1]
                        if next_name in self.vertex_info:
                            next_end = next_vertex_str.endswith("'")
                            # Adding connection information (edge) to both of the related vertices
                            # even it is only mentioned once in some SPAdes output files
                            # assign uni_overlap info latter
                            self.vertex_info[vertex_name].connections[this_end][(next_name, next_end)] = None
                            self.vertex_info[next_name].connections[next_end][(vertex_name, this_end)] = None
                # sequence
                if not self.vertex_info[vertex_name].seq[True]:
                    # self.vertex_info[vertex_name]["seq"] = {}
                    if this_end:
                        self.vertex_info[vertex_name].seq[True] = seq.seq
                        self.vertex_info[vertex_name].seq[False] = complementary_seq(seq.seq)
                    else:
                        self.vertex_info[vertex_name].seq[True] = complementary_seq(seq.seq)
                        self.vertex_info[vertex_name].seq[False] = seq.seq

        """detect general kmer"""
        ## find initial kmer candidate values
        initial_kmer = set()
        no_connection_at_all = True
        for vertex_name in self.vertex_info:
            if sum([len(self.vertex_info[vertex_name].connections[this_e]) for this_e in (True, False)]) != 0:
                no_connection_at_all = False
                for this_e in (True, False):
                    for next_name, next_end in self.vertex_info[vertex_name].connections[this_e]:
                        for test_k in range(21, 128, 2):
                            this_seq = self.vertex_info[vertex_name].seq[this_e][-test_k:]
                            next_seq = self.vertex_info[next_name].seq[not next_end][:test_k]
                            if this_seq == next_seq:
                                initial_kmer.add(test_k)
                        break
                    if initial_kmer:
                        break
            if initial_kmer:
                break
        if no_connection_at_all:
            self.__uni_overlap = 0
        else:
            ## check all edges
            testing_vertices = set(self.vertex_info)
            while initial_kmer and testing_vertices:
                vertex_name = testing_vertices.pop()
                for this_end in (True, False):
                    for next_name, next_end in self.vertex_info[vertex_name].connections[this_end]:
                        for test_k in list(initial_kmer):
                            this_seq = self.vertex_info[vertex_name].seq[this_end][-test_k:]
                            next_seq = self.vertex_info[next_name].seq[not next_end][:test_k]
                            if this_seq != next_seq:
                                initial_kmer.discard(test_k)
            if len(initial_kmer) >= 1:
                self.__uni_overlap = max(initial_kmer)
            else:
                self.__uni_overlap = 0
                # raise ProcessingGraphFailed("No kmer detected!")
        # assign general kmer to all edges
        for vertex_name in self.vertex_info:
            for this_end in (True, False):
                for next_tuple in self.vertex_info[vertex_name].connections[this_end]:
                    self.vertex_info[vertex_name].connections[this_end][next_tuple] = self.__uni_overlap

    def uni_overlap(self):
        if self.__uni_overlap is None:
            return None
        else:
            return int(self.__uni_overlap)

    def write_to_fasta(self, out_file, interleaved=None, check_postfix=True):
        if check_postfix and not out_file.endswith(".fasta"):
            out_file += ".fasta"
        out_matrix = SequenceList()
        for vertex_name in self.vertex_info:
            out_matrix.append(Sequence(vertex_name, self.vertex_info[vertex_name].seq[True]))
        out_matrix.interleaved = 70
        out_matrix.write_fasta(out_file, interleaved=interleaved)

    def write_to_gfa(self, out_file, check_postfix=True, other_attr=None):
        """
        :param out_file: str
        :param check_postfix: bool
        :param other_attr: dict, e.g. {"CL":"z", "C2":"z"}
        """
        if check_postfix and not out_file.endswith(".gfa"):
            out_file += ".gfa"
        if not other_attr:
            other_attr = {}
        out_file_handler = open(out_file, "w")
        for vertex_name in self.vertex_info:
            out_file_handler.write("\t".join(
                [
                    "S", vertex_name, self.vertex_info[vertex_name].seq[True],
                    "LN:i:" + str(self.vertex_info[vertex_name].len),
                    "RC:i:" + str(int(self.vertex_info[vertex_name].len * self.vertex_info[vertex_name].cov))] +
                [
                    "%s:%s:%s" % (attr_name, attr_type, self.vertex_info[vertex_name].other_attr.get(attr_name, ""))
                    for attr_name, attr_type in other_attr.items()
                    if self.vertex_info[vertex_name].other_attr.get(attr_name, False)
                ]) + "\n")
        recorded_connections = set()
        for vertex_name in self.vertex_info:
            for this_end in (False, True):
                for (next_v, next_e), this_overlap in self.vertex_info[vertex_name].connections[this_end].items():
                    this_con = tuple(sorted([(vertex_name, this_end), (next_v, next_e)]))
                    if this_con not in recorded_connections:
                        recorded_connections.add(this_con)
                        out_file_handler.write("\t".join([
                            "L", vertex_name, ("-", "+")[this_end], next_v, ("-", "+")[not next_e],
                            str(this_overlap) + "M"
                        ]) + "\n")

