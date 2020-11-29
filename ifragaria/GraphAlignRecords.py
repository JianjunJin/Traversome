#!/usr/bin/env python

"""
Class objects to store Graph Alignments 
"""


import csv
import re
import sys
from loguru import logger
from .Assembly import Assembly  # used here to validate type



CONVERT_QUERY_STRAND = {"+": True, "-": False}
CIGAR_ALPHA_REG = "([MIDNSHPX=])"



class GAFRecord(object):
    """
    Multiple GAFRecord objects make up a GraphAlignRecords object.
    ref: http://www.liheng.org/downloads/rGFA-GAF.pdf
    """
    def __init__(self, record_line_split, parse_cigar=False):

        # store information
        self.query_name = record_line_split[0]
        self.query_len = int(record_line_split[1])
        self.q_start = int(record_line_split[2])
        self.q_end = int(record_line_split[3])
        self.q_strand = CONVERT_QUERY_STRAND[record_line_split[4]]
        self.path_str = record_line_split[5]
        self.path = self.parse_gaf_path()
        self.p_len = int(record_line_split[6])
        self.p_start = int(record_line_split[7])
        self.p_end = int(record_line_split[8])
        self.p_align_len = self.p_end - self.p_start
        self.num_match = int(record_line_split[9])
        self.align_len = int(record_line_split[10])
        self.align_quality = int(record_line_split[11])
        self.optional_fields = {}
        
        # ...
        for flag_type_val in record_line_split[12:]:
            op_flag, op_type, op_val = flag_type_val.split(":")
            if op_type == "i":
                self.optional_fields[op_flag] = int(op_val)
            elif op_type == "Z":
                self.optional_fields[op_flag] = op_val
            elif op_type == "f":
                self.optional_fields[op_flag] = float(op_val)
        if parse_cigar and "cg" in self.optional_fields:
            self.cigar = self.split_cigar_str()
        else:
            self.cigar = None
        self.identity = self.optional_fields.get("id", self.num_match / float(self.align_len))


    def parse_gaf_path(self):
        path_list = []
        for segment in re.findall(r".[^\s><]*", self.path_str):
            if segment[0] == ">":
                path_list.append((segment[1:], True))
            elif segment[0] == "<":
                path_list.append((segment[1:], False))
            else:
                path_list.append((segment, True))
        return path_list


    def split_cigar_str(self):
        cigar_str = self.optional_fields['cg']
        cigar_split = re.split(CIGAR_ALPHA_REG, cigar_str)[:-1]  # empty end
        cigar_list = []
        for go_part in range(0, len(cigar_split), 2):
            cigar_list.append((int(cigar_split[go_part]), cigar_split[go_part + 1]))
        return cigar_list




class GraphAlignRecords(object):
    """
    Stores GraphAlign records...
 
    Parameters
    ----------
    gaf_file (str):
        path to a GAF file.
    parse_cigar (bool):
        parsing CIGARs allows for ... default=False.
    min_aligned_path_len (int):
        ...
    """
    def __init__(
        self, 
        gaf_file, 
        parse_cigar=False, 
        min_aligned_path_len=0, 
        min_align_len=0, 
        min_identity=0.,
        trim_overlap_with_graph=False, 
        assembly_graph=None, 
        log_handler=None):

        # store params to self
        self.gaf_file = gaf_file
        self.parse_cigar = parse_cigar
        self.min_align_len = min_align_len
        self.min_aligned_path_len = min_aligned_path_len
        self.min_identity = min_identity
        self.trim_overlap_with_graph = trim_overlap_with_graph
        self.assembly_graph = assembly_graph
        self.log_handler = log_handler

        # destination for parsed results
        self.records = []

        # run the parsing function
        logger.debug("Parsing GAF to GraphAlignRecords (.alignment)")
        self.parse_gaf()


    def parse_gaf(self):
        """

        """

        # store a list of GAFRecord objects made for each line in GAF file.
        with open(self.gaf_file) as input_f:
            for line_split in csv.reader(input_f, delimiter="\t"):
                gaf = GAFRecord(line_split, parse_cigar=self.parse_cigar)
                self.records.append(gaf)

        # filtering GAF records based on min length
        if self.min_aligned_path_len:
            go_r = 0
            while go_r < len(self.records):
                if self.records[go_r].p_align_len < self.min_aligned_path_len:
                    del self.records[go_r]
                else:
                    go_r += 1

        # filtering GAF records based on min length
        if self.min_align_len > self.min_aligned_path_len:
            go_r = 0
            while go_r < len(self.records):
                if self.records[go_r].align_len < self.min_align_len:
                    del self.records[go_r]
                else:
                    go_r += 1

        # filtering GAF records by min identity
        if self.min_identity:
            go_r = 0
            while go_r < len(self.records):
                if self.records[go_r].identity < self.min_identity:
                    del self.records[go_r]
                else:
                    go_r += 1

        # filtering GAF records by overlap requirement
        if self.trim_overlap_with_graph:
            
            # check that assembly_graph is an Assembly class object
            check1 = isinstance(self.assembly_graph, Assembly)
            check2 = self.assembly_graph.overlap()

            # iterate over ... and do ...
            if check1 and check2: 
                this_overlap = self.assembly_graph.overlap()
                go_r = 0
                while go_r < len(self.records):
                    this_record = self.records[go_r]
                    if len(this_record.path) > 1:
                        # if path did not reach out the overlap region between the terminal vertex and
                        # the neighboring internal vertex, the terminal vertex should be trimmed from the path
                        head_vertex_len = self.assembly_graph.vertex_info[this_record.path[0][0]].len
                        tail_vertex_len = self.assembly_graph.vertex_info[this_record.path[-1][0]].len
                        if head_vertex_len - this_record.p_start - 1 <= this_overlap:
                            del this_record.path[0]
                        if tail_vertex_len - (this_record.p_len - this_record.p_end - 1) <= this_overlap:
                            del this_record.path[-1]
                        if not this_record.path:
                            del self.records[go_r]
                        else:
                            go_r += 1
                    else:
                        go_r += 1

        else:
            logger.warning("assembly graph not available, overlaps untrimmed")

