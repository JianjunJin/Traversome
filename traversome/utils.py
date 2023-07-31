#!/usr/bin/env python

"""
Code copied and simplified from GetOrganelle.
"""
import os
import sys
from copy import deepcopy
from math import log, inf
from scipy import stats
from loguru import logger
from enum import Enum
from collections import OrderedDict
import numpy as np
import random
import subprocess
# from pathos.multiprocessing import ProcessingPool as Pool
import dill
from loguru import logger


# PY VERSION CHECK: only 2.7 and 3.5+ (this could be enforced by pip/conda)
MAJOR_VERSION, MINOR_VERSION = sys.version_info[:2]
if MAJOR_VERSION == 2 and MINOR_VERSION >= 7:
    RecursionError = RuntimeError
    import string
    PAIRING_TRANSLATOR = string.maketrans("ATGCRMYKHBDVatgcrmykhbdv", "TACGYKRMDVHBtacgykrmdvhb")
    def complementary_seq(input_seq):
        return string.translate(input_seq, PAIRING_TRANSLATOR)[::-1]

elif MAJOR_VERSION == 3 and MINOR_VERSION >= 5:
    # python3
    PAIRING_TRANSLATOR = str.maketrans("ATGCRMYKHBDVatgcrmykhbdv", "TACGYKRMDVHBtacgykrmdvhb")
    def complementary_seq(input_seq):
        return str.translate(input_seq, PAIRING_TRANSLATOR)[::-1]
else:
    sys.stdout.write("Python version have to be 2.7+ or 3.5+")
    sys.exit(0)

if MAJOR_VERSION == 3 and MINOR_VERSION >= 9:
    # functools.cache is a decorator that was added in Python 3.9 as a simple, lightweight, unbounded function cache
    from functools import cache
else:
    # functools.lru_cache is a decorator that was added in Python 3.2 to cache the results of function calls
    # using the Least Recently Used (LRU) strategy
    from functools import lru_cache

    # define a cache function that is equivalent to using lru_cache with maxsize=None
    def cache(user_function):
        return lru_cache(maxsize=None)(user_function)


def complementary_seqs(input_seq_iter):
    return tuple([complementary_seq(seq) for seq in input_seq_iter])


DEGENERATE_PAIRS = [["R", ("A", "G")],
                    ["Y", ("C", "T")],
                    ["M", ("A", "C")],
                    ["K", ("G", "T")],
                    ["S", ("C", "G")],
                    ["W", ("A", "T")],
                    ["H", ("A", "C", "T")],
                    ["B", ("C", "G", "T")],
                    ["V", ("A", "C", "G")],
                    ["D", ("A", "G", "T")],
                    ["N", ("A", "C", "G", "T")]]
REV_DEGENERATE = {__degenerated_base: __bases for __degenerated_base, __bases in DEGENERATE_PAIRS}
TO_DEGENERATE = {__bases: __degenerated_base for __degenerated_base, __bases in DEGENERATE_PAIRS}


########################################################################
###   GLOBALS
########################################################################

ECHO_DIRECTION = ["_tail", "_head"]
CLASSIC_START_CODONS = {"ATG", "ATC", "ATA", "ATT", "GTG", "TTG"}
CLASSIC_STOP_CODONS = {"TAA", "TAG", "TGA"}
INF = float("inf")


########################################################################
###   SIMPLE CLASSES
########################################################################


class Sequence(object):
    """
    Helper class for formatting sequences into FASTA format.
    """
    def __init__(self, label, seq):
        self.label = label
        self.seq = seq

    def __len__(self):
        return len(self.seq)

    def fasta_str(self, interleaved=False):
        out_str = []
        if interleaved:
            out_str.extend(['>', self.label, '\n'])
            j = interleaved
            while j < len(self):
                out_str.append(self.seq[(j - interleaved):j])
                out_str.append('\n')
                j += interleaved
            out_str.append(self.seq[(j - interleaved):j])
            out_str.append('\n')
        else:
            out_str = ['>', self.label, '\n', self.seq, "\n"]
        return "".join(out_str)


class SequenceList(object):
    """
    Helper class for formatting multiple sequences into multi-FASTA format.
    """

    def __init__(self, input_fasta_file=None, indexed=False):
        self.sequences = []
        self.interleaved = False
        self.__dict = {}
        if input_fasta_file:
            self.read_fasta(input_fasta_file)
            if indexed:
                for go_s, seq in enumerate(self.sequences):
                    self.__dict[seq.label] = go_s

    def __len__(self):
        return len(self.sequences)

    def __iter__(self):
        for seq in self.sequences:
            yield seq

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.sequences[self.__dict[item]]
        else:
            return self.sequences[item]

    def append(self, sequence):
        self.sequences.append(sequence)

    def remove(self, names):
        del_names = set(names)
        go_to = 0
        while go_to < len(self.sequences):
            if self.sequences[go_to].label in del_names:
                del_names.remove(self.sequences[go_to].label)
                del self.sequences[go_to]
            else:
                go_to += 1
        if del_names:
            sys.stdout.write("Warning: sequence(s) " + ",".join(sorted(del_names)) + " not found!\n")

    def read_fasta(self, fasta_file):
        fasta_file = open(fasta_file, 'r')
        this_line = fasta_file.readline()
        interleaved = 0
        while this_line:
            if this_line.startswith('>'):
                this_name = this_line[1:].strip()
                this_seq = ''
                this_line = fasta_file.readline()
                seq_line_count = 0
                while this_line and not this_line.startswith('>'):
                    if seq_line_count == 1:
                        interleaved = len(this_seq)
                    this_seq += this_line.strip()
                    this_line = fasta_file.readline()
                    seq_line_count += 1
                self.append(Sequence(this_name, this_seq))
            else:
                this_line = fasta_file.readline()
        fasta_file.close()
        self.interleaved = interleaved

    def write_fasta(self, fasta_file, overwrite=True, interleaved=None):
        if not overwrite:
            while os.path.exists(fasta_file):
                fasta_file = '.'.join(fasta_file.split('.')[:-1]) + '_.' + fasta_file.split('.')[-1]
        this_interleaved = self.interleaved if interleaved is None else interleaved
        fasta_file_handler = open(fasta_file, 'w')
        for seq in self:
            fasta_file_handler.write(seq.fasta_str(this_interleaved))
        fasta_file_handler.close()


class SubPathInfo(object):
    def __init__(self):
        self.from_variants = {}
        self.mapped_records = []
        self.num_possible_X = -1  # The X in binomial: theoretical num of matched chances
        self.num_possible_Xs = OrderedDict()  # For generating Xs in multinomial: theoretical num of matched chances
        self.num_in_range = -1  # The n in binomial: observed num of reads in range
        self.num_matched = -1  # The x in binomial: observed num of matched reads = len(self.mapped_records)


class LogLikeFormulaInfo(object):
    def __init__(self, loglike_expression=0, variable_size=0, sample_size=0):
        self.loglike_expression = loglike_expression
        self.variable_size = variable_size
        self.sample_size = sample_size


class LogLikeFuncInfo(object):
    def __init__(self, loglike_func=0, variable_size=0, sample_size=0):
        self.loglike_func = loglike_func
        self.variable_size = variable_size
        self.sample_size = sample_size


class Criterion(str, Enum):
    AIC = "aic"
    BIC = "bic"  # BIC is actually unnecessary in this framework because observations will be always the same.


class WeightedGMMWithEM:
    def __init__(self,
                 data_array,
                 data_weights=None,
                 minimum_cluster=1,
                 maximum_cluster=5,
                 min_sigma_factor=1E-5,
                 cluster_limited=None):
        """
        :param data_array:
        :param data_weights:
        :param minimum_cluster:
        :param maximum_cluster:
        :param min_sigma_factor:
        :param cluster_limited: {dat_id1: {0, 1}, dat_id2: {0}, dat_id3: {0} ...}
        :param log_handler:
        :param verbose_log:
        :return:
        """
        self.data_array = np.array(data_array)
        self.data_len = len(self.data_array)
        self.min_sigma = min_sigma_factor * np.average(data_array, weights=data_weights)
        self.data_weights = None
        if not data_weights:
            self.data_weights = np.array([1. for foo in range(self.data_len)])
        else:
            assert len(data_weights) == self.data_len
            average_weights = float(sum(data_weights)) / self.data_len
            # normalized
            self.data_weights = np.array([raw_w / average_weights for raw_w in data_weights])
        self.cluster_limited = cluster_limited
        self.freedom_dat_item = None
        if cluster_limited:
            cls = set()
            for sub_cls in cluster_limited.values():
                cls |= sub_cls
            self.freedom_dat_item = self.data_len - len(cluster_limited) + len(cls)
        else:
            self.freedom_dat_item = self.data_len
        self.minimum_cluster = min(self.freedom_dat_item, minimum_cluster)
        self.maximum_cluster = min(self.freedom_dat_item, maximum_cluster)

    def run(self, criteria="bic"):
        assert criteria in ("aic", "bic")
        results = []
        for total_cluster_num in range(self.minimum_cluster, self.maximum_cluster + 1):
            # initialization
            labels = np.random.choice(total_cluster_num, self.data_len)
            if self.cluster_limited:
                temp_labels = []
                for dat_id in range(self.data_len):
                    if dat_id in self.cluster_limited:
                        if labels[dat_id] in self.cluster_limited[dat_id]:
                            temp_labels.append(labels[dat_id])
                        else:
                            temp_labels.append(sorted(self.cluster_limited[dat_id])[0])
                    else:
                        temp_labels.append(labels[dat_id])
                labels = np.array(temp_labels)
            norm_parameters = self.updating_parameter(self.data_array, self.data_weights, labels,
                                                 [{"mu": 0, "sigma": 1, "percent": total_cluster_num / self.data_len}
                                                  for foo in range(total_cluster_num)])
            loglike_shift = inf
            prev_loglike = -inf
            epsilon = 0.01
            count_iterations = 0
            best_loglike = prev_loglike
            best_parameter = norm_parameters
            try:
                while loglike_shift > epsilon:
                    count_iterations += 1
                    # expectation
                    labels = self.assign_cluster_labels(
                        self.data_array, self.data_weights, norm_parameters, self.cluster_limited)
                    # maximization
                    updated_parameters = self.updating_parameter(
                        self.data_array, self.data_weights, labels, deepcopy(norm_parameters))
                    # loglike shift
                    this_loglike = self.model_loglike(
                        self.data_array, self.data_weights, labels, updated_parameters)
                    loglike_shift = abs(this_loglike - prev_loglike)
                    # update
                    prev_loglike = this_loglike
                    norm_parameters = updated_parameters
                    if this_loglike > best_loglike:
                        best_parameter = updated_parameters
                        best_loglike = this_loglike
                labels = self.assign_cluster_labels(self.data_array, self.data_weights, best_parameter, None)
                results.append({"loglike": best_loglike, "iterates": count_iterations, "cluster_num": total_cluster_num,
                                "parameters": best_parameter, "labels": labels,
                                "aic": aic(prev_loglike, 2 * total_cluster_num),
                                "bic": bic(prev_loglike, 2 * total_cluster_num, self.data_len)})
            except TypeError as e:
                logger.error("This error might be caused by outdated version of scipy!")
                raise e
        logger.debug(str(results))
        best_scheme = sorted(results, key=lambda x: x[criteria])[0]
        return best_scheme

    def model_loglike(self, dat_arr, dat_w, lbs, parameters):
        total_loglike = 0
        for go_to_cl, pr in enumerate(parameters):
            points = dat_arr[lbs == go_to_cl]
            weights = dat_w[lbs == go_to_cl]
            if len(points):
                total_loglike += sum(stats.norm.logpdf(points, pr["mu"], pr["sigma"]) * weights + log(pr["percent"]))
        return total_loglike

    def assign_cluster_labels(self, dat_arr, dat_w, parameters, limited):
        # assign every data point to its most likely cluster
        if len(parameters) == 1:
            return np.array([0] * self.data_len)
        else:
            # the parameter set of the first cluster
            loglike_res = stats.norm.logpdf(dat_arr, parameters[0]["mu"], parameters[1]["sigma"]) * dat_w + \
                          log(parameters[1]["percent"])
            # the parameter set of the rest cluster
            for pr in parameters[1:]:
                loglike_res = np.vstack(
                    (loglike_res, stats.norm.logpdf(dat_arr, pr["mu"], pr["sigma"]) * dat_w + log(pr["percent"])))
            # assign labels
            new_labels = loglike_res.argmax(axis=0)
            if limited:
                intermediate_labels = []
                for here_dat_id in range(self.data_len):
                    if here_dat_id in limited:
                        if new_labels[here_dat_id] in limited[here_dat_id]:
                            intermediate_labels.append(new_labels[here_dat_id])
                        else:
                            intermediate_labels.append(sorted(limited[here_dat_id])[0])
                    else:
                        intermediate_labels.append(new_labels[here_dat_id])
                new_labels = np.array(intermediate_labels)
                # new_labels = np.array([
                # sorted(cluster_limited[dat_item])[0]
                # if new_labels[here_dat_id] not in cluster_limited[dat_item] else new_labels[here_dat_id]
                # if dat_item in cluster_limited else
                # new_labels[here_dat_id]
                # for here_dat_id, dat_item in enumerate(data_array)])
                limited_values = set(dat_arr[list(limited)])
            else:
                limited_values = set()
            # re-pick if some cluster are empty
            label_counts = {lb: 0 for lb in range(len(parameters))}
            for ct_lb in new_labels:
                label_counts[ct_lb] += 1
            for empty_lb in label_counts:
                if label_counts[empty_lb] == 0:
                    affordable_lbs = {af_lb: [min, max] for af_lb in label_counts if label_counts[af_lb] > 1}
                    for af_lb in sorted(affordable_lbs):
                        these_points = dat_arr[new_labels == af_lb]
                        if max(these_points) in limited_values:
                            affordable_lbs[af_lb].remove(max)
                        if min(these_points) in limited_values:
                            affordable_lbs[af_lb].remove(min)
                        if not affordable_lbs[af_lb]:
                            del affordable_lbs[af_lb]
                    if affordable_lbs:
                        chose_lb = random.choice(list(affordable_lbs))
                        chose_points = dat_arr[new_labels == chose_lb]
                        data_point = random.choice(affordable_lbs[chose_lb])(chose_points)
                        transfer_index = np.where(dat_arr == data_point)[0]
                        new_labels[transfer_index] = empty_lb
                        label_counts[chose_lb] -= len(transfer_index)
            return new_labels

    def updating_parameter(self, dat_arr, dat_w, lbs, parameters):

        for go_to_cl, pr in enumerate(parameters):
            these_points = dat_arr[lbs == go_to_cl]
            these_weights = dat_w[lbs == go_to_cl]
            if len(these_points) > 1:
                this_mean, this_std = weighted_mean_and_std(these_points, these_weights)
                pr["mu"] = this_mean
                pr["sigma"] = max(this_std, self.min_sigma)
                pr["percent"] = sum(these_weights)  # / data_len
            elif len(these_points) == 1:
                pr["sigma"] = max(dat_arr.std() / self.data_len, self.min_sigma)
                pr["mu"] = np.average(these_points, weights=these_weights) + pr["sigma"] * (2 * random.random() - 1)
                pr["percent"] = sum(these_weights)  # / data_len
            else:
                # exclude
                pr["mu"] = max(dat_arr) * 1E4
                pr["sigma"] = self.min_sigma
                pr["percent"] = 1E-10
        return parameters


# TODO get subpath adaptive to length=1, more general and less restrictions
class VariantSubPathsGenerator:
    def __init__(self, graph, force_circular, min_alignment_len, max_alignment_len, read_paths_hashed):
        self.graph = graph
        self.force_circular = force_circular
        self.min_alignment_len = min_alignment_len
        self.max_alignment_len = max_alignment_len
        self.read_paths_hashed = read_paths_hashed
        self.variant_subpath_counters = {}

    @cache
    def gen_subpaths(self, variant_path):
        if variant_path in self.variant_subpath_counters:
            return self.variant_subpath_counters[variant_path]
        else:
            # if this_overlap is None:
            #     this_overlap = self.graph.uni_overlap()
            these_sub_paths = dict()
            num_seg = len(variant_path)
            # print("run get")
            if self.force_circular:
                for go_start_v, start_segment in enumerate(variant_path):
                    # find the longest sub_path,
                    # that begins with start_segment and be in the range of alignment length
                    this_longest_sub_path = [start_segment]
                    this_internal_path_len = 0
                    go_next = (go_start_v + 1) % num_seg
                    while this_internal_path_len < self.max_alignment_len:
                        next_n, next_e = variant_path[go_next]
                        next_v_info = self.graph.vertex_info[next_n]
                        pre_n, pre_e = this_longest_sub_path[-1]
                        this_overlap = next_v_info.connections[not next_e][(pre_n, pre_e)]
                        this_longest_sub_path.append((next_n, next_e))
                        this_internal_path_len += next_v_info.len - this_overlap
                        go_next = (go_next + 1) % num_seg
                    # print("this_longest_sub_path", this_longest_sub_path)
                    # print(self.graph.get_path_internal_length(this_longest_sub_path), self.min_alignment_len)
                    # when the overlap is long and the contig is short,
                    # the path with internal_length shorter than tha alignment length can still help
                    # so remove the condition for min_alignment_len
                    # if len(this_longest_sub_path) < 2 \
                    #         or self.graph.get_path_internal_length(this_longest_sub_path) < self.min_alignment_len:
                    #     continue
                    # TODO size of 1 can also be included
                    if len(this_longest_sub_path) < 2:
                        continue
                    # print("this_longest_sub_path", this_longest_sub_path, "passed")

                    # record shorter sub_paths starting from start_segment
                    len_this_sub_p = len(this_longest_sub_path)
                    for skip_tail in range(len_this_sub_p - 1):
                        this_sub_path = \
                            self.graph.get_standardized_path(this_longest_sub_path[:len_this_sub_p - skip_tail])
                        # print("checking subpath existence", this_sub_path)
                        if this_sub_path not in self.read_paths_hashed:
                            continue
                        # print("checking subpath existence", this_sub_path, "passed")
                        # when the uni_overlap is long and the contig is short,
                        # the path with internal_length shorter than tha alignment length can still help
                        # so remove the condition for min_alignment_len
                        # if self.graph.get_path_internal_length(this_sub_path) < self.min_alignment_len:
                        #     break
                        if this_sub_path not in these_sub_paths:
                            these_sub_paths[this_sub_path] = 0
                        these_sub_paths[this_sub_path] += 1
            else:
                for go_start_v, start_segment in enumerate(variant_path):
                    # find the longest sub_path,
                    # that begins with start_segment and be in the range of alignment length
                    this_longest_sub_path = [start_segment]
                    this_internal_path_len = 0
                    go_next = go_start_v + 1
                    while go_next < num_seg and this_internal_path_len < self.max_alignment_len:
                        next_n, next_e = variant_path[go_next]
                        next_v_info = self.graph.vertex_info[next_n]
                        pre_n, pre_e = this_longest_sub_path[-1]
                        this_overlap = next_v_info.connections[not next_e][(pre_n, pre_e)]
                        this_longest_sub_path.append((next_n, next_e))
                        this_internal_path_len += next_v_info.len - this_overlap
                        go_next += 1
                    if len(this_longest_sub_path) < 2 \
                            or self.graph.get_path_internal_length(this_longest_sub_path) < self.min_alignment_len:
                        continue
                    # record shorter sub_paths starting from start_segment
                    len_this_sub_p = len(this_longest_sub_path)
                    for skip_tail in range(len_this_sub_p - 1):
                        this_sub_path = \
                            self.graph.get_standardized_path_circ(this_longest_sub_path[:len_this_sub_p - skip_tail])
                        if this_sub_path not in self.read_paths_hashed:
                            continue
                        if self.graph.get_path_internal_length(this_sub_path) < self.min_alignment_len:
                            break
                        if this_sub_path not in these_sub_paths:
                            these_sub_paths[this_sub_path] = 0
                        these_sub_paths[this_sub_path] += 1
            self.variant_subpath_counters[variant_path] = these_sub_paths
            return these_sub_paths


class ProcessingGraphFailed(Exception):
    def __init__(self, value=""):
        self.value = value

    def __str__(self):
        return repr(self.value)


# class MaxTraversalReached(Exception):
#     def __init__(self, value=""):
#         self.value = value
#
#     def __str__(self):
#         return repr(self.value)


########################################################################
###   FUNCTIONS OUTSIDE OF CLASSES
########################################################################

# def generate_clusters_from_connections(vertices, connections):
#     """
#     :param vertices: list or set
#     :param connections: symmetric raw_records, e.g.
#                         {"vertex_1": ["vertex_2", "vertex_3"],
#                          "vertex_2": ["vertex_1"],
#                          "vertex_3": ["vertex_1", "vertex_5"],
#                          "vertex_4": [],
#                          "vertex_5": ["vertex_3"]}
#     :return: e.g. [{"vertex_1", "vertex_2", "vertex_3", "vertex_5"}, {"vertex_4"}]
#     """
#     # Each cluster is a connected set of vertices.
#     vertex_clusters = []
#
#     # iterate over vertices
#     for this_vertex in sorted(vertices):
#
#         # build a set of connections (edges) from this vertex to established clusters
#         connecting_those = set()
#         for next_v in connections.get(this_vertex, []):
#             # for connected_set in self.vertex_info[this_vertex].connections.values():
#             #     for next_v, next_d in connected_set:
#             for go_to_set, cluster in enumerate(vertex_clusters):
#                 if next_v in cluster:
#                     connecting_those.add(go_to_set)
#
#         # if no edges then store just this one
#         if not connecting_those:
#             vertex_clusters.append({this_vertex})
#
#         # if 1 then add this one to its cluster.
#         elif len(connecting_those) == 1:
#             vertex_clusters[connecting_those.pop()].add(this_vertex)
#
#         # if many then ...
#         else:
#             sorted_those = sorted(connecting_those, reverse=True)
#             vertex_clusters[sorted_those[-1]].add(this_vertex)
#             for go_to_set in sorted_those[:-1]:
#                 for that_vertex in vertex_clusters[go_to_set]:
#                     vertex_clusters[sorted_those[-1]].add(that_vertex)
#                 del vertex_clusters[go_to_set]
#     return vertex_clusters


def generate_clusters_from_connections(vertices, connections):
    """
    faster than above algorithm. 2022-12-18.
    :param vertices: list or set
    :param connections: symmetric records, e.g.
                        {"vertex_1": ["vertex_2", "vertex_3"],
                         "vertex_2": ["vertex_1"],
                         "vertex_3": ["vertex_1", "vertex_5"],
                         "vertex_4": [],
                         "vertex_5": ["vertex_3"]}
    :return: e.g. [{"vertex_1", "vertex_2", "vertex_3", "vertex_5"}, {"vertex_4"}]
    """
    vertex_clusters = []
    candidate_vs = set(vertices)
    while candidate_vs:
        new_root = candidate_vs.pop()
        vertex_clusters.append({new_root})
        waiting_vs = set([next_v
                          for next_v in connections[new_root]
                          if next_v in candidate_vs])
        while candidate_vs and waiting_vs:
            next_v = waiting_vs.pop()
            vertex_clusters[-1].add(next_v)
            candidate_vs.discard(next_v)
            for n_next_v in connections[next_v]:
                if n_next_v in candidate_vs:
                    waiting_vs.add(n_next_v)
    # for reproducible, not necessary for some cases
    vertex_clusters.sort(key=lambda x: max(x))
    return vertex_clusters


def find_greatest_common_divisor(number_list):
    "euclid_algorithm"
    number_list = number_list[:]
    if len(number_list) == 1:
        return number_list[0]
    elif len(number_list) == 0:
        return
    else:
        a = number_list[0]
        for i in range(len(number_list) - 1):
            a = number_list[i]
            b = number_list[i + 1]
            while b:
                a, b = b, a % b
            number_list[i + 1] = a
        return a


# divide numbers by their greatest common divisor
def reduce_list_with_gcd(number_list):
    if len(number_list) == 1:
        return [1] if number_list[0] != 0 else number_list
    elif len(number_list) == 0:
        return []
    else:
        gcd_num = find_greatest_common_divisor(number_list)
        return [int(raw_number / gcd_num) for raw_number in number_list]


def bic(loglike, len_param, len_data):
    return log(len_data) * len_param - 2 * loglike


def aic(loglike, len_param):
    return 2 * len_param - 2 * loglike


def weighted_mean_and_std(values, weights):
    mean = np.average(values, weights=weights)
    std = np.average((values-mean)**2, weights=weights)**0.5
    return mean, std


def weighted_gmm_with_em_aic(
    data_array, 
    data_weights=None, 
    minimum_cluster=1, 
    maximum_cluster=5, 
    min_sigma_factor=1E-5,
    cluster_limited=None, 
    log_handler=None, 
    verbose_log=False):
    """
    :param data_array:
    :param data_weights:
    :param minimum_cluster:
    :param maximum_cluster:
    :param min_sigma_factor:
    :param cluster_limited: {dat_id1: {0, 1}, dat_id2: {0}, dat_id3: {0} ...}
    :param log_handler:
    :param verbose_log:
    :return:
    """
    min_sigma = min_sigma_factor * np.average(data_array, weights=data_weights)

    def model_loglike(dat_arr, dat_w, lbs, parameters):
        total_loglike = 0
        for go_to_cl, pr in enumerate(parameters):
            points = dat_arr[lbs == go_to_cl]
            weights = dat_w[lbs == go_to_cl]
            if len(points):
                total_loglike += sum(stats.norm.logpdf(points, pr["mu"], pr["sigma"]) * weights + log(pr["percent"]))
        return total_loglike

    def assign_cluster_labels(dat_arr, dat_w, parameters, limited):
        # assign every data point to its most likely cluster
        if len(parameters) == 1:
            return np.array([0] * int(data_len))
        else:
            # the parameter set of the first cluster
            loglike_res = stats.norm.logpdf(dat_arr, parameters[0]["mu"], parameters[1]["sigma"]) * dat_w + \
                          log(parameters[1]["percent"])
            # the parameter set of the rest cluster
            for pr in parameters[1:]:
                loglike_res = np.vstack(
                    (loglike_res, stats.norm.logpdf(dat_arr, pr["mu"], pr["sigma"]) * dat_w + log(pr["percent"])))
            # assign labels
            new_labels = loglike_res.argmax(axis=0)
            if limited:
                intermediate_labels = []
                for here_dat_id in range(int(data_len)):
                    if here_dat_id in limited:
                        if new_labels[here_dat_id] in limited[here_dat_id]:
                            intermediate_labels.append(new_labels[here_dat_id])
                        else:
                            intermediate_labels.append(sorted(limited[here_dat_id])[0])
                    else:
                        intermediate_labels.append(new_labels[here_dat_id])
                new_labels = np.array(intermediate_labels)
                # new_labels = np.array([
                # sorted(cluster_limited[dat_item])[0]
                # if new_labels[here_dat_id] not in cluster_limited[dat_item] else new_labels[here_dat_id]
                # if dat_item in cluster_limited else
                # new_labels[here_dat_id]
                # for here_dat_id, dat_item in enumerate(data_array)])
                limited_values = set(dat_arr[list(limited)])
            else:
                limited_values = set()
            # re-pick if some cluster are empty
            label_counts = {lb: 0 for lb in range(len(parameters))}
            for ct_lb in new_labels:
                label_counts[ct_lb] += 1
            for empty_lb in label_counts:
                if label_counts[empty_lb] == 0:
                    affordable_lbs = {af_lb: [min, max] for af_lb in label_counts if label_counts[af_lb] > 1}
                    for af_lb in sorted(affordable_lbs):
                        these_points = dat_arr[new_labels == af_lb]
                        if max(these_points) in limited_values:
                            affordable_lbs[af_lb].remove(max)
                        if min(these_points) in limited_values:
                            affordable_lbs[af_lb].remove(min)
                        if not affordable_lbs[af_lb]:
                            del affordable_lbs[af_lb]
                    if affordable_lbs:
                        chose_lb = random.choice(list(affordable_lbs))
                        chose_points = dat_arr[new_labels == chose_lb]
                        data_point = random.choice(affordable_lbs[chose_lb])(chose_points)
                        transfer_index = np.where(dat_arr == data_point)[0]
                        new_labels[transfer_index] = empty_lb
                        label_counts[chose_lb] -= len(transfer_index)
            return new_labels

    def updating_parameter(dat_arr, dat_w, lbs, parameters):

        for go_to_cl, pr in enumerate(parameters):
            these_points = dat_arr[lbs == go_to_cl]
            these_weights = dat_w[lbs == go_to_cl]
            if len(these_points) > 1:
                this_mean, this_std = weighted_mean_and_std(these_points, these_weights)
                pr["mu"] = this_mean
                pr["sigma"] = max(this_std, min_sigma)
                pr["percent"] = sum(these_weights)  # / data_len
            elif len(these_points) == 1:
                pr["sigma"] = max(dat_arr.std() / data_len, min_sigma)
                pr["mu"] = np.average(these_points, weights=these_weights) + pr["sigma"] * (2 * random.random() - 1)
                pr["percent"] = sum(these_weights)  # / data_len
            else:
                # exclude
                pr["mu"] = max(dat_arr) * 1E4
                pr["sigma"] = min_sigma
                pr["percent"] = 1E-10
        return parameters

    data_array = np.array(data_array)
    data_len = float(len(data_array))
    if not len(data_weights):
        data_weights = np.array([1. for foo in range(int(data_len))])
    else:
        assert len(data_weights) == data_len
        average_weights = float(sum(data_weights)) / data_len
        # normalized
        data_weights = np.array([raw_w / average_weights for raw_w in data_weights])

    results = []
    if cluster_limited:
        cls = set()
        for sub_cls in cluster_limited.values():
            cls |= sub_cls
        freedom_dat_item = int(data_len) - len(cluster_limited) + len(cls)
    else:
        freedom_dat_item = int(data_len)
    minimum_cluster = min(freedom_dat_item, minimum_cluster)
    maximum_cluster = min(freedom_dat_item, maximum_cluster)
    for total_cluster_num in range(minimum_cluster, maximum_cluster + 1):
        # initialization
        labels = np.random.choice(total_cluster_num, int(data_len))
        if cluster_limited:
            temp_labels = []
            for dat_id in range(int(data_len)):
                if dat_id in cluster_limited:
                    if labels[dat_id] in cluster_limited[dat_id]:
                        temp_labels.append(labels[dat_id])
                    else:
                        temp_labels.append(sorted(cluster_limited[dat_id])[0])
                else:
                    temp_labels.append(labels[dat_id])
            labels = np.array(temp_labels)
        norm_parameters = updating_parameter(data_array, data_weights, labels,
                                             [{"mu": 0, "sigma": 1, "percent": total_cluster_num/data_len}
                                              for foo in range(total_cluster_num)])
        loglike_shift = INF
        prev_loglike = -INF
        epsilon = 0.01
        count_iterations = 0
        best_loglike = prev_loglike
        best_parameter = norm_parameters
        try:
            while loglike_shift > epsilon:
                count_iterations += 1
                # expectation
                labels = assign_cluster_labels(data_array, data_weights, norm_parameters, cluster_limited)
                # maximization
                updated_parameters = updating_parameter(data_array, data_weights, labels, deepcopy(norm_parameters))
                # loglike shift
                this_loglike = model_loglike(data_array, data_weights, labels, updated_parameters)
                loglike_shift = abs(this_loglike - prev_loglike)
                # update
                prev_loglike = this_loglike
                norm_parameters = updated_parameters
                if this_loglike > best_loglike:
                    best_parameter = updated_parameters
                    best_loglike = this_loglike
            labels = assign_cluster_labels(data_array, data_weights, best_parameter, None)
            results.append({"loglike": best_loglike, "iterates": count_iterations, "cluster_num": total_cluster_num,
                            "parameters": best_parameter, "labels": labels,
                            "aic": aic(prev_loglike, 2 * total_cluster_num),
                            "bic": bic(prev_loglike, 2 * total_cluster_num, data_len)})
        except TypeError as e:
            if log_handler:
                log_handler.error("This error might be caused by outdated version of scipy!")
            else:
                sys.stdout.write("This error might be caused by outdated version of scipy!\n")
            raise e
    if verbose_log:
        if log_handler:
            log_handler.info(str(results))
        else:
            sys.stdout.write(str(results) + "\n")
    best_scheme = sorted(results, key=lambda x: x["bic"])[0]
    return best_scheme


def generate_index_combinations(index_list):
    if not index_list:
        yield []
    else:
        for go_id in range(index_list[0]):
            for next_ids in generate_index_combinations(index_list[1:]):
                yield [go_id] + next_ids

#
#
# def smart_trans_for_sort(candidate_item):
#     if type(candidate_item) in (tuple, list):
#         return [smart_trans_for_sort(this_sub) for this_sub in candidate_item]
#     elif type(candidate_item) == bool:
#         return not candidate_item
#     else:
#         all_e = candidate_item.split("_")
#         for go_e, this_ele in enumerate(all_e):
#             try:
#                 all_e[go_e] = int(this_ele)
#             except ValueError:
#                 try:
#                     all_e[go_e] = float(this_ele)
#                 except ValueError:
#                     pass
#         return all_e


def get_orf_lengths(sequence_string, threshold=200, which_frame=None,
                    here_stop_codons=None, here_start_codons=None):
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


def get_id_range_in_increasing_values(min_num, max_num, increasing_numbers):
    assert max_num >= min_num
    assert min_num <= increasing_numbers[-1], \
        "minimum value {} out of range {}..{}".format(min_num, increasing_numbers[0], increasing_numbers[-1])
    assert max_num >= increasing_numbers[0], \
        "maximum value {} out of range {}..{}".format(max_num, increasing_numbers[0], increasing_numbers[-1])
    len_list = len(increasing_numbers)
    left_id = 0
    while left_id < len_list and increasing_numbers[left_id] < min_num:
        left_id += 1
    right_id = len_list - 1
    while right_id > -1 and increasing_numbers[right_id] > max_num:
        right_id -= 1
    return left_id, right_id


def generate_align_len_lookup_table(align_len_at_path_sorted):
    min_alignment_length = min(align_len_at_path_sorted)
    max_alignment_length = max(align_len_at_path_sorted)
    its_left_id = 0
    its_right_id = 0
    max_id = len(align_len_at_path_sorted) - 1
    align_len_lookup_table = \
        {potential_len:
             {"its_left_id": None, "its_right_id": None, "as_left_lim_id": None, "as_right_lim_id": None}
         for potential_len in range(min_alignment_length, max_alignment_length + 1)}
    for potential_len in range(min_alignment_length, max_alignment_length + 1):
        if potential_len == align_len_at_path_sorted[its_right_id]:
            align_len_lookup_table[potential_len]["as_left_lim_id"] = \
                align_len_lookup_table[potential_len]["its_left_id"] = its_left_id = its_right_id
            while potential_len == align_len_at_path_sorted[its_right_id]:
                align_len_lookup_table[potential_len]["its_right_id"] = its_right_id
                align_len_lookup_table[potential_len]["as_right_lim_id"] = its_right_id
                if its_right_id == max_id:
                    break
                else:
                    its_left_id = its_right_id
                    its_right_id += 1
        else:
            align_len_lookup_table[potential_len]["its_right_id"] = its_right_id
            align_len_lookup_table[potential_len]["as_left_lim_id"] = its_right_id
            align_len_lookup_table[potential_len]["its_left_id"] = its_left_id
            align_len_lookup_table[potential_len]["as_right_lim_id"] = its_left_id
    return align_len_lookup_table


def harmony_weights(raw_weights, diff):
    weights = np.array(raw_weights)
    weights_trans = weights**diff
    return weights_trans / sum(weights_trans)


# following the solution using dill: https://stackoverflow.com/a/24673524
def run_dill_encoded(payload):
    fun, args = dill.loads(payload)
    return fun(*args)


DEAD_CODES = (2, 126, 127)  # python 3.5+


def executable(test_this):
    return True if subprocess.getstatusoutput(test_this)[0] not in DEAD_CODES else False


def run_graph_aligner(
        graph_file: str,
        seq_file: str,
        alignment_file: str,
        num_processes: int = 1):
    logger.info("Making alignment using GraphAligner ..")
    this_command = os.path.join("", "GraphAligner") + \
                   " -g " + graph_file + " -f " + seq_file + " --multimap-score-fraction 0.95" + \
                   " -x vg -t " + str(num_processes) + \
                   " -a " + alignment_file + ".tmp.gaf"
    logger.debug(this_command)
    ga_run = subprocess.Popen(this_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
    output, err = ga_run.communicate()
    if "error" in output.decode("utf8").lower() or "(ERR)" in output.decode("utf8"):
        logger.error(output.decode("utf8"))
        exit()
    else:
        os.rename(alignment_file + ".tmp.gaf", alignment_file)


