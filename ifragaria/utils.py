#!/usr/bin/env python

"""
Code copied and simplified from GetOrganelle.
"""
import os
import sys
from itertools import combinations, product
from hashlib import sha256
from collections import OrderedDict
from copy import deepcopy
from math import log, ceil
# from sympy import Symbol, solve, lambdify
from scipy import optimize, stats
import numpy as np
import random


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



def complementary_seqs(input_seq_iter):
    return tuple([complementary_seq(seq) for seq in input_seq_iter])


########################################################################
###   GLOBALS
########################################################################

ECHO_DIRECTION = ["_tail", "_head"]
CLASSIC_START_CODONS = {"ATG", "ATC", "ATA", "ATT", "GTG", "TTG"}
CLASSIC_STOP_CODONS = {"TAA", "TAG", "TGA"}

DEGENERATE_BASES = {"N", "V", "H", "D", "B", "Y", "R", "K", "M", "S", "W"}

DEGENERATE_DICT = {  # degenerate
    "N": ["A", "C", "G", "T"],
    "V": ["A", "C", "G"], "H": ["A", "C", "T"], "D": ["A", "G", "T"], "B": ["C", "G", "T"],
    "Y": ["C", "T"], "R": ["A", "G"], "K": ["G", "T"], "M": ["A", "C"],
    "S": ["C", "G"], "W": ["A", "T"],
    "A": ["A"], "C": ["C"], "G": ["G"], "T": ["T"],
    #  consensus
    ('A', 'C', 'G', 'T'): 'N',
    ('A', 'C', 'G'): 'V', ('A', 'C', 'T'): 'H', ('A', 'G', 'T'): 'D', ('C', 'G', 'T'): 'B',
    ('C', 'T'): 'Y', ('A', 'G'): 'R', ('G', 'T'): 'K', ('A', 'C'): 'M',
    ('C', 'G'): 'S', ('A', 'T'): 'W',
    ('A',): 'A', ('C',): 'C', ('G',): 'G', ('T',): 'T',
}

DEGENERATE_DICT_DIGIT = {  
    # degenerate
    "N": [1, 2, 4, 8],
    "V": [1, 2, 4], "H": [1, 2, 8], "D": [1, 4, 8], "B": [2, 4, 8],
    "Y": [2, 8], "R": [1, 4], "K": [4, 8], "M": [1, 2],
    "S": [2, 4], "W": [1, 8],
    "A": [1], "C": [2], "G": [4], "T": [8],
    #  consensus
    15: 'N',
    7: 'V', 11: 'H', 13: 'D', 14: 'B',
    10: 'Y', 5: 'R', 12: 'K', 3: 'M',
    6: 'S', 9: 'W',
    1: "A", 2: "C", 4: "G", 8: "T",
}



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


class ProcessingGraphFailed(Exception):
    def __init__(self, value=""):
        self.value = value

    def __str__(self):
        return repr(self.value)


class Vertex(object):
    def __init__(self, v_name, length=None, coverage=None, forward_seq=None, reverse_seq=None,
                 tail_connections=None, head_connections=None, fastg_form_long_name=None):
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



########################################################################
###   FUNCTIONS OUTSIDE OF CLASSES
########################################################################



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



def generate_consensus(*seq_str):
    consensus_res = []
    seq_num = len(seq_str)
    for go_to_base in range(len(seq_str[0])):
        this_base_set = set()
        for go_to_seq in range(seq_num):
            this_base_set.update(DEGENERATE_DICT_DIGIT.get(seq_str[go_to_seq][go_to_base], []))
        consensus_res.append(DEGENERATE_DICT_DIGIT[sum(this_base_set)])
    return "".join(consensus_res)



def generate_index_combinations(index_list):
    if not index_list:
        yield []
    else:
        for go_id in range(index_list[0]):
            for next_ids in generate_index_combinations(index_list[1:]):
                yield [go_id] + next_ids



def smart_trans_for_sort(candidate_item):
    if type(candidate_item) in (tuple, list):
        return [smart_trans_for_sort(this_sub) for this_sub in candidate_item]
    elif type(candidate_item) == bool:
        return not candidate_item
    else:
        all_e = candidate_item.split("_")
        for go_e, this_ele in enumerate(all_e):
            try:
                all_e[go_e] = int(this_ele)
            except ValueError:
                try:
                    all_e[go_e] = float(this_ele)
                except ValueError:
                    pass
        return all_e
