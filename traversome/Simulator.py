#!/usr/bin/env python
import sys

import numpy as np
from typing import Union, List, Tuple
from numpy.typing import ArrayLike
from scipy.stats import gamma
from traversome.Assembly import Assembly
from traversome.utils import find_id_using_binary_search, Sequence
from loguru import logger
import gzip
# from pympler.asizeof import asizeof


class SimpleSimulator(object):
    """
    """
    def __init__(
            self,
            graph_obj: Assembly,
            variants: Union[List[tuple], Tuple[tuple]],
            variant_proportions: ArrayLike,  # ArrayLike[float],
            length_distribution: Union[str, Tuple[float]],
            data_size: int,
            out_gaf: Union[str, None] = None,
            out_fasta: Union[str, None] = None,
            random_seed: int = 12345):
        """
        :param data_size: total number of bases to simulate
        :param length_distribution: ont/pb/hifi/(mean, std_dev)
            Gamma distribution following https://github.com/rrwick/Badread#fragment-lengths.
        """
        self.graph_obj = graph_obj
        self.out_gaf = out_gaf
        self.out_fasta = out_fasta
        assert len(variants) == len(variant_proportions)
        for variant in variants:
            assert graph_obj.contain_path(variant), "{} not found in the graph!".format(variant)
        self.variants = variants
        self.variant_seqs = []
        self._is_variant_circ = []
        self.variant_lengths = []
        self.variant_sizes = []
        self._gen_variant_stat()
        # normalize the proportions to sum to 1
        self.v_prop = np.array(variant_proportions) / sum(variant_proportions)
        self.random_seed = random_seed
        self.data_size = data_size
        # use mean and standard deviation to calculate the shape and scale for gamma distribution
        if length_distribution == "ont":  # approximately 10~100 kb
            self.mean, self.std_dev = 15000, 13000
        elif length_distribution == "pb":  # ? approximately 10~20 kb
            self.mean, self.std_dev = 13000, 10000
        elif length_distribution == "hifi":  # ? approximately 12~24 kb
            self.mean, self.std_dev = 15000, 10000
        elif isinstance(length_distribution, (tuple, list)) and len(length_distribution) == 2:
            self.mean, self.std_dev = length_distribution
        else:
            raise ValueError("Invalid input for length_distribution!")
        shape = (self.mean/self.std_dev) ** 2
        scale = self.std_dev ** 2 / self.mean
        self.len_distribution = gamma(shape, scale=scale)
        np.random.seed(self.random_seed)

        # to be generated
        self._r_lengths = []
        self._variant_path_starts_table = []
        # self._variant_path_starts_table_m = []
        self._variant_path_ends_table = []
        self._variant_path_ends_table_m = []
        self._variant_template = []
        self._variant_seq_template = []
        self._random_01 = []
        self._cached_path_len = {}
        # self.gaf_records = []
        # # for faster generating end-to-start paths
        # self._cached_concat_units = {}  # {var_id: {num_units(INT): path(TUPLE), ...}}
        # self._cached_concat_u_acc_lengths = {}   # {var_id: {num_units[INT]: lengths(LIST[INT]), ...}}

    def run(self):
        """

        """
        # TODO: Improvement or Notice needed.
        #       Here we assume that the read length is not limited by the length and topology of any template:
        #       either cyclic topology template can be infinitely concatenating units,
        #       or acyclic template will have longer length than the reads.
        #       If above assumptions were violated, the original gamma distribution will be distorted;
        #       lengths longer than the variant will be directly trimmed.
        self._r_lengths = self._sim_read_lengths()
        # logger.info(f"self._r_lengths: {asizeof(self._r_lengths)}")
        self._gen_variant_path_starts_lookup_table()
        # logger.info(f"self._variant_path_starts_table: {asizeof(self._variant_path_starts_table)}")
        self._gen_variant_path_ends_lookup_table()
        # logger.info(f"self._variant_path_ends_table: {asizeof(self._variant_path_ends_table)}")
        num_reads = len(self._r_lengths)
        logger.info(f"num_reads = {num_reads}")

        # randomly assign reads to the variants according to variant_proportions (self.v_prop)
        variant_ids = np.random.choice(range(len(self.v_prop)), size=num_reads, p=self.v_prop)
        # logger.info(f"variant_ids: {asizeof(variant_ids)}")
        # random
        self._random_01 = list(np.random.randint(2, size=num_reads))
        # logger.info(f"self._random_01: {asizeof(self._random_01)}")
        # prepare sequence
        if self.out_fasta:
            self.variant_seqs = [self.graph_obj.export_path_seq_str(v_, False) for v_ in self.variants]
        else:
            self.variant_seqs = ["" for v_ in self.variants]
        logger.info("initialized")

        # generate the start points of the read, start/end lookup table, and prepare the templates
        align_start_points = []
        # self._variant_path_starts_table_m = []
        self._variant_path_ends_table_m = []
        for var_id, variant in enumerate(self.variants):
            is_circular = self._is_variant_circ[var_id]
            var_len = self.variant_lengths[var_id]
            here_r_lengths = self._r_lengths[variant_ids == var_id]
            here_size = len(here_r_lengths)
            # find the start point and update the read_lengths if applied
            if is_circular:
                lens_start_choices = np.full(here_size, var_len)
            else:
                # lengths longer than the variant will be directly trimmed
                self._r_lengths[variant_ids == var_id] = np.where(here_r_lengths > var_len, var_len, here_r_lengths)
                here_r_lengths = self._r_lengths[variant_ids == var_id]
                # number of start choices is L-K+1 for linear variant
                lens_start_choices = var_len - here_r_lengths + 1
                # lens_start_choices[lens_start_choices < 1] = 1  # reads longer than the template were trimmed
            here_start_points = (np.random.random(size=here_size) * lens_start_choices - 1e-15).astype(int)
            align_start_points.append(list(here_start_points))
            # multiply the units of start/end points and template according to the maximum read length
            if is_circular:
                max_n_units = max(here_r_lengths) // var_len + 2
                # multiply start and end lookup table
                # self._variant_path_starts_table_m.append([x_ + nu * var_len
                #                                           for nu in range(max_n_units)
                #                                           for x_ in self._variant_path_starts_table[var_id]])
                self._variant_path_ends_table_m.append([x_ + nu * var_len
                                                        for nu in range(max_n_units)
                                                        for x_ in self._variant_path_ends_table[var_id]])
                # generate the template by create max_n_units times of the original variant
                self._variant_template.append(variant * max_n_units)
                self._variant_seq_template.append(self.variant_seqs[var_id] * max_n_units)
                logger.info(f"max_n_units= {max_n_units}")
            else:
                # keep the variant unchanged
                self._variant_template.append(variant)
                self._variant_seq_template.append(self.variant_seqs[var_id])
                # keep start and end lookup table unchanged
                # self._variant_path_starts_table_m.append(self._variant_path_starts_table[var_id])
                self._variant_path_ends_table_m.append(self._variant_path_ends_table[var_id])
            # logger.info(f"self._variant_template: {asizeof(self._variant_template)}")
            # logger.info(f"self._variant_path_starts_table_m: {asizeof(self._variant_path_starts_table_m)}")
            # logger.info(f"self._variant_path_ends_table_m: {asizeof(self._variant_path_ends_table_m)}")
            logger.info(f"prepared start/end for var_id={var_id}")

        # start simulating GAF-format alignments
        # https://github.com/lh3/gfatools/blob/master/doc/rGFA.md#the-graph-alignment-format-gaf
        # TODO: (?) can be merged into GraphAlignRecords.py, which is currently a parser
        # self.gaf_records = []
        logger.info("Simulating reads ..")
        n_digits = len(str(num_reads))
        n_echo_bins = 10
        step = num_reads // n_echo_bins
        output_ali_h = None
        if self.out_gaf:
            if self.out_gaf.endswith("gz"):
                output_ali_h = gzip.open(self.out_gaf, "wt")
            else:
                output_ali_h = open(self.out_gaf, "w")
        output_fas_h = None
        if self.out_fasta:
            if self.out_fasta.endswith("gz"):
                output_fas_h = gzip.open(self.out_fasta, "wt")
            else:
                output_fas_h = open(self.out_fasta, "w")
        for go_r, (r_len, var_id) in enumerate(zip(self._r_lengths, variant_ids)):
            ali_start_p = align_start_points[var_id].pop(0)
            query_seq_name = f"r{go_r + 1:0{n_digits}d}"
            if output_fas_h:
                # TODO reverse strand, not necessary for now
                seq = Sequence(label=query_seq_name,
                               seq=self._variant_seq_template[var_id][ali_start_p: ali_start_p + r_len])
                output_fas_h.write(f"{seq.fasta_str(interleaved=True)}\n")
            if output_ali_h:
                query_seq_len = r_len
                query_start = 0  # 0-based; closed
                query_end = r_len  # 0-based; open
                # TODO reverse strand, not necessary for now
                strand_relative_to_path = "+"  # strand can be reverse in the pair-end case
                path_matching, start_on_path, end_on_path = \
                    self.sim_path(var_id, align_start=ali_start_p, align_len=query_seq_len)
                path_str = "".join([(">" if ve else "<") + vn for vn, ve in path_matching])
                if path_matching in self._cached_path_len:
                    path_len = self._cached_path_len[path_matching]
                else:
                    path_len = self._cached_path_len[path_matching] = \
                        self.graph_obj.get_path_length(path_matching, False, False)
                num_match = query_seq_len  # perfect matching
                align_block_len = query_seq_len  # perfect matching
                # quality = 255  # missing
                quality = 60  # great
                optional_id_f = "id:f:1.0"
                record = [query_seq_name, query_seq_len, query_start, query_end, strand_relative_to_path,
                          path_str, path_len, start_on_path, end_on_path, num_match, align_block_len, quality,
                          optional_id_f]
                # self.gaf_records.append(record)
                output_ali_h.write("\t".join([str(x) for x in record]) + "\n")
            if (go_r + 1) % step == 0:
                percentage = float(go_r + 1) / num_reads * 100.
                progress = (go_r + 1) // step
                logger.info(f"[{'#' * progress}{' ' * (n_echo_bins - progress)}] {percentage:.1f}%")
        if self.out_gaf:
            output_ali_h.close()
        if self.out_fasta:
            output_fas_h.close()
        logger.info("Simulating reads finished.")
        # with open(self.out_gaf, "w") as output_h:
        #     for record in self.gaf_records:
        #         output_h.write("\t".join([str(x) for x in record]) + "\n")

    def _gen_variant_stat(self):
        for variant in self.variants:
            # TODO: whether circular should be defined as an attribute of the path in further implementation (?)
            is_circular = self.graph_obj.is_circular_path(variant)
            self._is_variant_circ.append(is_circular)
            self.variant_lengths.append(self.graph_obj.get_path_length(variant, adjust_for_cyclic=is_circular))
            self.variant_sizes.append(len(variant))

    def _sim_read_lengths(
            self,
            min_len: int = 100  # TODO may slightly bias the total distribution
    ):
        lengths = []
        sum_base = 0
        while sum_base < self.data_size:
            # triple the desired sample size to ensure enough reads at once
            sim_num_reads = int(2 * ((self.data_size - sum_base) / float(self.mean) + 1))
            len_pool = self.len_distribution.rvs(size=sim_num_reads)
            for r_len in len_pool:
                if r_len < min_len:
                    continue
                lengths.append(int(r_len))
                sum_base += r_len
                if sum_base >= self.data_size:
                    break
        return np.array(lengths)

    def _gen_variant_path_starts_lookup_table(self):
        """
        Each start point correspond to a new vertex, as the start of a read path.
        """
        self._variant_path_starts_table = [[] for foo in range(len(self.variants))]
        for var_id, variant in enumerate(self.variants):
            pointer = 0  # closed
            self._variant_path_starts_table[var_id].append(pointer)
            for (previous_n, previous_e), (next_n, next_e) in zip(variant[:-1], variant[1:]):
                previous_vertex = self.graph_obj.vertex_info[previous_n]
                overlap = previous_vertex.connections[previous_e][(next_n, not next_e)]
                previous_len = previous_vertex.len - overlap
                pointer += previous_len
                self._variant_path_starts_table[var_id].append(pointer)

    def _gen_variant_path_ends_lookup_table(self):
        self._variant_path_ends_table = [[] for foo in range(len(self.variants))]
        for var_id, variant in enumerate(self.variants):
            # logger.trace(f"len(variant)={len(variant)}")
            first_n, first_e = variant[0]
            # last_n, last_e = variant[-1]
            # pointer = self.graph_obj.vertex_info[last_n].connections[last_e][(first_n, not first_e)]  # open
            # self._variant_path_ends_table[var_id].append(pointer)  # point to the last v
            pointer = self.graph_obj.vertex_info[first_n].len  # open
            self._variant_path_ends_table[var_id].append(pointer)  # point to the first
            for go, ((previous_n, previous_e), (next_n, next_e)) in enumerate(zip(variant[:-1], variant[1:])):
                next_v = self.graph_obj.vertex_info[next_n]
                overlap = next_v.connections[not next_e][(previous_n, previous_e)]
                add_len = next_v.len - overlap
                # print(var_id, go, previous_n, add_len)
                pointer += add_len
                self._variant_path_ends_table[var_id].append(pointer)
            # print(var_id, len(self._variant_path_ends_table[var_id]))
            # logger.info(f"len_ends_table{var_id}={len(self._variant_path_ends_table[var_id])}")

    def sim_path(self, var_id, align_start, align_len):
        start_v_id, s_gap = find_id_using_binary_search(
            s_points=self._variant_path_starts_table[var_id],
            seek_value=align_start,
            ceiling=False,
            return_gap=True)
        # align_start + align_len are open index of the base, matches path_ends_table, which are also open index
        # As a result, end_v_id is a close index for the vertex id in the path,
        # meaning that we have to use it in the form of end_v_id + 1 for slicing purpose
        end_v_id, e_gap = find_id_using_binary_search(
            s_points=self._variant_path_ends_table_m[var_id],
            seek_value=align_start + align_len,
            ceiling=True,
            return_gap=True)
        if start_v_id > end_v_id:
            # a path is in the overlap between two or even more vertices, randomly choose one of the vertices
            # For example following sequential contigs with overlaps
            # contig_1 |----------------------|
            # contig_2               |-----------|
            # contig_3                  |-------------|
            # read_1                      <-->
            #    will result start_v_id = end_v_id + 2
            # raise ValueError(f"start_v_id ({start_v_id}) > end_v_id + 1 ({end_v_id + 1})!\n")
            middle_id = int((start_v_id + end_v_id)/2.)
            pick_ = [middle_id, middle_id + 1][self._random_01.pop()]
            new_path = self._variant_template[var_id][pick_: pick_ + 1]
        else:
            new_path = self._variant_template[var_id][start_v_id: end_v_id + 1]
        start_pos_on_path = s_gap  # 0-based according to GAF definition
        # end_pos_on_path = s_gap + align_len - 1  # 0-based according to GAF definition
        # open end according to the example
        end_pos_on_path = s_gap + align_len  # 0-based according to GAF definition
        # for debug: can be removed latter
        # assert s_gap + align_len + e_gap == self.graph_obj.get_path_length(new_path, adjust_for_cyclic=False), \
        #     f"checking failed: {s_gap} + {align_len} + {e_gap} != " \
        #     f"{self.graph_obj.get_path_length(new_path, adjust_for_cyclic=False)} " \
        #     f"~ size ({len(new_path)}) ~ len{new_path}!\n\n" \
        #     f"Given that start_v_id={start_v_id}, end_v_id={end_v_id}, len_unit={self.variant_sizes[var_id]}, " \
        #     f"len_template={len(self._variant_template[var_id])}, \nvariant={self.variants[var_id]}, \n\n" \
        #     f"end_table({len(self._variant_path_ends_table[var_id])})={self._variant_path_ends_table[var_id]}\n" \
        #     f"end_table_m({len(self._variant_path_ends_table_m[var_id])})={self._variant_path_ends_table_m[var_id]}"
        return new_path, start_pos_on_path, end_pos_on_path
