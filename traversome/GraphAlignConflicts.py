#!/usr/bin/env python
"""
Class objects for detecting conflicts between assembly graph and graph alignment
"""
import math
import numpy as np
from loguru import logger


class GraphAlignConflicts(object):
    def __init__(
            self, 
            graph, 
            graph_alignment, 
            window_size=150, 
            window_step=100) -> None:
        self.graph = graph
        self.alignment = graph_alignment
        self.window_size = window_size
        self.window_step = window_step
        self.n_simulations = 100000
        self.alpha = 0.001
        # to be generated
        self.n_bins = None
        self.n_balls = None
        self.max_load = None

    def run(self):
        v_window_conflicts = self._find_vertex_window_wise_conflicts()
        all_conflicts = []
        for conflicts in v_window_conflicts.values():
            all_conflicts.extend(conflicts)
        self.n_bins = len(all_conflicts)
        self.n_balls = sum(all_conflicts)
        logger.debug(f"Total number of bins: {self.n_bins}")
        logger.debug(f"Total number of balls: {self.n_balls}")
        if self.n_balls == 0:
            return [], []
        else:
            self.max_load = self._find_possible_max_load(N=self.n_bins, k=self.n_balls)
            logger.debug(f"Possible max load: {self.max_load}")
            conflict_n = []
            max_loads = []
            for v_n, conflicts in v_window_conflicts.items():
                here_max_load = max(conflicts)
                if here_max_load >= self.max_load:
                    conflict_n.append(v_n)
                    max_loads.append(here_max_load)
            return conflict_n, max_loads

    def _find_vertex_window_wise_conflicts(self):
        v_window_conflicts = {}
        v_lengths = {}
        for v_n, v_info in self.graph.vertex_info.items():
            v_lengths[v_n] = v_info.len
            v_window_conflicts[v_n] = [0 for foo in range(self.count_bins(v_info.len))]

        logger.debug(f"Total number of reads: {len(self.alignment.read_records)}")
        logger.debug(f"Total number of records: {sum([len(r_records) for r_records in self.alignment.read_records.values()])}")
        for r_records in self.alignment.read_records.values():
            if len(r_records) > 1:
                r_records.sort_by()
                for go_r, rec in enumerate(r_records):
                    if go_r != 0:  # is not start part of the query, the start of the path means conflict/chimeric
                        conflict_n, conflict_e = rec.path[0]
                        conflict_site = rec.p_start  # zero based
                        max_len = v_lengths[conflict_n]
                        if conflict_e:
                            conflict_site += 1
                        else:
                            conflict_site = max_len - conflict_site
                        bins = self.find_bin_numbers(base=conflict_site, max_base=max_len)
                        for b_n in bins:
                            v_window_conflicts[conflict_n][b_n] += 1
                    if go_r != len(r_records) - 1:  # is not end part of the query, the end of the path means conflict/chimeric
                        conflict_n, conflict_e = rec.path[-1]
                        conflict_site = rec.p_len - rec.p_end  # zero based in the reverse direction
                        max_len = v_lengths[conflict_n]
                        if conflict_e:
                            conflict_site = max_len - conflict_site
                        else:
                            conflict_site += 1
                        bins = self.find_bin_numbers(base=conflict_site, max_base=max_len)
                        for b_n in bins:
                            v_window_conflicts[conflict_n][b_n] += 1
        return v_window_conflicts
    
    def _find_possible_max_load(self, N, k):
        assert self.n_simulations * self.alpha >= 10
        max_load_counts = {}
        for _ in range(self.n_simulations):
            # simulate placing each ball into a bin
            bins = np.random.randint(0, N, size=k)
            # Count the number of balls in each bin
            bin_counts = np.bincount(bins, minlength=N)
            # Check if the maximum load is exactly r
            max_load = np.max(bin_counts)
            if max_load not in max_load_counts:
                max_load_counts[max_load] = 1
            else:
                max_load_counts[max_load] += 1
        # Estimate the probability
        most_n_cases = self.n_simulations * (1 - self.alpha)
        accumulated = 0
        logger.debug(f"Max load counts: {max_load_counts}")
        for max_load, counts in sorted(max_load_counts.items()):
            if accumulated >= most_n_cases:
                return max_load
            else:
                accumulated += counts
        return max(max_load_counts)
    
    def count_bins(self, max_base):
        return math.ceil((max_base - self.window_size) / self.window_step) + 1
    
    def find_bin_numbers(self, base, max_base):
        assert max_base >= base >= 1, "base should be in the range [1, max_base]"
        adjusted_base = base - 1
        if adjusted_base >= max_base - self.window_size:
            # the last bin is [max_base - self.window_size + 1, max_base]
            initial_bin = self.count_bins(max_base) - 1
        else:
            initial_bin = adjusted_base // self.window_step
        bin_numbers = [initial_bin]
    
        # check previous bin ids
        current_start = initial_bin * self.window_step
        while current_start > 0:
            initial_bin -= 1
            current_start -= self.window_step
            if current_start <= adjusted_base <= current_start + self.window_size - 1:
                bin_numbers.insert(0, initial_bin)
            else:
                break
        return bin_numbers
