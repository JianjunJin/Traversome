#!/usr/bin/env python
"""
Class objects for detecting conflicts between assembly graph and graph alignment
"""
import math
import numpy as np
from loguru import logger
from collections import OrderedDict
import numpy as np
from traversome.utils import GaussianMixtureModel


# TODO: need to weight the probability of the bin if the bin has size smaller than the window size
class GraphAlignConflicts(object):
    def __init__(
            self, 
            graph, 
            graph_alignment, 
            output_dir,
            window_size=50, 
            window_step=40) -> None:
        self.graph = graph
        self.alignment = graph_alignment
        self.output_dir = output_dir
        self.window_size = window_size
        self.window_step = window_step
        self.n_simulations = 100000
        self.alpha = 0.001
        # to be generated
        self.v_window_conflicts_info = {}
        self.v_window_conflicts_counts = {}
        self.n_bins = None
        self.n_balls = None
        self.max_load = None
        self.conflict_n = []
        self.max_loads = []

    def detect(self):
        """
        Executes the algorithm to find conflicts in the graph and determine the maximum load.

        Returns:
            conflict_n (list): A list of vertex numbers with conflicts.
            max_loads (list): A list of maximum loads for each vertex with conflicts.
        """
        self._find_vertex_window_wise_conflicts()
        all_conflicts = []
        for conflicts in self.v_window_conflicts_counts.values():
            all_conflicts.extend(conflicts)
        self.n_bins = len(all_conflicts)
        self.n_balls = sum(all_conflicts)
        logger.debug(f"Total number of bins: {self.n_bins}")
        logger.debug(f"Total number of balls: {self.n_balls}")
        self.conflict_n = []
        self.max_loads = []
        if self.n_balls == 0:
            pass
        else:
            self.max_load = self._find_possible_max_load(N=self.n_bins, k=self.n_balls)
            logger.debug(f"Possible max load: {self.max_load}")
            self.conflict_n = []
            self.max_loads = []
            for v_n, conflicts in self.v_window_conflicts_counts.items():
                here_max_load = max(conflicts)
                if here_max_load >= self.max_load:
                    self.conflict_n.append(v_n)
                    self.max_loads.append(here_max_load)
        return list(self.conflict_n), list(self.max_loads)
    
    def modify_graph(
            self, 
            min_n_reads: int, 
            # max_conflict_distance: int = 20
            gmm_max_std: float = 50.0
            ):
        """
        Add edges to the graph to resolve conflicts.

        :param min_n_reads: Minimum number of reads to consider a vaid break point.
        # :param max_conflict_distance: Maximum distance for two reads to be considered supporting the same conflict.
        """
        ########
        # TODO add an extra contig for the added edge when there are unaligned regions between segments
        ########
        # order the conflicts by read segments
        # the positions (from_pos, to_pos) are 1-based and on the forward direction.
        min_n_reads = max(min_n_reads, self.max_load)
        read_seg_clusters = {}
        for conflict_n in self.conflict_n:
            for bin_n, conflict_bin_info in self.v_window_conflicts_info[conflict_n].items():
                if self.v_window_conflicts_counts[conflict_n][bin_n] >= min_n_reads:
                    # TODO no need to record self.v_window_conflicts_info in the way of binning, directly record the conflict info
                    for is_to_conflict, read_n, go_r, go_r_next, conflict_e, conflict_site in conflict_bin_info:
                        read_seg = (read_n, go_r, go_r_next)
                        if read_seg not in read_seg_clusters:
                            read_seg_clusters[read_seg] = [None, None]
                        read_seg_clusters[read_seg][int(is_to_conflict)] = (conflict_n, conflict_e, conflict_site)
        ########
        # record post-filtered bk points for gmm clustering
        post_filtered_bk_points = {}  # {v_name: [pos1, pos2, ...]} # 0-based, directly used to slice the contig
        conflict_clusters = {}
        # count_abnormal = 0
        for read_seg, conflict_info in read_seg_clusters.items():
            # the other side of the conflict is not within self.conflict_n or filtered out by min_n_reads
            if None in conflict_info:
                # count_abnormal += 1
                # logger.warning(f"Abnormal conflict info: {read_seg} {conflict_info}")
                continue
            (from_v, from_e, from_pos), (to_v, to_e, to_pos) = conflict_info
            # if from_v == "379111":
            #     logger.warning(str(conflict_info))
            #########
            # sort the direction to standardize the conflict record
            # if from_v > to_v or (from_v == to_v and from_e > to_e) or (from_v == to_v and from_e == to_e and from_pos > to_pos):
            if from_v > to_v or (from_v == to_v and from_pos > to_pos):
                # reverse the direction of the conflict record
                # reversing the direction of the contig as well
                #  the pos is 1-based and relative to the forward direction of the contig, so the pos remain unchanged
                from_v, from_e, from_pos, to_v, to_e, to_pos = to_v, not to_e, to_pos, from_v, not from_e, from_pos
            candidate_edge = (from_v, from_e, to_v, to_e)
            # if from_v == to_v == "379183":
            #     logger.warning(f"Conflict info: {read_seg} {conflict_info}")
            #     logger.warning(f"adjusted conflict edge: {candidate_edge}")

            if candidate_edge not in conflict_clusters:
                conflict_clusters[candidate_edge] = [[from_pos, to_pos, read_seg]]
            else:
                conflict_clusters[candidate_edge].append([from_pos, to_pos, read_seg])
            # record post-filtered bk points for gmm clustering
            if from_v not in post_filtered_bk_points:
                post_filtered_bk_points[from_v] = []
            if from_e:
                post_filtered_bk_points[from_v].append(from_pos)
            else:
                post_filtered_bk_points[from_v].append(from_pos - 1)
            if to_v not in post_filtered_bk_points:
                post_filtered_bk_points[to_v] = []
            if to_e:
                post_filtered_bk_points[to_v].append(to_pos - 1)
            else:
                post_filtered_bk_points[to_v].append(to_pos)
        # logger.warning(f"Number of abnormal conflict ratio: {count_abnormal}:{len(read_seg_clusters)}")
        ########
        # automatically cluster the conflicts positions using gmm-em
        # TODO: consider joining breaks in neighboring contigs, which may be complex mathematically and skip for now
        # for now just add arbitrary distance (e.g. max_pos+1000) between positions in different contigs, e.g
        # TODO: skip the gmm clustering if the number is less than 2
        sorted_vertices = sorted(post_filtered_bk_points.keys())
        # max_pos is used to separate the positions in different contigs in a uniform space, so that converting back and forth is easy
        max_pos = max([max(post_filtered_bk_points[v_name]) for v_name in sorted_vertices]) + 1000
        flattened_bk_points = []
        for go_v, v_name in enumerate(sorted_vertices):
            offset = go_v * max_pos
            flattened_bk_points.extend([x + offset for x in post_filtered_bk_points[v_name]])
        gmm = GaussianMixtureModel(max_std_dev=gmm_max_std)
        gmm.fit(list(flattened_bk_points))
        means = np.array(gmm.means)
        gmm_clusters = gmm.predict(list(flattened_bk_points))
        # skip a cluster if the size of the cluster is less than min_n_reads
        cluster_sizes = np.bincount(gmm_clusters)
        for go_c, c_size in enumerate(cluster_sizes):
            if c_size < min_n_reads:
                means[go_c] = None

        # for debugging
        # logger.debug(f"means: {means}")
        # for go_p, p in enumerate(flattened_bk_points):
        #     logger.debug(f"{p} -> {gmm_clusters[go_p]}")

        # generate the mapper from cluster to (v, mean_pos)

        cluster_to_v_means = {}
        v_means = {}  # generate the valid mean positions for all vertices
        for go_m, mean_val in enumerate(means):
            # if mean_val is not None:
            # None in an np array is not None, so use np.isnan
            if not np.isnan(mean_val):
                v_n = sorted_vertices[int(mean_val // max_pos)]
                v_mean_pos = int(mean_val % max_pos)
                if v_n not in v_means:
                    v_means[v_n] = []
                v_means[v_n].append(v_mean_pos)
                cluster_to_v_means[go_m] = (v_n, v_mean_pos)
        # logger.debug(f"cluster_to_v_means: {cluster_to_v_means}")
            
        # generate mapper for original_pos to mean_pos for all break points in each vertex
        v_pos_to_mean_pos = {}
        go_point = 0
        for go_v, v_name in enumerate(sorted_vertices):
            if v_name not in v_pos_to_mean_pos:
                v_pos_to_mean_pos[v_name] = {}
            for pos in post_filtered_bk_points[v_name]:
                cluster_id = gmm_clusters[go_point]
                if cluster_id in cluster_to_v_means:
                    c_v_name, c_mean_pos = cluster_to_v_means[cluster_id]
                    v_pos_to_mean_pos[v_name][pos] = c_mean_pos
                    assert c_v_name == v_name  # theoretically can be skipped, TODO test
                go_point += 1
        # logger.debug(f"v_pos_to_mean_pos: {v_pos_to_mean_pos}")

        # create edge clusters based on the break point clusters
        averaged_edge_clusters = {}
        for candidate_edge, from_to_seg_list in conflict_clusters.items():
            from_v, from_e, to_v, to_e = candidate_edge
            if from_v not in v_pos_to_mean_pos or to_v not in v_pos_to_mean_pos:
                continue
            for from_pos, to_pos, read_seg in from_to_seg_list:
                if from_e:
                    if from_pos not in v_pos_to_mean_pos[from_v]:
                        # logger.warning(f"from_pos {from_pos} not in v_pos_to_mean_pos[{from_v}]")
                        continue
                    else:
                        from_mean_pos = v_pos_to_mean_pos[from_v][from_pos]
                else:
                    if from_pos - 1 not in v_pos_to_mean_pos[from_v]:
                        # logger.warning(f"from_pos {from_pos - 1} not in v_pos_to_mean_pos[{from_v}]")
                        continue
                    else:
                        from_mean_pos = v_pos_to_mean_pos[from_v][from_pos - 1] + 1 # v_pos_to_mean_pos is 0-based, so need to adjust
                if to_e:
                    if to_pos - 1 not in v_pos_to_mean_pos[to_v]:
                        # logger.warning(f"to_pos {to_pos - 1} not in v_pos_to_mean_pos[{to_v}]")
                        continue
                    else:
                        to_mean_pos = v_pos_to_mean_pos[to_v][to_pos - 1] + 1
                else:
                    if to_pos not in v_pos_to_mean_pos[to_v]:
                        # logger.warning(f"to_pos {to_pos} not in v_pos_to_mean_pos[{to_v}]")
                        continue
                    else:
                        to_mean_pos = v_pos_to_mean_pos[to_v][to_pos]                     
                if candidate_edge not in averaged_edge_clusters:
                    averaged_edge_clusters[candidate_edge] = OrderedDict()
                if (from_mean_pos, to_mean_pos) not in averaged_edge_clusters[candidate_edge]:
                    averaged_edge_clusters[candidate_edge][(from_mean_pos, to_mean_pos)] = []
                averaged_edge_clusters[candidate_edge][(from_mean_pos, to_mean_pos)].append(read_seg)
            # additional filtering for num of reads supporting the conflict can be added here
            # if candidate_edge in averaged_edge_clusters:
            #     for mean_pos_pair, read_segs in list(averaged_edge_clusters[candidate_edge].items()):
            #         if len(read_segs) < min_n_reads:
            #             del averaged_edge_clusters[candidate_edge][mean_pos_pair]
            #     if not averaged_edge_clusters[candidate_edge]:
            #         del averaged_edge_clusters[candidate_edge]
        # TODO
        # print number of edges and number of conflicts being added
        for candidate_edge, conflict_info in averaged_edge_clusters.items():
            from_v, from_e, to_v, to_e = candidate_edge
            for (from_mean_pos, to_mean_pos), read_segs in conflict_info.items():
                logger.debug(f"Adding edge: {from_v}({from_mean_pos}){'+' if from_e else '-'} -> {to_v}({to_mean_pos}){'+' if to_e else '-'}: ")

        # write new edge information to a tab file
        if averaged_edge_clusters:
            join_vs = []
            with open(f"{self.output_dir}/conflict_edges.tab", "w") as f_tab:
                f_tab.write("from_v\tfrom_e\tfrom_pos\tto_v\tto_e\tto_pos\tread_name[from_seg,to_seg]\n")
                for candidate_edge, conflict_info in averaged_edge_clusters.items():
                    from_v, from_e, to_v, to_e = candidate_edge
                    for (mean_from_pos, mean_to_pos), read_segs in conflict_info.items():
                        read_info = ";".join([f"{x[0]}[{x[1]},{x[2]}]" for x in read_segs])
                        f_tab.write(f"{from_v}\t{from_e}\t{mean_from_pos}\t{to_v}\t{to_e}\t{mean_to_pos}\t{read_info}\n")
                        join_vs.append((from_v, from_e, mean_from_pos, to_v, to_e, mean_to_pos))
            # write new edge information to a graph file
            logger.info(f"Adding {len(join_vs)} edges to the graph")
            new_graph = self.graph.add_edges_inside_contigs(join_vs)
            return new_graph
        else:
            return None
        
        ##########
        # old version of clustering
        ##########
        # # automatically cluster the conflicts within each candidate edge by from_pos and to_pos, 
        # # within each cluster, both from_pos and to_pos should be close to each other
        # # if either of the from_pos and to_pos are far away from the cluster, split the cluster
        # conflict_clusters_filtered = []
        # for candidate_edge, conflict_info in conflict_clusters.items():
        #     if len(conflict_info) < min_n_reads:
        #         continue
        #     cluster_ids = self._cluster_2d_data([cc[:2] for cc in conflict_info], max_conflict_distance, max_conflict_distance)[0]
        #     for go_c in sorted(set(cluster_ids)):
        #         current_cluster = []
        #         for go_cc, cc in enumerate(conflict_info):
        #             if cluster_ids[go_cc] == go_c:
        #                 current_cluster.append(cc)
        #         if len(current_cluster) >= min_n_reads:
        #             conflict_clusters_filtered.append((candidate_edge, current_cluster))
        # ########
        # # for candidate_edge, conflict_info in conflict_clusters_filtered:
        # #     logger.error(f"{str(candidate_edge)}: {len(conflict_info)}: "
        # #                  f"{min([x[0] for x in conflict_info])}-{max([x[0] for x in conflict_info])}, "
        # #                  f"{min([x[1] for x in conflict_info])}-{max([x[1] for x in conflict_info])}")
        # # write new edge information to a tab file
        # if conflict_clusters_filtered:
        #     join_vs = []  # (from_v, from_e, from_pos, to_v, to_e, to_pos)
        #     with open(f"{self.output_dir}/conflict_edges.tab", "w") as f_tab:
        #         f_tab.write("from_v\tfrom_e\tfrom_pos\tto_v\tto_e\tto_pos\tread_name[from_seg,to_seg]\n")
        #         for candidate_edge, conflict_info in conflict_clusters_filtered:
        #             from_v, from_e, to_v, to_e = candidate_edge
        #             # mean_from_pos, mean_to_pos = np.mean([cc[:2] for cc in conflict_info], axis=0)
        #             # mean_from_pos, mean_to_pos = round(mean_from_pos), round(mean_to_pos)
        #             mean_from_pos, mean_to_pos = np.median([cc[:2] for cc in conflict_info], axis=0)
        #             mean_from_pos, mean_to_pos = round(mean_from_pos), round(mean_to_pos)
        #             read_info = ";".join([f"{cc[2][0]}[{cc[2][1]},{cc[2][2]}]" for cc in conflict_info])
        #             f_tab.write(f"{from_v}\t{from_e}\t{mean_from_pos}\t{to_v}\t{to_e}\t{mean_to_pos}\t{read_info}\n")
        #             join_vs.append((from_v, from_e, mean_from_pos, to_v, to_e, mean_to_pos))
        #     # write new edge information to a graph file
        #     new_graph = self.graph.add_edges_inside_contigs(join_vs)
        #     return new_graph
        # else:
        #     return None

    def _find_vertex_window_wise_conflicts(self):
        # can be improved by optionally store self.v_window_conflicts_info only when needed
        self.v_window_conflicts_info = {}
        self.v_window_conflicts_counts = {}
        v_lengths = {}
        for v_n, v_info in self.graph.vertex_info.items():
            v_lengths[v_n] = v_info.len
            n_bins = self.count_bins(v_info.len)
            self.v_window_conflicts_counts[v_n] = [0 for foo in range(n_bins)]
            self.v_window_conflicts_info[v_n] = {}

        total_reads = len(self.alignment.read_records)
        total_records = sum([len(r_records) for r_records in self.alignment.read_records.values()])
        logger.debug(f"Total number of reads: {total_reads}")
        logger.debug(f"Total number of records: {total_records}")
        if total_reads != total_records:
            for read_n, r_records in self.alignment.read_records.items():
                if len(r_records) > 1:
                    r_records.sort_by()
                    for go_r, rec in enumerate(r_records):
                        # if read_n == 'SRR11434954.25514 25514 length=13075':
                        #     logger.warning(f"SRR11434954.25514 25514 length=13075 -- Record {go_r}: {rec.path}, {rec.p_start}, {rec.p_end}, {rec.p_len}")
                        if go_r != 0:  # if is not start part of the query, then the start of the path means a conflict
                            conflict_n, conflict_e = rec.path[0]
                            conflict_site = rec.p_start  # zero based
                            max_len = v_lengths[conflict_n]
                            if conflict_e:
                                conflict_site += 1
                            else:
                                conflict_site = max_len - conflict_site
                            bins = self.find_bin_numbers(base=conflict_site, max_base=max_len)
                            for b_n in bins:
                                self.v_window_conflicts_counts[conflict_n][b_n] += 1
                                # is_to_conflict, read_name, from_record, to_record, conflict_e, conflict_site
                                if b_n not in self.v_window_conflicts_info[conflict_n]:
                                    self.v_window_conflicts_info[conflict_n][b_n] = []
                                self.v_window_conflicts_info[conflict_n][b_n].append(
                                    (True, read_n, go_r-1, go_r, conflict_e, conflict_site))
                                # self.v_window_conflicts_info[conflict_n].setdefault(b_n, []).append(
                                #     (True, read_n, go_r-1, go_r, conflict_e, conflict_site))
                                # if read_n == 'SRR11434954.25514 25514 length=13075':
                                #     logger.warning(f"SRR11434954.25514 25514 length=13075 -- to conflict added")
                        if go_r != len(r_records) - 1:  # is not end part of the query, the end of the record means a conflict
                            conflict_n, conflict_e = rec.path[-1]
                            conflict_site = rec.p_len - rec.p_end  # zero based in the reverse direction
                            max_len = v_lengths[conflict_n]
                            if conflict_e:
                                conflict_site = max_len - conflict_site
                            else:
                                conflict_site += 1
                            try:
                                bins = self.find_bin_numbers(base=conflict_site, max_base=max_len)
                            except AssertionError as e:
                                logger.error(f"Error in find_bin_numbers: {e}")
                                logger.error(f"conflict_site: {conflict_site}, max_len: {max_len}")
                                logger.error(f"conflict_n: {conflict_n}, conflict_e: {conflict_e}")
                                logger.error(f"rec: {rec.query_name}, {rec.query_len}, {rec.path}, {rec.p_start}, {rec.p_end}, {rec.p_len}")
                                raise e
                            for b_n in bins:
                                self.v_window_conflicts_counts[conflict_n][b_n] += 1
                                # is_to_conflict (False means from), read_name, from_record, to_record, conflict_e, conflict_site
                                if b_n not in self.v_window_conflicts_info[conflict_n]:
                                    self.v_window_conflicts_info[conflict_n][b_n] = []
                                self.v_window_conflicts_info[conflict_n][b_n].append(
                                    (False, read_n, go_r, go_r+1, conflict_e, conflict_site))
                                # if read_n == 'SRR11434954.25514 25514 length=13075':
                                #     logger.warning(f"SRR11434954.25514 25514 length=13075 -- from conflict added")
                                # self.v_window_conflicts_info[conflict_n].setdefault(b_n, []).append(
                                #     (False, read_n, go_r, go_r+1, conflict_e, conflict_site))
    
    def _find_possible_max_load(self, N, k):
        # TODO: need to weight the probability of the bin if the bin (contig) has size smaller than the window size
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
        return max(math.ceil((max_base - self.window_size) / self.window_step), 0) + 1
    
    def find_bin_numbers(self, base, max_base):
        assert max_base >= base >= 1, "base should be in the range [1, {max_base}]".format(max_base=max_base)
        adjusted_base = base - 1
        # the first bin is [1, window_size]
        if adjusted_base < self.window_size:
            return [0]
        #
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
    
    # @staticmethod
    # def _cluster_2d_data(data, threshold1, threshold2):

    #     def euclidean_distance(p1, p2):
    #         return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    #     clusters = []
    #     cluster_ids = []
    #     for point in data:
    #         found_cluster = False
    #         for go_c, cluster in enumerate(clusters):
    #             for cluster_point in cluster:
    #                 if (euclidean_distance(point, cluster_point) <= threshold1 and
    #                     abs(point[1] - cluster_point[1]) <= threshold2):
    #                     cluster.append(point)
    #                     cluster_ids.append(go_c)
    #                     found_cluster = True
    #                     break
    #             if found_cluster:
    #                 break
    #         if not found_cluster:
    #             clusters.append([point])
    #             cluster_ids.append(len(clusters) - 1)
    #     return cluster_ids, clusters



