#!/usr/bin/env python
"""
Pangenome Assembly Graph class object and associated class objects
"""
import numpy as np
from loguru import logger
from typing import Union
from copy import deepcopy
from collections import OrderedDict
from traversome.AssemblySimple import Vertex
from traversome.Assembly import Assembly
from traversome.utils import comb_indices


class LocInfo:
    def __init__(self, vt_loc, next_keep_raw_strand, prev_keep_raw_strand):
        self.vt_loc = vt_loc
        self.next_keep_raw_strand = next_keep_raw_strand
        self.prev_keep_raw_strand = prev_keep_raw_strand

    def get_tuple(self):
        return self.vt_loc, self.next_keep_raw_strand, self.prev_keep_raw_strand


class VariantIndexer:
    def __init__(self, variant_paths, variant_props, variant_circle_info):
        self._variant_paths = variant_paths
        self._variant_props = variant_props
        self._variant_circ = variant_circle_info
        self.ve_to_loc = {}
        self.cv_id_bs_to_next = {}
        self.max_copy_to_vs = {}
        self.v_to_num_shared = {}  # numbers of variants shared this v
        self.num_shared_to_vs = {}  # number of variants to vs_set
        # =============
        # variables to be updated later
        # =============
        #     invalid ones will be skipped during, self.update_v_name_copies, self.update_num_variants_shared
        #     self.iter_variants_and_locs
        self.valid_vt_ids = set()

        # run initial indexing
        self._index_variants_for_pangenome()
        self.update_cv_distribution()

    def _index_variants_for_pangenome(self):
        """
        run once
        """
        self.valid_vt_ids = {(pid, vid)
                             for pid in self._variant_props
                             for vid in range(len(self._variant_paths[pid]))}
        # v_n: vertex name
        # v_e: vertex end/direction
        # ve: v_n, v_e
        self.ve_to_loc = {}
        self.cv_id_bs_to_next = {}
        for pid in self._variant_props:
            variant_p = self._variant_paths[pid]
            v_len = len(variant_p)
            if self._variant_circ[pid]:
                for go_v, (v_n, v_e) in enumerate(variant_p):
                    # following the v_e direction
                    next_go_v = (go_v + 1) % v_len
                    # following the reverse direction
                    prev_go_v = (go_v - 1) % v_len
                    # directed_v_loc = (v_id,
                    #                whether keep the strand if going forward,
                    #                whether keep the strand if going reverse)
                    self.__update_ve_and_next(v_n, v_e, pid, go_v, next_go_v, prev_go_v)
            else:
                for go_v, (v_n, v_e) in enumerate(variant_p):
                    # following the v_e direction
                    next_go_v = go_v + 1 if go_v + 1 < v_len else None
                    # following the reverse direction
                    prev_go_v = go_v - 1 if go_v > 0 else None
                    self.__update_ve_and_next(v_n, v_e, pid, go_v, next_go_v, prev_go_v)

    def __update_ve_and_next(self, v_n, v_e, pid, go_v, next_go_v, prev_go_v):
        self.__update_ve(v_element=(v_n, v_e), pid=pid, directed_v_loc=LocInfo(go_v, True, False))
        self.__update_ve(v_element=(v_n, not v_e), pid=pid, directed_v_loc=LocInfo(go_v, False, True))
        self.cv_id_bs_to_next[(pid, go_v, v_e)] = {True: next_go_v, False: prev_go_v}  # strand
        self.cv_id_bs_to_next[(pid, go_v, not v_e)] = {False: next_go_v, True: prev_go_v}  # strand

    def __update_ve(self, v_element, pid, directed_v_loc):
        """
        called by self._index_variants_for_pangenome
        """
        if v_element not in self.ve_to_loc:
            self.ve_to_loc[v_element] = {pid: [directed_v_loc]}
        elif pid not in self.ve_to_loc[v_element]:
            self.ve_to_loc[v_element][pid] = [directed_v_loc]
        else:
            self.ve_to_loc[v_element][pid].append(directed_v_loc)

    def __filter_vt_loc_list(self, variant_pid, vt_loc_list):
        return [loc_info
                for loc_info in vt_loc_list
                if (variant_pid, loc_info.vt_loc) in self.valid_vt_ids]

    def archive_ids(self, co_linear_block):
        for cv_list in co_linear_block:
            for variant_pid, vt_loc, vt_e in cv_list:
                self.valid_vt_ids.remove((variant_pid, vt_loc))
                # print(f"len(self.valid_vt_ids) = {len(self.valid_vt_ids)} after removing {(variant_pid, vt_loc)}")

    def update_cv_distribution(self):
        """
        update self.max_copy_to_vs, self.v_to_num_shared, self.num_shared_to_vs, according to self.ve_to_loc
        """
        self.max_copy_to_vs = {}
        self.v_to_num_shared = {}
        self.num_shared_to_vs = {}
        # multiple-copied ones in any of the variant
        for (v_n, v_e), pid_to_v in self.ve_to_loc.items():
            if v_e:  # only consider one direction
                max_copy = 0
                num_shared = 0
                for pid, v_loc_list in pid_to_v.items():
                    v_loc_list = self.__filter_vt_loc_list(variant_pid=pid, vt_loc_list=v_loc_list)
                    if v_loc_list:
                        max_copy = max(max_copy, len(v_loc_list))
                        num_shared += 1
                if max_copy:
                    if max_copy not in self.max_copy_to_vs:
                        self.max_copy_to_vs[max_copy] = set()
                    self.max_copy_to_vs[max_copy].add(v_n)
                    self.v_to_num_shared[v_n] = num_shared
        #
        for v_n, num_vars in self.v_to_num_shared.items():
            if num_vars not in self.num_shared_to_vs:
                self.num_shared_to_vs[num_vars] = {v_n}
            else:
                self.num_shared_to_vs[num_vars].add(v_n)

    def iter_variants_and_locs(self, v_n_e):
        for variant_pid, vt_loc_list in self.ve_to_loc[v_n_e].items():
            vt_loc_list = self.__filter_vt_loc_list(variant_pid=variant_pid, vt_loc_list=vt_loc_list)
            if vt_loc_list:
                yield variant_pid, vt_loc_list

    def get_next_id(self, variant_pid, vt_loc, vt_e, block_strand):
        return self.cv_id_bs_to_next[(variant_pid, vt_loc, vt_e)][block_strand]
    

class PanGenome:
    def __init__(
            self,
            original_graph,
            variant_paths_sorted,
            variant_props_ordered,
            variant_labels):
        assert len(variant_paths_sorted) == len(variant_props_ordered) == len(variant_labels)
        self.old_graph = original_graph
        self.old_variant_paths = variant_paths_sorted
        self.variant_props = variant_props_ordered
        self.variant_labels = variant_labels
        # tmp variables
        self.var_indexer = None
        self._c_v_id_to_lb_id = {}
        self._skip_start = set()
        self._candidate_lbs = []
        self._candidate_lbs_set = set()
        # to be generated
        self.new_variant_paths = []
        self.circ_info = {}
        self.colinear_blocks = []
        self._cv_id_map_block_index = {}
        self._vn_divisions = {}  # the division of coverage after repeat resolution
        self.lb_pos_v_name = {}  # (lb_id, lb_pos) mapped to updated vertex name
        self.pan_graph = None
        self.pan_simplified_graph = None  # TODO

    # def gen_simplified_pan_graph(self):
    #     """
    #     Use colinear_blocks and variants and graph to create a new assembly graph
    #     """
    #     self.construct_colinear_blocks()

    def gen_raw_pan_graph(self):
        """
        Use colinear_blocks and variants and graph to create a new assembly graph
        """
        logger.debug("Searching for longest colinear blocks ..")
        self.construct_colinear_blocks()
        logger.debug("Updating paths post repeat resolution ..")
        self._update_paths_according_to_lbs()
        logger.debug("Constructing the new pan genome graph ..")
        self.pan_graph = Assembly()
        # 1. create new vertices
        for pid, (old_variant_p, new_variant_p) in enumerate(zip(self.old_variant_paths, self.new_variant_paths)):
            for vid, ((vn, ve), (new_vn, new_ve)) in enumerate(zip(old_variant_p, new_variant_p)):
                # get information for the colinear blocks
                lb_pos = self._cv_id_map_block_index[(pid, vid)]
                # update the graph
                if new_vn not in self.pan_graph.vertex_info:
                    old_vertex = self.old_graph.vertex_info[vn]
                    self.pan_graph.vertex_info[new_vn] = \
                        Vertex(v_name=new_vn,
                               length=old_vertex.len,
                               coverage=self._vn_divisions[vn][lb_pos] * old_vertex.cov,
                               forward_seq=old_vertex.seq[True])
        # 2. construct the links
        for pid, (old_variant_p, new_variant_p) in enumerate(zip(self.old_variant_paths, self.new_variant_paths)):
            is_circular = self.circ_info[pid]
            len_v = len(old_variant_p)
            for vid, ((old_vn, old_ve), (new_vn, new_ve)) in enumerate(zip(old_variant_p, new_variant_p)):
                # get next vertex info
                if vid == len_v - 1:
                    if is_circular:
                        nxt_vid = 0
                        old_nxt_v, old_nxt_e = old_variant_p[nxt_vid]
                        new_nxt_n, new_nxt_e = new_variant_p[nxt_vid]
                    else:
                        # nxt_vid = new_nxt_n = new_nxt_e = None
                        break
                else:
                    nxt_vid = vid + 1
                    old_nxt_v, old_nxt_e = old_variant_p[nxt_vid]
                    new_nxt_n, new_nxt_e = new_variant_p[nxt_vid]
                # if edge does not exist, transfer the overlap information
                if (new_nxt_n, not new_nxt_e) not in self.pan_graph.vertex_info[new_vn].connections[new_ve]:

                    self.pan_graph.vertex_info[new_vn].connections[new_ve][(new_nxt_n, not new_nxt_e)] = \
                        self.pan_graph.vertex_info[new_nxt_n].connections[not new_nxt_e][(new_vn, new_ve)] =\
                        self.old_graph.vertex_info[old_vn].connections[old_ve][(old_nxt_v, not old_nxt_e)]
        # TODO temporary
        # add path as P (GFA1) or O (ordered collection in GFA2)
        # add label names
        path_info = OrderedDict()
        for go_v, (label_id, path) in enumerate(zip(self.variant_labels, self.new_variant_paths)):
            path_info[label_id] = \
                {"path": path, "circular": self.circ_info[go_v], "prop": float(self.variant_props[go_v])}
        self.pan_graph.paths = path_info

    def _update_paths_according_to_lbs(self):
        self.new_variant_paths = []  # with vertex name renamed
        # record the ongoing copy id of a vertex when there are multiple copies
        vn_go_copy = {}
        # record the map from (lb_id, pos) to new_vertex_name in the new graph,
        #     the key modification to the original vertex name may be a potential addition of _copyX
        self.lb_pos_v_name = {}
        for pid, variant_p in enumerate(self.old_variant_paths):
            self.new_variant_paths.append([])
            for vid, (vn, ve) in enumerate(variant_p):
                # 1. get information for the colinear blocks
                lb_pos = self._cv_id_map_block_index[(pid, vid)]
                # 2. determine if updating the vertex name
                if lb_pos in self.lb_pos_v_name:  # lb_pos exists
                    new_vn = self.lb_pos_v_name[lb_pos]
                else:
                    if vn not in vn_go_copy:
                        vn_go_copy[vn] = 1
                        new_vn = vn
                    else:
                        vn_go_copy[vn] += 1
                        new_vn = f"{vn}_copy{vn_go_copy[vn]}"
                    self.lb_pos_v_name[lb_pos] = new_vn
                # update the path
                self.new_variant_paths[-1].append((new_vn, ve))
            self.new_variant_paths[-1] = tuple(self.new_variant_paths[-1])

    def construct_colinear_blocks(self):
        """ find colinear blocks using greedy search for longest colinear blocks
        # TODO can be improved later
        """
        self.circ_info = {pid: self.old_graph.is_circular_path(self.old_variant_paths[pid])
                          for pid in self.variant_props}
        self.var_indexer = VariantIndexer(
            variant_paths=self.old_variant_paths,
            variant_props=self.variant_props,
            variant_circle_info=self.circ_info)
        if len(self.old_variant_paths) > 1:
            while self.var_indexer.num_shared_to_vs:
                num_shared = max(self.var_indexer.num_shared_to_vs)
                logger.debug(f"searching for blocks shared by {num_shared} variants ..")
                # print("num_shared", num_shared)
                if num_shared == 0:
                    break
                # starting from the least copied ones for efficiency,
                #     then compare the remaining multi-copied ones that is not involved above
                vn_set = self._get_nv_set_with_least_copies(num_shared=num_shared)
                # print("vn_set", vn_set)
                prev_single_candidate = {}
                # if the candidate is empty or no more updating
                while vn_set and vn_set != prev_single_candidate:
                    self._reset_temp_variables()
                    prev_single_candidate = deepcopy(vn_set)
                    # compare all possible correspondence
                    # 1.1. generate the single-copy ones first,
                    #      which has no potential errors and minimize the remaining options
                    self._search_and_index_candidate_lbs(v_n_candidates=vn_set, num_shared=num_shared)
                    # print("candidate lbs", self._candidate_lbs)
                    # 1.2. compare the self._candidate_lbs and remove the conflicting minor ones
                    self._rm_conflicting_lbs()
                    # 1.3 add de-conflicting ones
                    self._add_colinear_blocks()
                    # print("self.var_indexer.num_shared_to_vs", self.var_indexer.num_shared_to_vs)
                    # 1.4 update vn_set, probably add and delete
                    vn_set = self._get_nv_set_with_least_copies(num_shared=num_shared)
                    # print("vn_set", vn_set)
                self.var_indexer.update_cv_distribution()
                # print(self.var_indexer.valid_vt_ids)
                # print(self.var_indexer.num_shared_to_vs)
                # import time
                # time.sleep(1)
        else:
            only_block = []
            pid = 0
            variant_p = self.old_variant_paths[pid]
            for vid, (vn, ve) in enumerate(variant_p):
                only_block.append(((pid, vid, ve),))
            self.colinear_blocks = [tuple(only_block)]
        self._index_colinear_blocks()

    def _index_colinear_blocks(self):
        for go_b, lb in enumerate(self.colinear_blocks):
            for go_pos, lb_unit in enumerate(lb):
                for pid, vid, v_e in lb_unit:
                    self._cv_id_map_block_index[(pid, vid)] = (go_b, go_pos)
        # compute the division of coverage after repeat resolution
        self._vn_divisions = {}
        for pid, variant_p in enumerate(self.old_variant_paths):
            for vid, (vn, ve) in enumerate(variant_p):
                if vn not in self._vn_divisions:
                    self._vn_divisions[vn] = {}
                lb_pos = self._cv_id_map_block_index[(pid, vid)]
                if lb_pos not in self._vn_divisions[vn]:
                    self._vn_divisions[vn][lb_pos] = 0.
                self._vn_divisions[vn][lb_pos] += self.variant_props[pid]
        for vn, lb_pos_dict in self._vn_divisions.items():
            total_val = float(sum(lb_pos_dict.values()))
            if total_val != 0:
                for lb_pos in lb_pos_dict:
                    lb_pos_dict[lb_pos] /= total_val

    def _get_nv_set_with_least_copies(self, num_shared):
        if self.var_indexer.max_copy_to_vs:
            for min_copy in sorted(self.var_indexer.max_copy_to_vs):
                copied_vs = self.var_indexer.max_copy_to_vs[min_copy]
                vn_set = copied_vs & self.var_indexer.num_shared_to_vs.get(num_shared, set())
                if vn_set:
                    return vn_set
        else:
            return set()

    def _add_colinear_blocks(self):
        for this_lb in self._candidate_lbs:
            logger.debug("Adding colinear block: " + str(this_lb))
            self.colinear_blocks.append(this_lb)
            self.var_indexer.archive_ids(this_lb)
        self.var_indexer.update_cv_distribution()

    def _rm_conflicting_lbs(self):
        # sort potential conflicting lbs by weights
        # weight = length * sum(props)
        weights = []
        for this_lb in self._candidate_lbs:
            this_path = []
            prop = 0.
            for cv_list in this_lb:
                variant_pid, vt_loc, vt_e = cv_list[0]
                for variant_pid, vt_loc, vt_e in cv_list:
                    prop += self.variant_props[variant_pid]
                this_path.append((self.old_variant_paths[variant_pid][vt_loc][0], vt_e))
            p_len = self.old_graph.get_path_length(this_path)
            weights.append(p_len * prop)
        sorting_indices = np.argsort(-np.array(weights))
        new_candidate_lbs = []
        update_lb_ids = {}
        for new_lb_id, old_id in enumerate(sorting_indices):
            update_lb_ids[old_id] = new_lb_id
            new_candidate_lbs.append(self._candidate_lbs[old_id])
        # update self._c_v_id_to_lb_id with new lb_id
        for variant_vt_id, lb_id_list in self._c_v_id_to_lb_id.items():
            for go_, old_id in enumerate(lb_id_list):
                lb_id_list[go_] = update_lb_ids[old_id]
        # record lbs that conflicts with lb of higher rank
        rm_lbs = set()
        for new_lb_id, this_lb in enumerate(new_candidate_lbs):
            if new_lb_id not in rm_lbs:
                for cv_list in this_lb:
                    for variant_pid, vt_loc, vt_e in cv_list:
                        variant_vt_id = variant_pid, vt_loc
                        for go_lb in self._c_v_id_to_lb_id[variant_vt_id]:
                            # if the go_lb is not the current one,
                            # which means that there is another linear block (go_lb) was try to employ variant_vt_id
                            # which creates a conflict.
                            if go_lb != new_lb_id:
                                rm_lbs.add(go_lb)
        # remove the conflicting lbs
        for del_lb_id in sorted(rm_lbs, reverse=True):
            self._candidate_lbs_set.remove(new_candidate_lbs[del_lb_id])
            del new_candidate_lbs[del_lb_id]
        self._candidate_lbs = new_candidate_lbs

    def _reset_temp_variables(self):
        self._c_v_id_to_lb_id = {}  # helps to construct the conflicting_lb_id_network
        self._candidate_lbs = []
        self._candidate_lbs_set = set()
        self._skip_start = set()  # used to skip start for single-copy in non-branching senario

    def _update_variant_vt_usage(self, new_lb, go_lb):
        """
        update c_v_id_to_lb_id, the conflicting information
        """
        # record the lb in c_v_id_to_lb_id
        for cv_list in new_lb:
            for variant_pid, vt_loc, vt_e in cv_list:
                variant_vt_id = variant_pid, vt_loc
                if variant_vt_id not in self._c_v_id_to_lb_id:
                    self._c_v_id_to_lb_id[variant_vt_id] = [go_lb]
                else:
                    # conflicting_ids = c_v_id_to_lb_id[c_v_id]
                    self._c_v_id_to_lb_id[variant_vt_id].append(go_lb)

    def _search_and_index_candidate_lbs(self, v_n_candidates, num_shared):
        for start_n in sorted(v_n_candidates):
            if start_n not in self._skip_start and start_n in self.var_indexer.num_shared_to_vs[num_shared]:
                new_lb_list = sorted(self._find_lbs(v_name=start_n))
                for new_lb in new_lb_list:
                    if new_lb not in self._candidate_lbs_set:
                        lb_id = len(self._candidate_lbs)
                        self._candidate_lbs.append(new_lb)
                        self._candidate_lbs_set.add(new_lb)
                        self._update_variant_vt_usage(new_lb=new_lb, go_lb=lb_id)

    def _find_lbs(self, v_name):
        """
        find colinear blocks given a vertex name and number of shared variants (now just for checking)
        """
        indexer = self.var_indexer
        vt_e = True  # only consider one direction of the v_name
        #
        aligning_cvs = []
        num_repeats = []
        for var_pid, vt_loc_list in indexer.iter_variants_and_locs(v_n_e=(v_name, vt_e)):
            aligning_cvs.append((var_pid, vt_loc_list))
            num_repeats.append(len(vt_loc_list))
        start_non_branching = set(num_repeats) == {1}
        #
        all_lb_set = set()
        for indices in comb_indices(*num_repeats):
            is_non_branching = start_non_branching
            ongoing_lb = [[]]
            ongoing_used = set()
            # start one
            next_info = []
            prev_info = []
            for (pid, v_t_l), try_rep in zip(aligning_cvs, indices):
                vt_id, next_e_keep, prev_e_keep = v_t_l[try_rep].get_tuple()
                ongoing_lb[-1].append((pid, vt_id, vt_e))   # <-----
                ongoing_used.add((pid, vt_id))
                next_info.append((pid, vt_id, next_e_keep))
                prev_info.append((pid, vt_id, prev_e_keep))
            ongoing_lb[-1] = tuple(ongoing_lb[-1])
            # print("ongoing_lb", ongoing_lb)
            for block_strand, going_info in enumerate([next_info, prev_info]):
                block_strand = not bool(block_strand)
                # print("shifting strand direction", block_strand)
                consistent = True
                current_e = vt_e
                while consistent:
                    next_going_info = []
                    going_ve = set()
                    proposing_lb = []
                    for pid, vt_id, keep_e in going_info:
                        going_vt_id = indexer.get_next_id(
                            variant_pid=pid, vt_loc=vt_id, vt_e=current_e, block_strand=block_strand)
                        # print(f"pid={pid}, vt_id={vt_id}, block_strand={block_strand} ---> going_vt_id={going_vt_id}")
                        # if it reaches the end, break
                        if going_vt_id is None:
                            consistent = False
                            break
                        # if it reaches back again to a particular pid, vid, break
                        if (pid, going_vt_id) in ongoing_used:
                            consistent = False
                            break
                        else:
                            ongoing_used.add((pid, going_vt_id))
                        # must be valid, not conflicting with previous lbs
                        if (pid, going_vt_id) not in self.var_indexer.valid_vt_ids:
                            consistent = False
                            break
                        this_n, this_e = self.old_variant_paths[pid][going_vt_id]
                        this_e = this_e if keep_e == block_strand else not this_e
                        if not going_ve:
                            going_ve.add((this_n, this_e))
                        if (this_n, this_e) not in going_ve:
                            consistent = False
                            break
                        # add to proposing
                        proposing_lb.append((pid, going_vt_id, this_e))
                        # update the going_vt_id
                        next_going_info.append((pid, going_vt_id, keep_e))
                    if consistent:
                        # extend the ongoing linear block
                        if block_strand:
                            ongoing_lb.append(tuple(proposing_lb))
                        else:
                            ongoing_lb.insert(0, tuple(proposing_lb))
                        # print("ongoing_lb", ongoing_lb)
                        # check if encounter repeats and modify the skip accordingly
                        this_n, this_e = going_ve.pop()
                        if is_non_branching:
                            # if this_n is single-copied
                            if this_n in self.var_indexer.max_copy_to_vs.get(1, {}):
                                self._skip_start.add(this_n)
                            else:
                                is_non_branching = False
                        current_e = this_e
                        # reset
                        going_info = next_going_info
            all_lb_set.add(tuple(ongoing_lb))
        return all_lb_set

