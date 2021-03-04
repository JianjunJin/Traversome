#!/usr/bin/env python

"""
Top-level class for running the CLI
"""

import os
import sys
from loguru import logger

from ifragaria.Assembly import Assembly
from ifragaria.GraphAlignRecords import GraphAlignRecords
from ifragaria.utils import ProcessingGraphFailed, get_id_range_in_increasing_values
from ifragaria.ModelFitMaxLike import ModelFitMaxLike
from ifragaria.ModelFitBayesian import ModelFitBayesian


class iFragaria(object):
    """
    keep_temp (bool):
        If True then alignment lenghts are saved to a tmp file during
        the get_align_len_dist function call.
    """

    def __init__(self, gfa, gaf, outdir,
                 do_bayesian=False, out_prob_threshold=0.001, keep_temp=False, loglevel="DEBUG"):
        # store input files and params
        self.gaf = gaf
        self.gfa = gfa
        self.outdir = outdir
        self.do_bayesian = do_bayesian,
        self.out_prob_threshold = out_prob_threshold
        self.keep_temp = keep_temp

        # init logger
        self.logfile = os.path.join(self.outdir, "logfile.txt")
        self.loglevel = loglevel.upper()
        self.setup_logger(loglevel.upper())

        # values to be generated
        self.align_len_at_path_sorted = None
        self.max_alignment_length = None
        self.isomer_paths_with_labels = []  # each element is a tuple(path, path_label)
        self.isomer_sizes = None
        self.num_of_isomers = None
        self.isomer_subpath_counters = []  # each element is a dict(sub_path->sub_path_counts)
        self.all_sub_paths = {}

        #
        self.max_like_fit = None
        self.bayesian_fit = None


    def run(self):
        """
        Parse the assembly graph files ...
        """
        self.graph = Assembly(self.gfa)

        self.alignment = GraphAlignRecords(
            self.gaf, 
            min_aligned_path_len=100, 
            min_identity=0.7,
            trim_overlap_with_graph=True,
            assembly_graph=self.graph,
        )

        self.get_align_len_dist()
        logger.debug("Generating candidate isomer paths ...")
        self.get_candidate_isopaths()
        logger.debug("Fitting candidate isomer paths model...")
        if self.do_bayesian:
            pass
        else:
            probs = self.fit_model_using_maximum_likelihood()


        logger.exception("end of devel.")


    def get_align_len_dist(self):
        """
        Get sorted alignment lengths, optionally save to file 
        and store longest to self.
        """
        logger.debug("Summarizing alignment length distribution")

        # get sorted alignment lengths
        align_len_at_path_sorted = sorted([rec.p_align_len for rec in self.alignment])

        # optionally save temp files
        if self.keep_temp:
            opath = os.path.join(self.outdir, "align_len_at_path_sorted.txt")
            with open(opath, "w") as out:
                out.write("\n".join(map(str, align_len_at_path_sorted)))

        # store max value 
        self.max_alignment_length = align_len_at_path_sorted[-1]

        # report result
        logger.info(
            "Maximum alignment length at path: {}".format(
            self.max_alignment_length)
        )


    def get_candidate_isopaths(self):
        """
        generate candidate isomer paths from the graph
        """
        self.graph.estimate_multiplicity_by_cov(mode="all")
        self.graph.estimate_multiplicity_precisely(maximum_copy_num=8, debug=self.loglevel in ("DEBUG", "TRACE", "ALL"))

        try:
            self.isomer_paths_with_labels = self.graph.get_all_circular_isomers(mode="all")
        except ProcessingGraphFailed as e:
            logger.info("Disentangling circular isomers failed: " + str(e).strip())
            logger.info("Disentangling linear isomers ..")
            self.isomer_paths_with_labels = self.graph.get_all_isomers(mode="all")
        #
        self.isomer_sizes = [self.graph.get_path_length(isomer_p)
                             for (isomer_p, isomer_l) in self.isomer_paths_with_labels]
        self.num_of_isomers = len(self.isomer_paths_with_labels)

        # generate subpaths: the binomial sets
        if self.num_of_isomers > 1:
            logger.info("Generating sub-paths ..")
            self.get_sub_paths()
        else:
            logger.warning("Only one genomic configuration found for the input assembly graph.")


    def get_sub_paths(self):
        """
        generate all sub paths and their occurrences for each candidate isomer
        """

        # count sub path occurrences for each candidate isomer and recorded in self.isomer_subpath_counters
        this_overlap = self.graph.overlap()
        self.isomer_subpath_counters = []
        for go_path, (this_path, extra_label) in enumerate(self.isomer_paths_with_labels):
            these_sub_paths = dict()
            num_seg = len(this_path)
            for go_start_v, start_segment in enumerate(this_path):
                this_longest_sub_path = [start_segment]
                this_internal_path_len = 0
                go_next = (go_start_v + 1) % num_seg
                while this_internal_path_len < self.max_alignment_length:
                    next_segment = this_path[go_next]
                    this_longest_sub_path.append(next_segment)
                    this_internal_path_len += self.graph.vertex_info[next_segment[0]].len - this_overlap
                    go_next = (go_next + 1) % num_seg
                # this_internal_path_len -= assembly_graph.vertex_info[this_longest_sub_path[-1][0]].len
                len_this_sub_p = len(this_longest_sub_path)
                for skip_tail in range(len_this_sub_p - 1):
                    this_sub_path = \
                        self.graph.get_standardized_path(this_longest_sub_path[:len_this_sub_p - skip_tail], dc=False)
                    if this_sub_path not in these_sub_paths:
                        these_sub_paths[this_sub_path] = 0
                    these_sub_paths[this_sub_path] += 1
            self.isomer_subpath_counters.append(these_sub_paths)

        # transform self.isomer_subpath_counters to self.all_sub_paths
        self.all_sub_paths = {}
        for go_isomer, sub_paths_group in enumerate(self.isomer_subpath_counters):
            for this_sub_path, this_sub_freq in sub_paths_group.items():
                if this_sub_path not in self.all_sub_paths:
                    self.all_sub_paths[this_sub_path] = {"from_paths": {}, "mapped_records": []}
                self.all_sub_paths[this_sub_path]["from_paths"][go_isomer] = this_sub_freq

        # to simplify downstream calculation, remove shared sub-paths shared by all isomers
        deleted = []
        for this_sub_path, this_sub_path_info in list(self.all_sub_paths.items()):
            if len(this_sub_path_info["from_paths"]) == self.num_of_isomers and \
                    len(set(this_sub_path_info["from_paths"].values())) == 1:
                for sub_paths_group in self.isomer_subpath_counters:
                    deleted.append(this_sub_path)
                    del sub_paths_group[this_sub_path]
                del self.all_sub_paths[this_sub_path]

        # match graph alignments to all_sub_paths
        for go_record, record in enumerate(self.alignment.records):
            this_sub_path = self.graph.get_standardized_path(record.path, dc=False)
            if this_sub_path in self.all_sub_paths:
                self.all_sub_paths[this_sub_path]["mapped_records"].append(go_record)


    def get_likelihood_formula(self, isomer_percents, log_func):
        """
        use a combination of multiple binormial distributions
        :param isomer_percents:
             input sympy.Symbols for maximum likelihood analysis (scipy),
                 e.g. [Symbol("P" + str(isomer_id)) for isomer_id in range(self.num_of_isomers)].
             input pm.Dirichlet for bayesian analysis (pymc3),
                 e.g. pm.Dirichlet(name="props", a=np.ones(isomer_num), shape=(isomer_num,)).
        :param log_func:
             input sympy.log for maximum likelihood analysis using scipy,
             inut tt.log for bayesian analysis using pymc3
        :return: likelihood_formula
        """
        # logger step
        go_sp = 0
        total_sp_num = len(self.all_sub_paths)
        if total_sp_num > 200:
            logger_step = int(total_sp_num/100)
        else:
            logger_step = 1

        loglike_expression = 0
        for this_sub_path, this_sub_path_info in self.all_sub_paths.items():
            go_sp += 1
            internal_len = self.graph.get_path_internal_length(this_sub_path)
            external_len_without_overlap = self.graph.get_path_len_without_terminal_overlaps(this_sub_path)
            left_id, right_id = get_id_range_in_increasing_values(
                min_num=internal_len + 2, max_num=external_len_without_overlap,
                increasing_numbers=self.align_len_at_path_sorted)
            if int((left_id + right_id) / 2) == (left_id + right_id) / 2.:
                median_len = self.align_len_at_path_sorted[int((left_id + right_id) / 2)]
            else:
                median_len = (self.align_len_at_path_sorted[int((left_id + right_id) / 2)] +
                              self.align_len_at_path_sorted[int((left_id + right_id) / 2) + 1]) / 2.
            num_possible_X = self.graph.get_num_of_possible_alignment_start_points(
                read_len=median_len, align_to_path=this_sub_path, path_internal_len=internal_len)
            if num_possible_X < 1:
                continue
            total_starting_points = 0
            for go_isomer, sub_path_freq in this_sub_path_info["from_paths"].items():
                total_starting_points += isomer_percents[go_isomer] * sub_path_freq * num_possible_X
            total_length = 0
            for go_isomer, go_length in enumerate(self.isomer_sizes):
                total_length += isomer_percents[go_isomer] * float(go_length)
            this_prob = total_starting_points / total_length
            n__num_reads_in_range = right_id + 1 - left_id
            x__num_matched_reads = len(this_sub_path_info["mapped_records"])
            loglike_expression += x__num_matched_reads * log_func(this_prob) + \
                                          (n__num_reads_in_range - x__num_matched_reads) * log_func(1 - this_prob)
            if go_sp % logger_step == 0:
                logger.debug("Summarized subpaths: %i/%i" % (go_sp, total_sp_num))
        return loglike_expression


    def fit_model_using_maximum_likelihood(self):
        self.max_like_fit = ModelFitMaxLike(self)
        return self.max_like_fit.run()


    def output_seqs(self, probs, threshold=0.001):
        logger.info("Output seqs: ")
        with open(os.path.join(self.outdir, "isomers.fasta"), "w") as output_handler:
            for go_isomer, this_prob in enumerate(probs):
                if this_prob > threshold:
                    this_seq = self.graph.export_path(self.isomer_paths_with_labels[go_isomer][0])
                    output_handler.write(">" + this_seq.label + " prop=%.4f" % this_prob + "\n" +
                                         this_seq.seq + "\n")
                    logger.info(">" + this_seq.label + " prop=%.4f" % this_prob)


    def setup_logger(self, loglevel="INFO"):
        """
        Configure Loguru to log to stdout and logfile.
        """
        # add stdout logger
        config = {
            "handlers": [
                {
                    "sink": sys.stdout, 
                    "format": (
                        "{time:YYYY-MM-DD-hh:mm} | "
                        "<magenta>{file: >22} | </magenta>"
                        "<cyan>{function: <22} | </cyan>"
                        "<level>{message}</level>"
                    ),
                    "level": loglevel,
                    },
                {
                    "sink": self.logfile,                   
                    "format": "{time:YYYY-MM-DD} | {function} | {message}",
                    "level": loglevel,
                    }
            ]
        }
        logger.configure(**config)
        logger.enable("ifragaria")

        # if logfile exists then reset it.
        if os.path.exists(self.logfile):
            logger.debug('Clearing previous log file.')
            open(self.logfile, 'w').close()
