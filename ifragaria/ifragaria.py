#!/usr/bin/env python

"""
Top-level class for running the CLI
"""

import os
import sys
from loguru import logger

from .Assembly import Assembly
from .GraphAlignRecords import GraphAlignRecords
from .EstCopyDepthFromCov import EstCopyDepthFromCov
from .EstCopyDepthPrecise import EstCopyDepthPrecise


class Fragaria(object):
    """
    keep_temp (bool):
        If True then alignment lenghts are saved to a tmp file during
        the get_align_len_dist function call.
    """

    def __init__(self, gfa, gaf, outdir, keep_temp=False, loglevel="DEBUG"):
        # store input files and params
        self.gaf = gaf
        self.gfa = gfa
        self.outdir = outdir
        self.keep_temp = keep_temp

        # init logger
        self.logfile = os.path.join(self.outdir, "logfile.txt")
        self.setup_logger(loglevel.upper())

        # values to be generated
        self.max_alignment_length = None
        self.isomer_paths_with_labels = None
        self.isomer_lengths = None
        self.num_of_isomers = None


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
        self.get_candidate_isopaths()



    def get_align_len_dist(self):
        """
        Get sorted alignment lengths, optionally save to file 
        and store longest to self.
        """
        logger.debug("Summarizing alignment length distribution")

        # get sorted alignment lengths
        align_len_at_path_sorted = sorted([
            rec.p_align_len for rec in self.alignment.records]
        )

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
        generate candidate paths from ...
        """
        logger.debug("Generating candidate isomer paths ...")
        EstCopyDepthFromCov(graph=self.graph, mode="all").run()

        logger.debug("Fitting candidate isomer paths model...")
        EstCopyDepthPrecise(graph=self.graph).run()
        logger.exception("end of devel.")

        # DEBUGGING THIS COMES NEXT
        # try:
        #     self.isomer_paths_with_labels = assembly_graph.get_all_circular_paths(mode="all", log_handler=self.log)
        # except ProcessingGraphFailed as e:
        #     logger.info("Disentangling circular isomers failed: " + str(e).strip())
        #     logger.info("Disentangling linear isomers ..")
        #     self.isomer_paths_with_labels = self.assembly_graph.get_all_paths(mode="all")

        # # ...
        # self.isomer_lengths = [
        #     get_path_length(isomer_p, self.assembly_graph) 
        #     for (isomer_p, isomer_l) in self.isomer_paths_with_labels
        # ]
        # self.num_of_isomers = len(self.isomer_paths_with_labels)



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
                    "level": "INFO",
                    }
            ]
        }
        logger.configure(**config)
        logger.enable("ifragaria")

        # if logfile exists then reset it.
        if os.path.exists(self.logfile):
            logger.debug('Clearing previous log file.')
            open(self.logfile, 'w').close()
