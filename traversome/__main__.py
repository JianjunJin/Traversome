#! /usr/bin/env python

"""
Command-line Interface to traversome
"""

import os
import sys
from optparse import OptionParser

from .traversome import Traversome



def get_options():
    "parse command line with OptionParser"

    # init object and set usage
    parser = OptionParser(usage="traversome -g graph.gfa -a align.gaf -o .")

    # create param flags
    parser.add_option(
        "-g", 
        dest="graph_file",
        help="GFA format Graph file. ",
    )
    parser.add_option(
        "-a", 
        dest="gaf_file",
        help="GAF format alignment file. ",
    )
    parser.add_option(
        "-o", 
        dest="output_dir",
        help="Output directory. ",
    )
    parser.add_option(
        "--out-seq-prop",
        dest="out_seq_prop",
        default=0.001,
        type=float,
        help="(Currently working for ML only) Output sequences with proportion >= %default")
    parser.add_option(
        "-B", 
        dest="do_bayesian", 
        action="store_true", 
        default=False,
        help="Use Bayesian implementation. ",
    )
    parser.add_option(
        "--loglevel", 
        dest="loglevel", 
        default="INFO", 
        action="store",
        help="Default=INFO. Use DEBUG for more, ERROR for less.",
    )
    parser.add_option(
        "--keep-temp", 
        dest="keep_temp", 
        default=False, 
        action="store_true",
        help="Keep temporary files for debug. Default: %default",
    )

    
    # parse and check param args
    options, argv = parser.parse_args()

    # check for required flags
    if not (options.graph_gfa and options.gaf_file and options.output_dir):
        parser.print_help()
        sys.exit()

    # check file paths
    if not os.path.isfile(options.graph_gfa):
        raise IOError(options.graph_gfa + " not found/valid!")
    if not os.path.isfile(options.gaf_file):
        raise IOError(options.gaf_file + " not found/valid!")
    if not os.path.exists(options.output_dir):
        os.mkdir(options.output_dir)
    assert 0 <= options.out_seq_prop <= 1

    return options



def main():
    "command line interface function workflow"
    # parse command line params and get logger
    opts = get_options()

    # create object with params
    frag = Traversome(
        graph=opts.graph_gfa,
        alignment=opts.gaf_file,
        outdir=opts.output_dir,
        do_bayesian=opts.do_bayesian,
        out_prob_threshold=opts.out_seq_prop,
        keep_temp=opts.keep_temp,
        loglevel=opts.loglevel
    )

    # run the object
    frag.run()


if __name__ == "__main__":
    main()