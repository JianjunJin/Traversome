#! /usr/bin/env python


import sys
import os
import time
import logging
from optparse import OptionParser
from sympy import Symbol, log, lambdify
from scipy import optimize
import numpy as np
from ifragaria.assembly_parser import Assembly, ProcessingGraphFailed
from ifragaria.alignment_parser import GraphAlignRecords
from ifragaria.pip_control_func import simple_log, timed_log
np.seterr(divide="ignore", invalid="ignore")
import matplotlib.pyplot as plt
import pymc3 as pm
import theano.tensor as tt
import theano

theano.config.gcc.cxxflags = "-fbracket-depth=16000"  # will significantly slow down the compiling


def get_options(description):
    parser = OptionParser(usage="estimate_isomer_frequencies.py -g graph.gfa -a align.gaf")
    parser.add_option("-g", dest="graph_file",
                      help="GFA format Graph file. ")
    parser.add_option("-a", dest="gaf_file",
                      help="GAF format alignment file. ")
    parser.add_option("-o", dest="output_dir",
                      help="Output directory. ")
    parser.add_option("-B", dest="do_bayesian", action="store_true", default=False,
                      help="Do a bayesian analysis. ")
    parser.add_option("--out-seq-prop", dest="out_seq_prop", default=0.001, type=float,
                      help="(Currently working for ML only) Output sequences with proportion >= %default")
    # parser.add_option("-p", dest="paths",
    #                   help="Optional. Input file containing paths rather than exporting "
    #                        "all circular paths from the assembly graph. "
    #                        "Paths will be checked along the graph and circularized if its head connects its tail. "
    #                        "Each path per line in the format of signed contig names separated by commas, "
    #                        "e.g. 1,-2,3,2")
    parser.add_option("--debug", dest="debug", default=False, action="store_true",
                      help="Debug mode. Default: %default")
    parser.add_option("--keep-temp", dest="keep_temp", default=False, action="store_true",
                      help="Keep temporary files for debug. Default: %default")
    options, argv = parser.parse_args()
    if not (options.graph_gfa and options.gaf_file and options.output_dir):
        parser.print_help()
        sys.exit()
    else:
        if not os.path.isfile(options.graph_gfa):
            raise IOError(options.graph_gfa + " not found/valid!")
        if not os.path.isfile(options.gaf_file):
            raise IOError(options.gaf_file + " not found/valid!")
        if not os.path.exists(options.output_dir):
            os.mkdir(options.output_dir)
        assert 0 <= options.out_seq_prop <= 1
        log_handler = simple_log(logging.getLogger(), options.output_dir, "ifragaria", file_handler_mode="a")
        log_handler.info(description)
        log_handler.info("Python " + str(sys.version).replace("\n", " "))
        log_handler.info("WORKING DIR: " + os.getcwd())
        log_handler.info(" ".join(["\"" + arg + "\"" if " " in arg else arg for arg in sys.argv]) + "\n")
        log_handler = timed_log(
            log_handler, options.output_dir, "ifragaria", log_level="DEBUG" if options.debug else "INFO")
        return options, log_handler


def sort_path(input_path):
    reverse_path = [(segment, not strand) for segment, strand in input_path][::-1]
    return sorted([input_path, reverse_path])[0]


def get_sub_paths(paths_with_labels, assembly_graph, max_internal_sub_path_len):
    this_overlap = assembly_graph.overlap()
    sub_paths_counter_list = []
    for go_path, (this_path, extra_label) in enumerate(paths_with_labels):
        these_sub_paths = dict()
        num_seg = len(this_path)
        for go_start_v, start_segment in enumerate(this_path):
            this_longest_sub_path = [start_segment]
            this_internal_path_len = 0
            go_next = (go_start_v + 1) % num_seg
            while this_internal_path_len < max_internal_sub_path_len:
                next_segment = this_path[go_next]
                this_longest_sub_path.append(next_segment)
                this_internal_path_len += assembly_graph.vertex_info[next_segment[0]].len - this_overlap
                go_next = (go_next + 1) % num_seg
            # this_internal_path_len -= assembly_graph.vertex_info[this_longest_sub_path[-1][0]].len
            len_this_sub_p = len(this_longest_sub_path)
            for skip_tail in range(len_this_sub_p - 1):
                this_sub_path = tuple(sort_path(this_longest_sub_path[:len_this_sub_p - skip_tail]))
                if this_sub_path not in these_sub_paths:
                    these_sub_paths[this_sub_path] = 0
                these_sub_paths[this_sub_path] += 1
        sub_paths_counter_list.append(these_sub_paths)
    return sub_paths_counter_list


def get_internal_length_from_path(input_path, assembly_graph):
    assert len(input_path) > 1
    # internal_len is allowed to be negative when this_overlap > 0 and len(the_repeat_path) == 2
    this_overlap = assembly_graph.overlap()
    internal_len = -this_overlap
    for seg_name, seg_strand in input_path[1:-1]:
        internal_len += assembly_graph.vertex_info[seg_name].len - this_overlap
    return internal_len


def get_path_len_without_terminal_overlaps(input_path, assembly_graph):
    assert len(input_path) > 1
    this_overlap = assembly_graph.overlap()
    path_len = -this_overlap
    for seg_name, seg_strand in input_path:
        path_len += assembly_graph.vertex_info[seg_name].len - this_overlap
    return path_len


def get_id_range_in_increasing_values(min_num, max_num, increasing_numbers):
    assert max_num >= min_num
    len_list = len(increasing_numbers)
    left_id = 0
    while left_id < len_list and increasing_numbers[left_id] < min_num:
        left_id += 1
    right_id = len_list - 1
    while right_id > -1 and increasing_numbers[right_id] > max_num:
        right_id -= 1
    return left_id, right_id


# given a read with certain length (i.e. median length of all candidate reads), calculate how many possible sites
def get_fitting_sites_in_range(read_len, input_path, internal_len, assembly_graph):
    maximum_num_cat = read_len - internal_len - 2
    left_trim = max(maximum_num_cat - assembly_graph.vertex_info[input_path[0][0]].len - assembly_graph.overlap(), 0)
    right_trim = max(maximum_num_cat - assembly_graph.vertex_info[input_path[-1][0]].len - assembly_graph.overlap(), 0)
    return maximum_num_cat - left_trim - right_trim


# TODO merge to assembly
def is_circular_path(input_path, assembly_graph):
    return (input_path[-1][0], not input_path[-1][1]) in \
           assembly_graph.vertex_info[input_path[0][0]].connections[input_path[0][1]]


# TODO merge to assembly
def get_path_length(input_path, assembly_graph):
    circular_len = sum([assembly_graph.vertex_info[name].len - assembly_graph.overlap() for name, strand in input_path])
    return circular_len + assembly_graph.overlap() * int(is_circular_path(input_path, assembly_graph))


def iso_mcmc(isomer_num, all_sub_paths, assembly_graph, align_len_at_path_sorted, isomer_lengths,
         n_generations, n_burn, log_handler):
    log_handler.info(str(len(all_sub_paths)) + " subpaths in total")
    with pm.Model() as isomer_model:
        isomer_percents = pm.Dirichlet(name="props", a=np.ones(isomer_num), shape=(isomer_num,))
        count = 0
        likes = 0
        for this_sub_path, this_sub_path_info in all_sub_paths.items():
            internal_len = get_internal_length_from_path(this_sub_path, assembly_graph)
            external_len_without_overlap = get_path_len_without_terminal_overlaps(this_sub_path, assembly_graph)
            left_id, right_id = get_id_range_in_increasing_values(
                min_num=internal_len + 2, max_num=external_len_without_overlap,
                increasing_numbers=align_len_at_path_sorted)
            if int((left_id + right_id) / 2) == (left_id + right_id) / 2.:
                median_len = align_len_at_path_sorted[int((left_id + right_id) / 2)]
            else:
                median_len = (align_len_at_path_sorted[int((left_id + right_id) / 2)] +
                              align_len_at_path_sorted[int((left_id + right_id) / 2) + 1]) / 2.
            num_fitting_sites = get_fitting_sites_in_range(
                read_len=median_len, input_path=this_sub_path, internal_len=internal_len, assembly_graph=assembly_graph)
            if num_fitting_sites < 1:
                continue
            total_starting_points = 0
            for go_isomer, sub_path_freq in this_sub_path_info["from_isomers"].items():
                total_starting_points += isomer_percents[go_isomer] * sub_path_freq * num_fitting_sites
            total_length = 0
            for go_isomer, go_length in enumerate(isomer_lengths):
                total_length += float(go_length) * isomer_percents[go_isomer]
            this_prob = total_starting_points / total_length
            n__num_reads_in_range = right_id + 1 - left_id
            x__num_matched_reads = len(this_sub_path_info["mapped_records"])
            count += 1
            # if count % 5 == 0:
            #     log_handler.info(str(count))
            likes += x__num_matched_reads * tt.log(this_prob) + \
                     (n__num_reads_in_range - x__num_matched_reads) * tt.log(1 - this_prob)
        pm.Potential("likelihood", likes)
        # pm.Deterministic("likelihood", likes)
        # pm.DensityDist?
        # pm.Mixture(name="likelihood", w=np.ones(len(components)), comp_dists=components, observed=data)
        # pm.Binomial("path_last", n=n__num_reads_in_range, p=this_prob, observed=x__num_matched_reads)
        # sample from the distribution
        start = pm.find_MAP(model=isomer_model)
        # trace = pm.sample_smc(n_generations, parallel=False)
        trace = pm.sample(
            n_generations, tune=n_burn, discard_tuned_samples=True, cores=1, init='adapt_diag', start=start)
        log_handler.info(pm.summary(trace))
    return trace


def get_neg_likelihood_of_iso_freq(
        symbol_dict_of_isomer_percents, all_sub_paths, assembly_graph, align_len_at_path_sorted, isomer_lengths,
        scipy_style=True):
    # use a mixture of multiple binormial distributions
    maximum_loglike_expression = 0
    for this_sub_path, this_sub_path_info in all_sub_paths.items():
        internal_len = get_internal_length_from_path(this_sub_path, assembly_graph)
        external_len_without_overlap = get_path_len_without_terminal_overlaps(this_sub_path, assembly_graph)
        left_id, right_id = get_id_range_in_increasing_values(
            min_num=internal_len + 2, max_num=external_len_without_overlap,
            increasing_numbers=align_len_at_path_sorted)
        if int((left_id + right_id) / 2) == (left_id + right_id) / 2.:
            median_len = align_len_at_path_sorted[int((left_id + right_id) / 2)]
        else:
            median_len = (align_len_at_path_sorted[int((left_id + right_id) / 2)] +
                          align_len_at_path_sorted[int((left_id + right_id) / 2) + 1]) / 2.
        num_fitting_sites = get_fitting_sites_in_range(
            read_len=median_len, input_path=this_sub_path, internal_len=internal_len, assembly_graph=assembly_graph)

        if num_fitting_sites < 1:
            continue

        total_starting_points = 0
        for go_isomer, sub_path_freq in this_sub_path_info["from_isomers"].items():
            total_starting_points += symbol_dict_of_isomer_percents[go_isomer] * sub_path_freq * num_fitting_sites
        total_length = 0
        for go_isomer, go_length in enumerate(isomer_lengths):
            total_length += symbol_dict_of_isomer_percents[go_isomer] * float(go_length)
        this_prob = total_starting_points / total_length

        n__num_reads_in_range = right_id + 1 - left_id
        x__num_matched_reads = len(this_sub_path_info["mapped_records"])
        maximum_loglike_expression += x__num_matched_reads * log(this_prob) + \
                                      (n__num_reads_in_range - x__num_matched_reads) * log(1 - this_prob)
        # print(this_sub_path, "~", x__num_matched_reads * log(this_prob) + \
        #                               (n__num_reads_in_range - x__num_matched_reads) * log(1 - this_prob))
    # print(maximum_loglike_expression)
    neg_likelihood_of_iso_freq = lambdify(
        args=[symbol_dict_of_isomer_percents[isomer_id] for isomer_id in range(len(symbol_dict_of_isomer_percents))],
        expr=-maximum_loglike_expression)
    if scipy_style:
        # for compatibility between scipy and sympy
        # positional arguments -> single tuple argument
        def neg_likelihood_of_iso_freq_single_arg(x):
            return neg_likelihood_of_iso_freq(*tuple(x))
        return neg_likelihood_of_iso_freq_single_arg
    else:
        return neg_likelihood_of_iso_freq


def minimize_neg_likelihood(likelihood_function, num_isomers, verbose):
    # all proportions should be in range [0, 1] and sum up to 1.
    constraints = ({"type": "eq", "fun": lambda x: sum(x) - 1})
    other_optimization_options = {"disp": verbose, "maxiter": 1000, "ftol": 1.0e-6, "eps": 1.0e-10}
    count_run = 0
    success_runs = []
    while count_run < 100:
        initials = np.random.random(num_isomers)
        initials /= sum(initials)
        # print("initials", initials)
        # np.full(shape=num_of_isomers, fill_value=float(1. / num_of_isomers), dtype=np.float)
        result = optimize.minimize(
            fun=likelihood_function,
            x0=initials,
            jac=False, method='SLSQP', constraints=constraints, bounds=[(0., 1.0)] * num_isomers,
            options=other_optimization_options)
        # bounds=[(-1.0e-9, 1.0)] * num_isomers will violate bound constraints and cause ValueError
        if result.success:
            success_runs.append(result)
            if len(success_runs) > 10:
                break
        count_run += 1
        # sys.stdout.write(str(count_run) + "\b" * len(str(count_run)))
        # sys.stdout.flush()
    return success_runs


def main():
    time0 = time.time()
    options, log_handler = get_options(description="\niFragaria\n")
    try:
        log_handler.info("Parsing graph ..")
        assembly_graph = Assembly(options.graph_gfa)

        log_handler.info("Parsing graph alignment ..")
        graph_alignment = GraphAlignRecords(
            options.gaf_file, min_aligned_path_len=100, min_identity=0.70,
            trim_overlap_with_graph=True, assembly_graph=assembly_graph)

        log_handler.info("Summarizing alignment length distribution ..")
        align_len_at_path_sorted = sorted([record.p_align_len for record in graph_alignment])
        if options.keep_temp:
            open(os.path.join(options.output_dir, "align_len_at_path_sorted.txt"), "w").\
                writelines([str(x) + "\n" for x in align_len_at_path_sorted])
        max_align_len_at_path = align_len_at_path_sorted[-1]
        log_handler.info("Maximum alignment length at path: %s" % max_align_len_at_path)

        # TODO: generate candidate isomer paths using reads evidence to simplify
        log_handler.info("Generating candidate isomer paths ..")
        assembly_graph.estimate_copy_and_depth_by_cov(mode="all", log_handler=log_handler, verbose=options.debug)
        assembly_graph.estimate_copy_and_depth_precisely(log_handler=log_handler, verbose=options.debug)
        try:
            isomer_paths_with_labels = assembly_graph.get_all_circular_paths(mode="all", log_handler=log_handler)
        except ProcessingGraphFailed as e:
            log_handler.info("Disentangling circular isomers failed: " + str(e).strip())
            log_handler.info("Disentangling linear isomers ..")
            isomer_paths_with_labels = assembly_graph.get_all_paths(mode="all")
        isomer_lengths = \
            np.array([get_path_length(isomer_p, assembly_graph) for isomer_p, isomer_l in isomer_paths_with_labels])
        num_of_isomers = len(isomer_paths_with_labels)

        if num_of_isomers > 1:
            log_handler.info("Generating sub-paths ..")
            sub_paths_counter_list = get_sub_paths(isomer_paths_with_labels, assembly_graph, max_align_len_at_path)

            # generate candidate sub-paths table: all_sub_paths
            all_sub_paths = {}
            for go_isomer, sub_paths_group in enumerate(sub_paths_counter_list):
                for this_sub_path, this_sub_freq in sub_paths_group.items():
                    if this_sub_path not in all_sub_paths:
                        all_sub_paths[this_sub_path] = {"from_isomers": {}, "mapped_records": []}
                    all_sub_paths[this_sub_path]["from_isomers"][go_isomer] = this_sub_freq

            # to simplify downstream calculation, remove shared sub-paths shared by all isomers
            deleted = []
            for this_sub_path, this_sub_path_info in list(all_sub_paths.items()):
                if len(this_sub_path_info["from_isomers"]) == num_of_isomers and \
                        len(set(this_sub_path_info["from_isomers"].values())) == 1:
                    for sub_paths_group in sub_paths_counter_list:
                        deleted.append(this_sub_path)
                        del sub_paths_group[this_sub_path]
                    del all_sub_paths[this_sub_path]

            # match graph alignments to all_sub_paths
            for go_record, record in enumerate(graph_alignment.records):
                this_sub_path = tuple(sort_path(record.path))
                if this_sub_path in all_sub_paths:
                    all_sub_paths[this_sub_path]["mapped_records"].append(go_record)
            # ??? remove sub-paths without records
            # ??? re-evaluate isomer paths after removing sub-paths

            # check
            # for m in sub_paths_counter_list:
            #     print("------------")
            #     for k, v in m.items():
            #         print(k, v, len(all_sub_paths[k]["mapped_records"]), get_internal_length_from_path(k, assembly_graph))

            """ ML or Bayesian """
            if options.do_bayesian:
                log_handler.info("Running MCMC .. ")
                trace = iso_mcmc(num_of_isomers, all_sub_paths, assembly_graph, align_len_at_path_sorted, isomer_lengths,
                     n_generations=10000, n_burn=1000, log_handler=log_handler)
                plt.plot()
                # call model arguments
                pm.traceplot(trace)
                plt.savefig(os.path.join(options.output_dir, "mcmc.pdf"))
            else:
                """ find proportion that maximize the likelihood """
                symbol_dict_of_isomer_percents = \
                    {isomer_id: Symbol("P" + str(isomer_id)) for isomer_id in range(num_of_isomers)}
                log_handler.info("Generating the likelihood function .. ")
                neg_loglike_function = get_neg_likelihood_of_iso_freq(
                    symbol_dict_of_isomer_percents, all_sub_paths, assembly_graph, align_len_at_path_sorted,
                    isomer_lengths)
                log_handler.info("Maximizing the likelihood function .. ")
                success_runs = minimize_neg_likelihood(neg_loglike_function, num_of_isomers, options.debug)
                if success_runs:
                    # for run_res in sorted(success_runs, key=lambda x: x.fun):
                    #     log_handler.info(str(run_res.fun) + str([round(m, 8) for m in run_res.x]))
                    log_handler.info("Proportion: %s Log-likelihood: %s" % (success_runs[0].x, -success_runs[0].fun))
                    log_handler.info("Output seqs: ")
                    with open(os.path.join(options.output_dir, "isomers.fasta"), "w") as output_handler:
                        for go_isomer, this_prob in enumerate(success_runs[0].x):
                            if this_prob > options.out_seq_prop:
                                this_seq = assembly_graph.export_path(isomer_paths_with_labels[go_isomer][0])
                                output_handler.write(">" + this_seq.label + " prop=%.4f" % this_prob + "\n" +
                                                     this_seq.seq + "\n")
                                log_handler.info(">" + this_seq.label + " prop=%.4f" % this_prob)
        log_handler = simple_log(log_handler, options.output_dir, "ifragaria")
        log_handler.info("\nTotal cost " + "%.2f" % (time.time() - time0) + " s")
        log_handler.info("Thank you!")
    except:
        log_handler.exception("")


if __name__ == '__main__':
    main()