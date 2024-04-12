#!/usr/bin/env python
from typing import Set
from collections import OrderedDict
from loguru import logger
from traversome.utils import LogLikeFormulaInfo


class PathMultinomialModel:
    def __init__(self, variant_sizes, variant_topos, bins_list, all_sub_paths):
        self.variant_sizes = variant_sizes
        self.variant_topos = variant_topos
        self.num_put_variants = len(variant_sizes)
        self.bins_list = bins_list
        self.all_sub_paths = all_sub_paths  # only used for assessing read_path coverage
        self.sample_size = None

    def get_like_formula_old(self, variant_percents, log_func, within_variant_ids: Set = None):
        """
        use a combination of multiple multinomial distributions
        :param variant_percents:
             input symengine.Symbols for maximum likelihood analysis (scipy),
                 e.g. [Symbol("P" + str(variant_id)) for variant_id in range(self.num_put_variants)].
             input pm.Dirichlet for bayesian analysis (pymc3),
                 e.g. pm.Dirichlet(name="vid", a=np.ones(variant_num), shape=(variant_num,)).
        :param log_func:
             input symengine.log for maximum likelihood analysis using scipy,
             input tt.log for bayesian analysis using pymc3
        :param within_variant_ids:
             constrain the variant testing scope. Test all variants by default.
                 e.g. set([0, 2])
        :return: LogLikeFormulaInfo object
        """
        if not within_variant_ids or within_variant_ids == set(range(self.num_put_variants)):
            within_variant_ids = None
        # total length (all possible matches, ignoring margin effect if not circular)
        total_length = 0
        if within_variant_ids:
            for go_variant, go_length in enumerate(self.variant_sizes):
                if go_variant in within_variant_ids:
                    total_length += variant_percents[go_variant] * float(go_length)
        else:
            for go_variant, go_length in enumerate(self.variant_sizes):
                total_length += variant_percents[go_variant] * float(go_length)

        # prepare subset of all_sub_paths in a list
        these_sp_info = OrderedDict()
        # if within_variant_ids:
        #     for go_sp, (this_sub_path, this_sp_info) in enumerate(self.all_sub_paths.items()):
        #         if set(this_sp_info.from_variants) & within_variant_ids:
        #             these_sp_info[go_sp] = this_sp_info
        # else:
        for go_sp, (this_sub_path, this_sp_info) in enumerate(self.all_sub_paths.items()):
            these_sp_info[go_sp] = this_sp_info
        # clean zero expectations to avoid nan formula
        for check_sp in list(these_sp_info):
            if these_sp_info[check_sp].num_possible_X < 1:
                del these_sp_info[check_sp]
                continue
            if within_variant_ids and not (set(these_sp_info[check_sp].from_variants) & within_variant_ids):
                del these_sp_info[check_sp]

        # get the observations
        observations = [len(this_sp_info.mapped_records) for this_sp_info in these_sp_info.values()]

        # sub path possible matches
        logger.debug("  Formulating the subpath probabilities ..")
        this_sbp_Xs = [these_sp_info[_go_sp_].num_possible_X for _go_sp_ in these_sp_info]
        for go_valid_sp, this_sp_info in enumerate(these_sp_info.values()):
            variant_weight = 0
            if within_variant_ids:
                sub_from_iso = {_go_iso_: _sp_freq_
                                for _go_iso_, _sp_freq_ in this_sp_info.from_variants.items()
                                if _go_iso_ in within_variant_ids}
                for go_variant, sp_freq in sub_from_iso.items():
                    variant_weight += variant_percents[go_variant] * sp_freq
            else:
                for go_variant, sp_freq in this_sp_info.from_variants.items():
                    variant_weight += variant_percents[go_variant] * sp_freq
            this_sbp_Xs[go_valid_sp] *= variant_weight
        this_sbp_prob = [_sbp_X / total_length for _sbp_X in this_sbp_Xs]

        # mark2, if include this, better removing code block under mark1
        # leading to nan like?
        # # the other unrecorded observed matches
        # observations.append(len(self.alignment.raw_records) - sum(observations))
        # # the other unrecorded expected matches
        # # Theano may not support sum, use for loop instead
        # other_prob = 1
        # for _sbp_prob in this_sbp_prob:
        #     other_prob -= _sbp_prob
        # this_sbp_prob.append(other_prob)

        for go_valid_sp, go_sp in enumerate(these_sp_info):
            logger.trace("  Subpath {} observation: {}".format(go_sp, observations[go_valid_sp]))
            logger.trace("  Subpath {} probability: {}".format(go_sp, this_sbp_prob[go_valid_sp]))
        # logger.trace("  Rest observation: {}".format(observations[-1]))
        # logger.trace("  Rest probability: {}".format(this_sbp_prob[-1]))

        # likelihood
        # logger.debug(f"  Summing up likelihood function from {len(observations)} bins ..")
        logger.info(f"  Summing up likelihood function from {len(observations)} bins ..")
        loglike_expression = 0
        for go_sp, obs in enumerate(observations):
            loglike_expression += log_func(this_sbp_prob[go_sp]) * obs
        variable_size = len(within_variant_ids) if within_variant_ids else self.num_put_variants
        self.sample_size = sum(observations)

        return LogLikeFormulaInfo(loglike_expression, variable_size, self.sample_size)

    def get_like_formula(self, variant_percents, log_func, within_variant_ids: Set = None):
        """use multiple multinomial distributions
        :param variant_percents:
             input symengine.Symbols for maximum likelihood analysis (scipy),
                 e.g. [Symbol("P" + str(variant_id)) for variant_id in range(self.num_put_variants)].
             input pm.Dirichlet for bayesian analysis (pymc3),
                 e.g. pm.Dirichlet(name="vid", a=np.ones(variant_num), shape=(variant_num,)).
        :param log_func:
             input symengine.log for maximum likelihood analysis using scipy,
             input tt.log for bayesian analysis using pymc3
        :param within_variant_ids:
             constrain the variant testing scope. Test all variants by default.
                 e.g. set([0, 2])
        :return: LogLikeFormulaInfo object
        """
        if not within_variant_ids or within_variant_ids == set(range(self.num_put_variants)):
            within_variant_ids = None
        # total length (all possible matches, ignoring margin effect if not circular)
        # TODO use self.variant_topos to consider margin effect
        total_length = 0
        if within_variant_ids:
            for go_variant, go_length in enumerate(self.variant_sizes):
                if go_variant in within_variant_ids:
                    total_length += variant_percents[go_variant] * float(go_length)
        else:
            for go_variant, go_length in enumerate(self.variant_sizes):
                total_length += variant_percents[go_variant] * float(go_length)

        # clean zero expectations to avoid nan formula
        check_a = 0
        while check_a < len(self.bins_list):
            bins = self.bins_list[check_a]
            check_b = 0
            while check_b < len(bins.rp_bins):
                if bins.rp_bins[check_b].num_possible_X < 1:
                    del bins.rp_bins[check_b]
                else:
                    check_b += 1
            if bins.rp_bins:
                check_a += 1
            else:
                del self.bins_list[check_a]

        # sub path possible matches
        logger.debug("  Formulating the probabilities ..")
        bin_probs = []
        bin_observations = []
        for bins in self.bins_list:
            for rp_bin in bins.rp_bins:
                variant_weight = 0
                if within_variant_ids:
                    for go_variant, sp_freq in rp_bin.from_variants.items():
                        if go_variant in within_variant_ids:
                            variant_weight += variant_percents[go_variant] * sp_freq
                else:
                    for go_variant, sp_freq in rp_bin.from_variants.items():
                        variant_weight += variant_percents[go_variant] * sp_freq
                this_Xs = variant_weight * rp_bin.num_possible_X
                bin_probs.append(this_Xs / total_length)
                bin_observations.append(rp_bin.num_matched)
        # likelihood
        logger.debug("  Summing up subpath likelihood function ..")
        loglike_expression = 0
        for b_prob, b_obs in zip(bin_probs, bin_observations):
            loglike_expression += log_func(b_prob) * b_obs
        variable_size = len(within_variant_ids) if within_variant_ids else self.num_put_variants
        self.sample_size = sum(bin_observations)

        return LogLikeFormulaInfo(loglike_expression, variable_size, self.sample_size)
