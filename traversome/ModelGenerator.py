#!/usr/bin/env python
from typing import Set
from collections import OrderedDict
from loguru import logger
from traversome.utils import LogLikeFormulaInfo


class PathMultinomialModel:
    def __init__(self, variant_sizes, all_sub_paths):
        self.variant_sizes = variant_sizes
        self.num_put_variants = len(variant_sizes)
        self.all_sub_paths = all_sub_paths
        self.sample_size = None

    def get_like_formula(self, variant_percents, log_func, within_variant_ids: Set = None):
        """
        use a combination of multiple multinomial distributions
        :param variant_percents:
             input symengine.Symbols for maximum likelihood analysis (scipy),
                 e.g. [Symbol("P" + str(variant_id)) for variant_id in range(self.num_put_variants)].
             input pm.Dirichlet for bayesian analysis (pymc3),
                 e.g. pm.Dirichlet(name="comp", a=np.ones(variant_num), shape=(variant_num,)).
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

        # calculate the observations
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

        # for go_sp, this_sp_info in these_sp_info.items():
        #     for record_id in this_sp_info.mapped_records:
        #         this_len_sp_xs = self.pal_len_sbp_Xs[self.alignment.raw_records[record_id].p_align_len]
        #         ...

        # likelihood
        logger.debug("  Summing up subpath likelihood function ..")
        loglike_expression = 0
        for go_sp, obs in enumerate(observations):
            loglike_expression += log_func(this_sbp_prob[go_sp]) * obs
        variable_size = len(within_variant_ids) if within_variant_ids else self.num_put_variants
        self.sample_size = sum(observations)

        return LogLikeFormulaInfo(loglike_expression, variable_size, self.sample_size)
