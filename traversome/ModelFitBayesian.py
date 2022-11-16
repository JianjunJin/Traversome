#!/usr/bin/env python
import os.path

from loguru import logger
from collections import OrderedDict
import pymc3 as pm
import theano.tensor as tt
import numpy as np
import arviz as az
from typing import OrderedDict as typingODict


class ModelFitBayesian(object):
    """
    """
    def __init__(self, traversome_obj):
        self.traversome = traversome_obj
        self.num_of_isomers = traversome_obj.num_of_components
        self.trace = None
        # self.graph = traversome_obj.graph

    def run_mcmc(self, n_generations, n_burn, chosen_ids: typingODict[int, bool] = None):
        logger.info("{} subpaths in total".format(len(self.traversome.all_sub_paths)))
        if chosen_ids:
            chosen_ids = OrderedDict([(self.traversome.be_unidentifiable_to[isomer_id], True)
                                      for isomer_id in chosen_ids])
        else:
            chosen_ids = OrderedDict([(isomer_id, True) for isomer_id in self.traversome.merged_components])
        chosen_num = len(chosen_ids)
        with pm.Model() as isomer_model:
            # Because many traversome attributes including subpath information were created using the original component
            # ids, so here we prefer not making traversome.get_multinomial_like_formula complicated. Instead, we create
            # isomer_percents with foo values inserted when that component id is not in chosen_ids.
            real_percents = pm.Dirichlet(name="props", a=np.ones(chosen_num), shape=(chosen_num,))
            isomer_percents = [False] * self.num_of_isomers
            for go_id_id, chosen_id in enumerate(chosen_ids):
                isomer_percents[chosen_id] = real_percents[go_id_id]
            #
            loglike_expression = self.traversome.get_multinomial_like_formula(
                isomer_percents=isomer_percents, log_func=tt.log, within_isomer_ids=set(chosen_ids)).loglike_expression
            pm.Potential("likelihood", loglike_expression)
            # pm.Deterministic("likelihood", likes)
            # pm.DensityDist?
            # pm.Mixture(name="likelihood", w=np.ones(len(components)), comp_dists=components, observed=data)
            # pm.Binomial("path_last", n=n__num_reads_in_range, p=this_prob, observed=x__num_matched_reads)
            # sample from the distribution

            # uses the BFGS optimization algorithm to find the maximum of the log-posterior
            logger.info("Searching the maximum of the log-posterior ..")
            start = pm.find_MAP(model=isomer_model)
            # trace = pm.sample_smc(n_generations, parallel=False)

            # In an upcoming release,
            # pm.sample will return an `arviz.InferenceData` object instead of a `MultiTrace` by default
            logger.info("Using NUTS sampler ..")
            self.trace = pm.sample(
                n_generations,
                tune=n_burn,
                discard_tuned_samples=True,
                cores=1,
                init='adapt_diag',
                start=start,
                return_inferencedata=True)

            logger.info("Summarizing the MCMC traces ..")
            summary = az.summary(self.trace)
            logger.info("\n{}".format(summary))
            axes = az.plot_trace(self.trace)
            fig = axes.ravel()[0].figure
            fig.savefig(os.path.join(self.traversome.outdir, "mcmc.trace_plot.pdf"))
        return OrderedDict([(_c_id, _prop) for _c_id, _prop in zip(chosen_ids, summary["mean"])])
