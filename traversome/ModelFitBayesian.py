#!/usr/bin/env python
import os.path

from loguru import logger
from collections import OrderedDict
try:
    import pymc3 as pm
except ModuleNotFoundError:
    # newer version of pymc
    import pymc as pm
try:
    import theano.tensor as tt
except ModuleNotFoundError:
    # newer version of pymc uses aesara (Theano-PyMC), instead of Theano which was no more maintained
    import aesara.tensor as tt
import numpy as np
import arviz as az
from typing import OrderedDict as typingODict


class ModelFitBayesian(object):
    """
    """
    def __init__(self, traversome_obj):
        self.traversome = traversome_obj
        self.num_of_variants = traversome_obj.num_put_variants
        self.trace = None
        # self.graph = traversome_obj.graph

    def run_mcmc(self, n_generations, n_burn, chosen_ids: typingODict[int, bool] = None):
        logger.info("{} subpaths in total".format(len(self.traversome.all_sub_paths)))
        if chosen_ids:
            chosen_ids = OrderedDict([(self.traversome.be_unidentifiable_to[variant_id], True)
                                      for variant_id in chosen_ids])
        else:
            chosen_ids = OrderedDict([(variant_id, True) for variant_id in self.traversome.merged_variants])
        chosen_num = len(chosen_ids)
        with pm.Model() as variants_model:
            # Because many traversome attributes including subpath information were created using the original variant
            # ids, so here we prefer not making traversome.get_multinomial_like_formula complicated. Instead, we create
            # variant_percents with foo values inserted when that variant id is not in chosen_ids_set.
            real_percents = pm.Dirichlet(name="comp", a=np.ones(chosen_num), shape=(chosen_num,))
            variant_percents = [False] * self.num_of_variants
            for go_id_id, chosen_id in enumerate(chosen_ids):
                variant_percents[chosen_id] = real_percents[go_id_id]
            #
            loglike_expression = self.traversome.model.get_like_formula(
                variant_percents=variant_percents,
                log_func=tt.log,
                within_variant_ids=set(chosen_ids)).loglike_expression
            pm.Potential("likelihood", loglike_expression)
            # pm.Deterministic("likelihood", likes)
            # pm.DensityDist?
            # pm.Mixture(name="likelihood", w=np.ones(len(variants)), comp_dists=variants, observed=data)
            # pm.Binomial("path_last", n=n__num_reads_in_range, p=this_prob, observed=x__num_matched_reads)
            # sample from the distribution

            # uses the BFGS optimization algorithm to find the maximum of the log-posterior
            logger.info("Searching the maximum of the log-posterior ..")
            start = pm.find_MAP(model=variants_model)
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
