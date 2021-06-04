#!/usr/bin/env python

from loguru import logger
import pymc3 as pm
import theano.tensor as tt
import numpy as np
import arviz as az


class ModelFitBayesian(object):
    """
    """
    def __init__(self, traversome_obj):
        self.traversome = traversome_obj
        self.trace = None
        # self.graph = traversome_obj.graph
        pass

    def run_mcmc(self, n_generations, n_burn):
        logger.info("{} subpaths in total".format(len(self.traversome.all_sub_paths)))
        isomer_num = self.traversome.num_of_isomers
        with pm.Model() as isomer_model:
            isomer_percents = pm.Dirichlet(name="props", a=np.ones(isomer_num), shape=(isomer_num,))
            loglike_expression = self.traversome.get_likelihood_binormial_formula(
                isomer_percents=isomer_percents, log_func=tt.log).loglike_expression
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
        return summary["mean"]
