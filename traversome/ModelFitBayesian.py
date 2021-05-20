#!/usr/bin/env python

from loguru import logger
import matplotlib.pyplot as plt
import pymc3 as pm
import theano.tensor as tt
import numpy as np


class ModelFitBayesian(object):
    """
    """
    def __init__(self, traversome_obj):
        self.traversome = traversome_obj
        # self.graph = traversome_obj.graph
        pass

    def run_mcmc(self, n_generations, n_burn):
        logger.info("{} subpaths in total".format(len(self.traversome.all_sub_paths)))
        isomer_num = self.traversome.num_of_isomers
        with pm.Model() as isomer_model:
            isomer_percents = pm.Dirichlet(name="props", a=np.ones(isomer_num), shape=(isomer_num,))
            loglike_expression = self.traversome.get_likelihood_formula(isomer_percents=isomer_percents, log_func=tt.log)
            pm.Potential("likelihood", loglike_expression)
            # pm.Deterministic("likelihood", likes)
            # pm.DensityDist?
            # pm.Mixture(name="likelihood", w=np.ones(len(components)), comp_dists=components, observed=data)
            # pm.Binomial("path_last", n=n__num_reads_in_range, p=this_prob, observed=x__num_matched_reads)
            # sample from the distribution
            start = pm.find_MAP(model=isomer_model)
            # trace = pm.sample_smc(n_generations, parallel=False)
            trace = pm.sample(
                n_generations, tune=n_burn, discard_tuned_samples=True, cores=1, init='adapt_diag', start=start)
            logger.info(pm.summary(trace))
        return trace