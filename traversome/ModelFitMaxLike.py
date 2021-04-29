#!/usr/bin/env python
from loguru import logger
from traversome.utils import get_id_range_in_increasing_values
from scipy import optimize
import numpy as np
import sympy
np.seterr(divide="ignore", invalid="ignore")



class ModelFitMaxLike(object):
    """
    Find the parameters (isomer proportions) to maximize the likelihood
    """
    def __init__(self, traversome_obj):
        self.traversome = traversome_obj
        self.num_of_isomers = traversome_obj.num_of_isomers
        self.all_sub_paths = traversome_obj.all_sub_paths
        # self.graph = traversome_obj.graph
        self.isomer_sizes = traversome_obj.isomer_sizes
        self.align_len_at_path_sorted = traversome_obj.align_len_at_path_sorted

        # to be generated
        self.isomer_percents = None
        self.neg_loglike_function = None
        self.best_proportions = []

    def run(self):
        self.isomer_percents = [sympy.Symbol("P" + str(isomer_id)) for isomer_id in range(self.num_of_isomers)]
        logger.info("Generating the likelihood function .. ")
        self.get_neg_likelihood_of_iso_freq()
        logger.info("Maximizing the likelihood function .. ")
        success_runs = self.minimize_neg_likelihood(verbose=self.traversome.loglevel in ("DEBUG", "TRACE", "ALL"))
        if success_runs:
            # for run_res in sorted(success_runs, key=lambda x: x.fun):
            #     logger.info(str(run_res.fun) + str([round(m, 8) for m in run_res.x]))
            logger.info("Proportion: %s Log-likelihood: %s" % (success_runs[0].x, -success_runs[0].fun))
            self.best_proportions = success_runs[0].x
            return self.best_proportions

    def get_neg_likelihood_of_iso_freq(self, scipy_style=True):
        loglike_expression = self.traversome.get_likelihood_formula(self.isomer_percents, log_func=sympy.log)
        # print(maximum_loglike_expression)
        neg_likelihood_of_iso_freq = sympy.lambdify(
            args=[self.isomer_percents[isomer_id] for isomer_id in
                  range(len(self.isomer_percents))],
            expr=-loglike_expression)
        if scipy_style:
            # for compatibility between scipy and sympy
            # positional arguments -> single tuple argument
            def neg_likelihood_of_iso_freq_single_arg(x):
                return neg_likelihood_of_iso_freq(*tuple(x))

            return neg_likelihood_of_iso_freq_single_arg
        else:
            return neg_likelihood_of_iso_freq

    def minimize_neg_likelihood(self, verbose):
        # all proportions should be in range [0, 1] and sum up to 1.
        constraints = ({"type": "eq", "fun": lambda x: sum(x) - 1})
        other_optimization_options = {"disp": verbose, "maxiter": 1000, "ftol": 1.0e-6, "eps": 1.0e-10}
        count_run = 0
        success_runs = []
        while count_run < 100:
            initials = np.random.random(self.num_of_isomers)
            initials /= sum(initials)
            # print("initials", initials)
            # np.full(shape=num_of_isomers, fill_value=float(1. / num_of_isomers), dtype=np.float)
            result = optimize.minimize(
                fun=self.neg_loglike_function,
                x0=initials,
                jac=False, method='SLSQP', constraints=constraints, bounds=[(0., 1.0)] * self.num_of_isomers,
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
