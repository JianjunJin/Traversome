#!/usr/bin/env python
from loguru import logger
# from traversome.utils import get_id_range_in_increasing_values
from scipy import optimize
from collections import OrderedDict
from traversome.utils import LogLikeFuncInfo, Criteria, aic, bic
import numpy as np
# import sympy
import symengine
# from math import inf
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

        # res without model selection
        self.pe_neg_loglike_function = None
        self.pe_best_proportions = None

    def point_estimate(self):
        # self.isomer_percents = [sympy.Symbol("P" + str(isomer_id)) for isomer_id in range(self.num_of_isomers)]
        self.isomer_percents = [symengine.Symbol("P" + str(isomer_id)) for isomer_id in range(self.num_of_isomers)]
        logger.info("Generating the likelihood function .. ")
        self.pe_neg_loglike_function = self.get_neg_likelihood_of_iso_freq(
            within_isomer_ids=set(self.traversome.represent_for_isomers))\
            .loglike_func
        logger.info("Maximizing the likelihood function .. ")
        success_run = self.minimize_neg_likelihood(
            neg_loglike_func=self.pe_neg_loglike_function,
            num_variables=self.num_of_isomers,
            verbose=self.traversome.loglevel in ("TRACE", "ALL"))
        if success_run:
            # for run_res in sorted(success_runs, key=lambda x: x.fun):
            #     logger.info(str(run_res.fun) + str([round(m, 8) for m in run_res.x]))
            self.pe_best_proportions, echo_prop = self.__summarize_run_prop(success_run)
            logger.info("Proportions: " + ", ".join(["%s:%.4f" % (_id, _p) for _id, _p, in echo_prop.items()]))
            logger.info("Log-likelihood: %s" % (-success_run.fun))
            return self.pe_best_proportions
        else:
            raise Exception("Likelihood maximization failed.")

    def reverse_model_selection(self, criteria=Criteria.AIC):
        # TODO be aware of unidentifiable situations
        diff_tolerance = 1e-9
        # self.isomer_percents = [sympy.Symbol("P" + str(isomer_id)) for isomer_id in range(self.num_of_isomers)]
        self.isomer_percents = [symengine.Symbol("P" + str(isomer_id)) for isomer_id in range(self.num_of_isomers)]
        chosen_ids = OrderedDict([(isomer_id, True) for isomer_id in self.traversome.represent_for_isomers])
        logger.debug("Test components {}".format(list(chosen_ids)))
        previous_prop, previous_echo, previous_like, previous_criteria = \
            self.__compute_like_and_criteria(chosen_id_set=set(chosen_ids), criteria=criteria)
        # logger.info("Proportions: %s " % {_iid: previous_prop[_gid] for _gid, _iid in enumerate(chosen_ids)})
        # logger.info("Log-likelihood: %s" % previous_like)
        # drop zero prob component
        self.__drop_zero_components(chosen_ids, previous_prop, previous_echo, diff_tolerance)
        # stepwise
        while len(chosen_ids) > 1:
            logger.info("Trying dropping {} component(s) ..".format(self.num_of_isomers - len(chosen_ids) + 1))
            test_id_res = OrderedDict()
            # maybe do this latter, accept both if two models are unidentifiable
            # chosen_this_round = set()
            for iso_id in chosen_ids:
                testing_ids = set(chosen_ids) - {iso_id}
                if self.traversome.cover_all_observed_sp(testing_ids):
                    logger.debug(
                        "Test components [{}] - {}".
                        format(", ".join([self.__str_rep_id(_c_i) for _c_i in chosen_ids]), self.__str_rep_id(iso_id)))
                    res_list = self.__compute_like_and_criteria(chosen_id_set=testing_ids, criteria=criteria)
                    test_id_res[iso_id] = \
                        {"prop": res_list[0], "echo": res_list[1], "loglike": res_list[2], criteria: res_list[3]}
                else:
                    logger.debug(
                        "Test components [{}] - {}: skipped for necessary subpath(s)"
                        .format(", ".join([self.__str_rep_id(_c_i) for _c_i in chosen_ids]), self.__str_rep_id(iso_id)))
            if test_id_res:
                best_drop_id, best_val = sorted([[_go_iso_, test_id_res[_go_iso_][criteria]]
                                                 for _go_iso_ in test_id_res],
                                                key=lambda x: x[1])[0]
                if best_val < previous_criteria:
                    previous_criteria = best_val
                    previous_like = test_id_res[best_drop_id]["loglike"]
                    previous_prop = test_id_res[best_drop_id]["prop"]
                    previous_echo = test_id_res[best_drop_id]["echo"]
                    del chosen_ids[best_drop_id]
                    logger.info("Drop {}".format(self.__str_rep_id(best_drop_id)))
                    logger.info("Proportions: " + ", ".join(["%s:%.4f" % (_id, _p) for _id, _p, in previous_echo.items()]))
                    logger.info("Log-likelihood: %s" % previous_like)
                    self.__drop_zero_components(chosen_ids, previous_prop, previous_echo, diff_tolerance)
                    # for iso_id in list(chosen_ids):
                    #     if abs(previous_prop[iso_id] - 0.) < diff_tolerance:
                    #         del chosen_ids[iso_id]
                    #         for uid_iso_id in self.traversome.represent_for_isomers[iso_id]:
                    #             del previous_prop[uid_iso_id]
                    #         del previous_echo[self.__str_rep_id(iso_id)]
                    #         logger.info("Drop {}".format(self.__str_rep_id(iso_id)))
                else:
                    logger.info("Proportions: " + ", ".join(["%s:%.4f" % (_id, _p) for _id, _p, in previous_echo.items()]))
                    logger.info("Log-likelihood: %s" % previous_like)
                    return previous_prop
            else:
                logger.info("Proportions: " + ", ".join(["%s:%.4f" % (_id, _p) for _id, _p, in previous_echo.items()]))
                logger.info("Log-likelihood: %s" % previous_like)
                return previous_prop
        logger.info("Proportions: " + ", ".join(["%s:%.4f" % (_id, _p) for _id, _p, in previous_echo.items()]))
        logger.info("Log-likelihood: %s" % previous_like)
        return previous_prop

    def __drop_zero_components(self, chosen_ids, representative_props, echo_props, diff_tolerance):
        for iso_id in list(chosen_ids):
            if abs(representative_props[iso_id] - 0.) < diff_tolerance:
                del chosen_ids[iso_id]
                for uid_iso_id in self.traversome.represent_for_isomers[iso_id]:
                    del representative_props[uid_iso_id]
                del echo_props[self.__str_rep_id(iso_id)]
                logger.info("Drop {}".format(self.__str_rep_id(iso_id)))

    def __compute_like_and_criteria(self, chosen_id_set, criteria):
        logger.debug("Generating the likelihood function .. ")
        neg_loglike_func_obj = self.get_neg_likelihood_of_iso_freq(
            within_isomer_ids=chosen_id_set)
        logger.info("Maximizing the likelihood function for {} components".format(len(chosen_id_set)))
        success_run = self.minimize_neg_likelihood(
            neg_loglike_func=neg_loglike_func_obj.loglike_func,
            num_variables=len(chosen_id_set),
            verbose=self.traversome.loglevel in ("TRACE", "ALL"))
        if success_run:
            # this_prop = list(success_run.x)
            this_like = -success_run.fun
            use_prop, echo_prop = self.__summarize_run_prop(success_run, chosen_id_set)
            logger.debug("Proportions: " + ", ".join(["%s:%.4f" % (_id, _p) for _id, _p, in echo_prop.items()]))
            logger.debug("Log-likelihood: %s" % this_like)
            if criteria == "aic":
                this_criteria = aic(
                    loglike=this_like,
                    len_param=neg_loglike_func_obj.variable_size)
                logger.debug("%s: %s" % (criteria, this_criteria))
            elif criteria == "bic":
                this_criteria = bic(
                    loglike=this_like,
                    len_param=neg_loglike_func_obj.variable_size,
                    len_data=neg_loglike_func_obj.sample_size)
                logger.debug("%s: %s" % (criteria, this_criteria))
            else:
                raise Exception("Invalid criteria {}".format(criteria))
            return use_prop, echo_prop, this_like, this_criteria
        else:
            raise Exception("Likelihood maximization failed.")

    def __summarize_run_prop(self, success_run, within_iso_ids=None):
        prop_dict = {}
        if within_iso_ids:
            representatives = [rep_id for rep_id in self.traversome.represent_for_isomers if rep_id in within_iso_ids]
        else:
            representatives = list(self.traversome.represent_for_isomers)
        echo_prop = OrderedDict()
        for go, this_prop in enumerate(success_run.x):
            echo_prop[self.__str_rep_id(representatives[go])] = this_prop
            unidentifiable_iso_ids = self.traversome.represent_for_isomers[representatives[go]]
            this_prop /= len(unidentifiable_iso_ids)
            for uid_iso_id in unidentifiable_iso_ids:
                prop_dict[uid_iso_id] = this_prop
        use_prop = OrderedDict([(_id, prop_dict[_id]) for _id in sorted(prop_dict)])
        return use_prop, echo_prop

    def __str_rep_id(self, rep_id):
        return "+".join([str(_uid_iso_id) for _uid_iso_id in self.traversome.represent_for_isomers[rep_id]])

    def get_neg_likelihood_of_iso_freq(self, within_isomer_ids=None, scipy_style=True):
        # log_like_formula = self.traversome.get_likelihood_binomial_formula(
        #     self.isomer_percents,
        #     log_func=sympy.log,
        #     within_isomer_ids=within_isomer_ids)
        log_like_formula = self.traversome.get_multinomial_like_formula(
            self.isomer_percents,
            # log_func=sympy.log,
            log_func=symengine.log,
            within_isomer_ids=within_isomer_ids)
        if within_isomer_ids is None:
            within_isomer_ids = set(range(self.num_of_isomers))
        # neg_likelihood_of_iso_freq = sympy.lambdify(
        neg_likelihood_of_iso_freq = symengine.lambdify(
            args=[self.isomer_percents[isomer_id]
                  for isomer_id in range(self.num_of_isomers) if isomer_id in within_isomer_ids],
            # expr=-log_like_formula.loglike_expression)
            exprs=[-log_like_formula.loglike_expression])
        logger.trace("Formula: {}".format(-log_like_formula.loglike_expression))
        if scipy_style:
            # for compatibility between scipy and sympy
            # positional arguments -> single tuple argument
            def neg_likelihood_of_iso_freq_single_arg(x):
                return neg_likelihood_of_iso_freq(*tuple(x))

            return LogLikeFuncInfo(
                loglike_func=neg_likelihood_of_iso_freq_single_arg,
                variable_size=log_like_formula.variable_size,
                sample_size=log_like_formula.sample_size)
        else:
            return LogLikeFuncInfo(
                loglike_func=neg_likelihood_of_iso_freq,
                variable_size=log_like_formula.variable_size,
                sample_size=log_like_formula.sample_size)

    def minimize_neg_likelihood(self, neg_loglike_func, num_variables, verbose):
        # all proportions should be in range [0, 1] and sum up to 1.
        constraints = ({"type": "eq", "fun": lambda x: sum(x) - 1})  # what if we relax this?
        other_optimization_options = {"disp": verbose, "maxiter": 1000, "ftol": 1.0e-6, "eps": 1.0e-10}
        count_run = 0
        success_runs = []
        while count_run < 10000:
            initials = np.random.random(num_variables)
            initials /= sum(initials)
            # logger.debug("initials", initials)
            # np.full(shape=num_of_isomers, fill_value=float(1. / num_of_isomers), dtype=np.float)
            result = optimize.minimize(
                fun=neg_loglike_func,
                x0=initials,
                jac=False, method='SLSQP', bounds=[(0., 1.0)] * num_variables, constraints=constraints,
                options=other_optimization_options)
            # bounds=[(-1.0e-9, 1.0)] * num_isomers will violate bound constraints and cause ValueError
            if result.success:
                success_runs.append(result)
                if len(success_runs) > 5:
                    break
            count_run += 1
            # sys.stdout.write(str(count_run) + "\b" * len(str(count_run)))
            # sys.stdout.flush()
        if success_runs:
            return sorted(success_runs, key=lambda x: x.fun)[0]
        else:
            return False

