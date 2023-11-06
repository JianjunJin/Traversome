#!/usr/bin/env python
from loguru import logger
from scipy import optimize
from collections import OrderedDict
from traversome.utils import LogLikeFuncInfo, Criterion, aic, bic, run_dill_encoded
import numpy as np
import symengine
from multiprocessing import Manager, Pool
import sys
import traceback
# import pickle
import dill
from typing import OrderedDict as typingODict
from typing import Union
# from math import inf
np.seterr(divide="ignore", invalid="ignore")


def minimize_neg_likelihood(neg_loglike_func, num_variables, verbose, err_queue=None):
    try:
        # logger.info("   loading picked function ..")
        # if isinstance(neg_loglike_func, str):
        #     with open(neg_loglike_func, "rb") as input_handler:
        #         neg_loglike_func = pickle.load(input_handler)
        # logger.info("   searching for ml result ..")
        # all proportions should be in range [0, 1] and sum up to 1.
        constraints = ({"type": "eq", "fun": lambda x: sum(x) - 1})  # what if we relax this?
        other_optimization_options = {"disp": verbose, "maxiter": 1000, "ftol": 1.0e-6, "eps": 1.0e-10}
        count_run = 0
        success_runs = []
        while count_run < 10000:
            initials = np.random.random(num_variables)
            initials /= sum(initials)
            # logger.debug("initials", initials)
            # np.full(shape=num_put_variants, fill_value=float(1. / num_put_variants), dtype=np.float)
            result = optimize.minimize(
                fun=neg_loglike_func,
                x0=initials,
                jac=False, method='SLSQP', bounds=[(0., 1.0)] * num_variables, constraints=constraints,
                options=other_optimization_options)
            # bounds=[(-1.0e-9, 1.0)] * num_variants will violate bound constraints and cause ValueError
            if result.success:
                success_runs.append(result)
                if len(success_runs) > 5:
                    break
            count_run += 1
            # sys.stdout.write(str(count_run) + "\b" * len(str(count_run)))
            # sys.stdout.flush()
        # logger.info("   searching for ml result fnished.")
        if success_runs:
            return sorted(success_runs, key=lambda x: x.fun)[0]
        else:
            return False
    except Exception as e:
        if err_queue:
            err_queue.put(e)
        else:
            raise e


class ModelFitMaxLike(object):
    """
    Find the parameters (variant proportions) to maximize the likelihood
    """
    # def __init__(self, traversome_obj):
    def __init__(self,
                 model,
                 variant_paths,
                 variant_subpath_counters,
                 sbp_to_sbp_id,
                 repr_to_merged_variants,
                 be_unidentifiable_to,
                 loglevel):
        self.model = model
        self.variant_paths = variant_paths
        self.num_put_variants = len(variant_paths)
        self.all_sub_paths = model.all_sub_paths
        self.variant_subpath_counters = variant_subpath_counters
        self.sbp_to_sbp_id = sbp_to_sbp_id
        self.repr_to_merged_variants = repr_to_merged_variants
        self.be_unidentifiable_to = be_unidentifiable_to
        self.loglevel = loglevel

        # self.graph = traversome_obj.graph
        # self.variant_sizes = traversome_obj.variant_sizes
        # self.align_len_at_path_sorted = traversome_obj.align_len_at_path_sorted

        # to be generated
        self.variant_percents = None
        self.observed_sbp_id_set = set()

        # res without model selection
        self.pe_neg_loglike_function = None
        self.pe_best_proportions = None

    def point_estimate(self,
                       chosen_ids: set = None):
                       # chosen_ids_set: typingODict[int, bool] = None):
        # self.variant_percents = [sympy.Symbol("P" + str(variant_id)) for variant_id in range(self.num_put_variants)]
        if chosen_ids:
            # chosen_ids_set = OrderedDict([(self.be_unidentifiable_to[variant_id], True)
            #                           for variant_id in chosen_ids_set])
            chosen_ids = {self.be_unidentifiable_to[variant_id] for variant_id in chosen_ids}
        else:
            # chosen_ids_set = OrderedDict([(variant_id, True) for variant_id in self.repr_to_merged_variants])
            chosen_ids = {variant_id for variant_id in self.repr_to_merged_variants}
        # Because many traversome attributes including subpath information were created using the original variant
        # ids, so here we prefer not making traversome.get_multinomial_like_formula complicated. Instead, we create
        # variant_percents with foo values inserted when that variant id is not in chosen_ids_set.
        self.variant_percents = [symengine.Symbol("P" + str(variant_id)) if variant_id in chosen_ids else False
                                 for variant_id in range(self.num_put_variants)]
        logger.info("Generating the likelihood function .. ")
        self.pe_neg_loglike_function = self.get_neg_likelihood_of_var_freq(
            within_variant_ids=chosen_ids)\
            .loglike_func
        logger.info("Maximizing the likelihood function .. ")
        success_run = minimize_neg_likelihood(
            neg_loglike_func=self.pe_neg_loglike_function,
            num_variables=len(chosen_ids),
            verbose=self.loglevel in ("TRACE", "ALL"))
        # TODO: we added chosen_ids_set at 2022-11-15, the result may be need to be checked
        if success_run:
            # for run_res in sorted(success_runs, key=lambda x: x.fun):
            #     logger.info(str(run_res.fun) + str([round(m, 8) for m in run_res.x]))
            self.pe_best_proportions, echo_prop = self.__summarize_run_prop(success_run, within_var_ids=chosen_ids)
            logger.info("Proportions: " + ", ".join(["%s:%.4f" % (_id, _p) for _id, _p, in echo_prop.items()]))
            logger.info("Log-likelihood: %s" % (-success_run.fun))
            return self.pe_best_proportions
        else:
            raise Exception("Likelihood maximization failed.")

    def reverse_model_selection(self,
                                n_proc,
                                criterion=Criterion.AIC,
                                chosen_ids: typingODict[int, bool] = None,
                                random_size: int = 100,
                                user_fixed_ids: Union[list, tuple, set, None] = None):
        """
        :param n_proc: number of processes
        :param criterion:
        :param chosen_ids:
            Only apply reverse model selection on chosen ids.
            The value of the OrderedDict has no use. We actually just want an OrderedSet.
        :param random_size: [0, INT)
            Each time randomly subsampling X models from the (N-1)-dim models
            instead of testing all of them like.
            Choose 0 to disable this process.
            # TODO further info required
        :param user_fixed_ids:
            user fixed variant ids that will not be dropped during model selection.
        """
        # TODO: tolerance can be larger
        if random_size != 0 and n_proc > random_size:
            logger.warning("random size {} is smaller than the num of processes {}, which is limited by the former.")

        # diff_tolerance = 1e-9
        diff_tolerance = 1e-6
        if chosen_ids:
            # chosen_ids = OrderedDict([(self.be_unidentifiable_to[variant_id], True)
            #                           for variant_id in chosen_ids])
            chosen_ids = {self.be_unidentifiable_to[variant_id] for variant_id in chosen_ids}
        else:
            # chosen_ids = OrderedDict([(variant_id, True) for variant_id in self.repr_to_merged_variants])
            chosen_ids = {variant_id for variant_id in self.repr_to_merged_variants}
        chosen_ids = set(chosen_ids)
        # Because many traversome attributes including subpath information were created using the original variant
        # ids, so here we prefer not making traversome.get_multinomial_like_formula complicated. Instead, we create
        # variant_percents with foo values inserted when that variant id is not in chosen_ids_set.
        self.variant_percents = [symengine.Symbol("P" + str(variant_id)) if variant_id in chosen_ids else False
                                 for variant_id in range(self.num_put_variants)]
        logger.debug("Test variants {}".format(list(chosen_ids)))

        # if one component is identified as indispensable in the n-dimension model,
        # it will be indispensable for subsequent (n-m)-dimension models
        if user_fixed_ids:  # in the case of user assigned fixed 'indispensable' variant id(s)
            indispensable_ids = {u_id: True for u_id in user_fixed_ids}
        else:
            indispensable_ids = {}

        previous_prop, previous_echo, previous_like, previous_criteria = \
            self.__compute_like_and_criteria(chosen_id_set=chosen_ids, criteria=criterion)

        # logger.info("Proportions: %s " % {_iid: previous_prop[_gid] for _gid, _iid in enumerate(chosen_ids_set)})
        # logger.info("Log-likelihood: %s" % previous_like)
        # drop zero prob variant
        self.__drop_zero_variants(chosen_ids, previous_prop, previous_echo, diff_tolerance, indispensable_ids)

        # stepwise
        while len(chosen_ids) > 1:
            logger.debug("Trying dropping {} variant(s) ..".format(self.num_put_variants - len(chosen_ids) + 1))
            chosen_ids_sorted = sorted(chosen_ids)
            chosen_rd_list = list(range(len(chosen_ids_sorted)))
            np.random.shuffle(chosen_rd_list)
            changed = False
            # TODO, better enumerate
            rs = random_size if random_size > 0 else len(chosen_rd_list)
            for go_rd_sp in range(0, len(chosen_rd_list), rs):
                this_rd_ids = chosen_rd_list[go_rd_sp: go_rd_sp+rs]
                test_id_res = OrderedDict()
                if n_proc == 1:
                    for rd_id in this_rd_ids:
                        self.__test_one_drop(
                            var_id=chosen_ids_sorted[rd_id],
                            chosen_ids=chosen_ids,
                            sorted_chosen_ids=chosen_ids_sorted,
                            criterion=criterion,
                            test_id_res=test_id_res,
                            indispensable_ids=indispensable_ids)
                else:
                    # TODO
                    manager = Manager()
                    error_queue = manager.Queue()
                    event = manager.Event()
                    lock = manager.Lock()
                    global_vars = manager.Namespace()
                    global_vars.w_id = 0  # worker id
                    global_vars.recorded_ids = manager.list()
                    global_vars.prop = manager.list()
                    global_vars.echo = manager.list()
                    global_vars.loglike = manager.list()
                    global_vars.criterion = manager.list()
                    global_vars.finished_w = 0
                    global_vars.indispensable_ids = manager.dict()
                    global_vars.indispensable_ids.update(indispensable_ids)
                    logger.info("Serializing traversome for multiprocessing ..")
                    payload = dill.dumps((self.__test_one_drop_worker,
                                          (this_rd_ids, chosen_ids, criterion, global_vars, lock, event, error_queue)))
                    pool_obj = Pool(processes=n_proc)
                    job_list = []
                    for go_w in range(len(this_rd_ids)):
                        # TODO: to fix the issue that the behaviour of the logger in the worker become be different
                        #       sim.alignment.new.100k.300k.100x.traversome-bic-N1000-user-p
                        logger.debug("assigning job to worker {}".format(go_w + 1))
                        job_list.append(pool_obj.apply_async(run_dill_encoded, (payload,)))
                        logger.debug("assigned job to worker {}".format(go_w + 1))
                    pool_obj.close()
                    event.wait()
                    pool_obj.terminate()
                    while not error_queue.empty():
                        e, tb, location = error_queue.get()
                        logger.error("\n" + "".join(tb))  # + "\n" + str(location) + "\n" + str(e))
                        sys.exit(0)
                    # pool.join()
                    # use the global_vars.recorded_ids to collect the corresponding resultI am
                    for go_r, var_id in enumerate(list(global_vars.recorded_ids)):
                        test_id_res[var_id] = \
                            {"prop": global_vars.prop[go_r], "echo": global_vars.echo[go_r],
                             "loglike": global_vars.loglike[go_r], criterion: global_vars.criterion[go_r]}
                    indispensable_ids.update(dict(global_vars.indispensable_ids))
                if test_id_res:
                    best_drop_id, best_val = sorted([[_go_var_, test_id_res[_go_var_][criterion]]
                                                     for _go_var_ in test_id_res],
                                                    key=lambda x: x[1])[0]
                    if best_val < previous_criteria:
                        previous_criteria = best_val
                        previous_like = test_id_res[best_drop_id]["loglike"]
                        previous_prop = test_id_res[best_drop_id]["prop"]
                        previous_echo = test_id_res[best_drop_id]["echo"]
                        chosen_ids.remove(best_drop_id)
                        # drop candidate id that minimize criteria
                        logger.info("Drop {}".format(self.__str_rep_id(best_drop_id)))
                        logger.info("Proportions: " +
                                    ", ".join(["%s:%.4f" % (_id, _p) for _id, _p, in previous_echo.items()]))
                        logger.info("Log-likelihood: %s" % previous_like)
                        self.__drop_zero_variants(
                            chosen_ids, previous_prop, previous_echo, diff_tolerance, indispensable_ids)
                        changed = True
                        break
                    # for var_id in list(chosen_ids_set):
                    #     if abs(previous_prop[var_id] - 0.) < diff_tolerance:
                    #         del chosen_ids_set[var_id]
                    #         for cid_var_id in self.traversome.repr_to_merged_variants[var_id]:
                    #             del previous_prop[cid_var_id]
                    #         del previous_echo[self.__str_rep_id(var_id)]
                    #         logger.info("Drop {}".format(self.__str_rep_id(var_id)))
                # else:
                #     logger.info("Proportions: " +
                #                 ", ".join(["%s:%.4f" % (_id, _p) for _id, _p, in previous_echo.items()]))
                #     logger.info("Log-likelihood: %s" % previous_like)
                #     return previous_prop

            if not changed:
                logger.info("Proportions: " +
                            ", ".join(["%s:%.4f" % (_id, _p) for _id, _p, in previous_echo.items()]))
                logger.info("Log-likelihood: %s" % previous_like)
                return previous_prop
        logger.info("Proportions: " + ", ".join(["%s:%.4f" % (_id, _p) for _id, _p, in previous_echo.items()]))
        logger.info("Log-likelihood: %s" % previous_like)
        return previous_prop

    def __drop_zero_variants(self, chosen_ids, representative_props, echo_props, diff_tolerance, indispensable_ids):
        for var_id in list(chosen_ids):
            # do not drop a candidate variant if it is fixed either by the user or by a read path
            if var_id in indispensable_ids:
                continue
            if abs(representative_props[var_id] - 0.) < diff_tolerance:
                chosen_ids.remove(var_id)
                for cid_var_id in self.repr_to_merged_variants[var_id]:
                    del representative_props[cid_var_id]
                del echo_props[self.__str_rep_id(var_id)]
                # drop candidate id that has estimated proportion of zero
                logger.info("Drop {}".format(self.__str_rep_id(var_id)))

    def __test_one_drop(self, var_id, chosen_ids: set, sorted_chosen_ids, criterion, test_id_res, indispensable_ids):
        if var_id not in indispensable_ids:
            testing_ids = chosen_ids - {var_id}
            if self.cover_all_observed_subpaths(testing_ids):
                logger.debug(
                    "Test variants [{}] - {}".
                        format(", ".join([self.__str_rep_id(_c_i)
                                          for _c_i in sorted_chosen_ids]), self.__str_rep_id(var_id)))
                res_list = self.__compute_like_and_criteria(chosen_id_set=testing_ids, criteria=criterion)
                test_id_res[var_id] = \
                    {"prop": res_list[0], "echo": res_list[1], "loglike": res_list[2], criterion: res_list[3]}
            else:
                indispensable_ids[var_id] = True
                logger.debug(
                    "Test variants [{}] - {}: skipped for necessary subpath(s) (case *)"
                        .format(", ".join([self.__str_rep_id(_c_i)
                                           for _c_i in sorted_chosen_ids]), self.__str_rep_id(var_id)))
        else:
            logger.debug(
                "Test variants [{}] - {}: skipped for necessary subpath(s) (case **)"
                .format(", ".join([self.__str_rep_id(_c_i)
                                   for _c_i in sorted_chosen_ids]), self.__str_rep_id(var_id)))

    def __test_one_drop_worker(
            self,
            this_rd_ids,
            chosen_ids_set: set,
            criterion,
            g_vars,
            lock,
            event,
            error_queue):
        try:
            lock.acquire()
            w_id = g_vars.w_id
            g_vars.w_id += 1
            lock.release()
            sorted_chosen_ids = sorted(chosen_ids_set)
            # use the worker id to pick a unique rd_id for each job,
            # then use the unique rd_id to randomly pick a var_id from the sorted_chosen_ids
            var_id = sorted_chosen_ids[this_rd_ids[w_id]]
            if var_id not in g_vars.indispensable_ids:
                testing_ids = chosen_ids_set - {var_id}
                if self.cover_all_observed_subpaths(testing_ids):
                    logger.debug(
                        "Test variants [{}] - {}".
                            format(", ".join([self.__str_rep_id(_c_i)
                                              for _c_i in sorted_chosen_ids]), self.__str_rep_id(var_id)))
                    res_list = self.__compute_like_and_criteria(chosen_id_set=testing_ids, criteria=criterion)
                    lock.acquire()
                    g_vars.recorded_ids.append(var_id)
                    g_vars.prop.append(res_list[0])
                    g_vars.echo.append(res_list[1])
                    g_vars.loglike.append(res_list[2])
                    g_vars.criterion.append(res_list[3])
                    lock.release()
                    # logger.debug("Generating the likelihood function .. ")
                    # neg_loglike_func_obj = self.get_neg_likelihood_of_var_freq(within_variant_ids=testing_ids)
                    # logger.info("Maximizing the likelihood function for {} variants".format(len(testing_ids)))
                    # # with open("/tmp/traversome." + str(var_id), "wb") as output_h:
                    # #     pickle.dump(neg_loglike_func_obj.loglike_func, output_h)
                    # minimize_neg_likelihood,
                    #                                  (neg_loglike_func_obj,
                    #                                   len(testing_ids),
                    #                                   self.traversome.loglevel in ("TRACE", "ALL"),
                    #                                   # error_queue
                    #                                   )))
                    # job_var_ids.append((testing_ids, neg_loglike_func_obj, var_id))
                    # # # TypeError("cannot pickle 'module' object")
                    # # job_list.append(pool.apply_async(self.__compute_like_and_criteria, (testing_ids, criterion)))
                    # # job_var_ids.append(var_id)
                else:
                    lock.acquire()
                    g_vars.indispensable_ids[var_id] = True
                    lock.release()
                    logger.debug(
                        "Test variants [{}] - {}: skipped for necessary subpath(s) (case *)"
                            .format(", ".join([self.__str_rep_id(_c_i)
                                               for _c_i in sorted_chosen_ids]), self.__str_rep_id(var_id)))
            else:
                logger.debug(
                    "Test variants [{}] - {}: skipped for necessary subpath(s) (case **)"
                    .format(", ".join([self.__str_rep_id(_c_i)
                                       for _c_i in sorted_chosen_ids]), self.__str_rep_id(var_id)))
        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            tb = traceback.format_exception(exc_type, exc_value, exc_traceback)
            location = traceback.extract_tb(exc_traceback)[-1]
            error_queue.put((e, tb, location))
            event.set()
            return
        lock.acquire()
        g_vars.finished_w += 1
        lock.release()
        # sent terminal signal if the last var_id was finished
        # logger.debug("chosen_ids_set:g_vars.finished_w={}:{}".format(len(this_rd_ids), g_vars.finished_w))
        if len(this_rd_ids) == g_vars.finished_w:
            event.set()

    def __compute_like_and_criteria(self, chosen_id_set, criteria):
        logger.debug("Generating the likelihood function .. ")
        neg_loglike_func_obj = self.get_neg_likelihood_of_var_freq(within_variant_ids=chosen_id_set)
        logger.info("Maximizing the likelihood function for {} variants".format(len(chosen_id_set)))
        success_run = minimize_neg_likelihood(
            neg_loglike_func=neg_loglike_func_obj.loglike_func,
            num_variables=len(chosen_id_set),
            verbose=self.loglevel in ("TRACE", "ALL"))
        return self.__summarize_like_and_criteria(success_run, chosen_id_set, criteria, neg_loglike_func_obj)

    # def __compute_like_and_criteria_1(self, chosen_id_set):
    #     logger.debug("Generating the likelihood function .. ")
    #     neg_loglike_func_obj = self.get_neg_likelihood_of_var_freq(within_variant_ids=chosen_id_set)
    #
    # def __compute_like_and_criteria_2(self, chosen_id_set, neg_loglike_func_obj):
    #     logger.info("Maximizing the likelihood function for {} variants".format(len(chosen_id_set)))
    #     success_run = minimize_neg_likelihood(
    #         neg_loglike_func=neg_loglike_func_obj.loglike_func,
    #         num_variables=len(chosen_id_set),
    #         verbose=self.traversome.loglevel in ("TRACE", "ALL"))

    def __summarize_like_and_criteria(self, success_run, chosen_id_set, criteria, neg_loglike_func_obj):
        if success_run:
            # this_prop = list(success_run.x)
            this_like = -success_run.fun
            use_prop, echo_prop = self.__summarize_run_prop(success_run, chosen_id_set)
            logger.debug("Proportions: " + ", ".join(["%s:%.4f" % (_id, _p) for _id, _p, in echo_prop.items()]))
            logger.debug("Log-likelihood: %s" % this_like)
            if criteria == "aic":
                logger.debug("len_param: %s" % neg_loglike_func_obj.variable_size)
                this_criteria = aic(
                    loglike=this_like,
                    len_param=neg_loglike_func_obj.variable_size)
                logger.debug("%s: %s" % (criteria, this_criteria))
            elif criteria == "bic":
                logger.debug("len_param: %s" % neg_loglike_func_obj.variable_size)
                logger.debug("len_data: %s" % neg_loglike_func_obj.sample_size)
                this_criteria = bic(
                    loglike=this_like,
                    len_param=neg_loglike_func_obj.variable_size,
                    len_data=neg_loglike_func_obj.sample_size)
                logger.debug("%s: %s" % (criteria, this_criteria))
            else:
                raise Exception("Invalid criterion {}".format(criteria))
            return use_prop, echo_prop, this_like, this_criteria
        else:
            raise Exception("Likelihood maximization failed.")

    def __summarize_run_prop(self, success_run, within_var_ids):
        prop_dict = {}
        representatives = [rep_id for rep_id in sorted(within_var_ids) if rep_id in self.repr_to_merged_variants]
        echo_prop = OrderedDict()
        for go, this_prop in enumerate(success_run.x):
            echo_prop[self.__str_rep_id(representatives[go])] = this_prop
            unidentifiable_var_ids = self.repr_to_merged_variants[representatives[go]]
            this_prop /= len(unidentifiable_var_ids)
            for cid_var_id in unidentifiable_var_ids:
                prop_dict[cid_var_id] = this_prop
        use_prop = OrderedDict([(_id, prop_dict[_id]) for _id in sorted(prop_dict)])
        return use_prop, echo_prop

    def __str_rep_id(self, rep_id):
        return "+".join([f"cid_{(_cid_var_id + 1)}" for _cid_var_id in self.repr_to_merged_variants[rep_id]])

    def get_neg_likelihood_of_var_freq(self, within_variant_ids: set = None, scipy_style=True):
        # log_like_formula = self.traversome.get_likelihood_binomial_formula(
        #     self.variant_percents,
        #     log_func=sympy.log,
        #     within_variant_ids=within_variant_ids)
        log_like_formula = self.model.get_like_formula(
            self.variant_percents,
            # log_func=sympy.log,
            log_func=symengine.log,
            within_variant_ids=within_variant_ids)
        if within_variant_ids is None:
            within_variant_ids = set(range(self.num_put_variants))
        logger.trace("Formula: {}".format(-log_like_formula.loglike_expression))
        if scipy_style:
            # for compatibility between scipy and sympy
            # positional arguments -> single tuple argument
            # def neg_likelihood_of_variant_freq_single_arg(x):
            #     return neg_likelihood_of_var_freq(*tuple(x))
            # neg_likelihood_of_var_freq = sympy.lambdify(
            neg_likelihood_of_variant_freq_single_arg = symengine.lambdify(
                args=tuple([self.variant_percents[variant_id]
                            for variant_id in range(self.num_put_variants) if variant_id in within_variant_ids]),
                # expr=-log_like_formula.loglike_expression)
                exprs=[-log_like_formula.loglike_expression],
                backend="llvm"  # see https://github.com/symengine/symengine.py/issues/294 for why using "llvm"
                )

            return LogLikeFuncInfo(
                loglike_func=neg_likelihood_of_variant_freq_single_arg,
                variable_size=log_like_formula.variable_size,
                sample_size=log_like_formula.sample_size)
        else:
            # neg_likelihood_of_var_freq = sympy.lambdify(
            neg_likelihood_of_var_freq = symengine.lambdify(
                args=[self.variant_percents[variant_id]
                      for variant_id in range(self.num_put_variants) if variant_id in within_variant_ids],
                # expr=-log_like_formula.loglike_expression)
                exprs=[-log_like_formula.loglike_expression],
                backend="llvm"  # see https://github.com/symengine/symengine.py/issues/294 for why using "llvm"
                )
            return LogLikeFuncInfo(
                loglike_func=neg_likelihood_of_var_freq,
                variable_size=log_like_formula.variable_size,
                sample_size=log_like_formula.sample_size)

    def update_observed_sp_ids(self):
        self.observed_sbp_id_set = set()
        for go_sp, (this_sub_path, this_sub_path_info) in enumerate(self.all_sub_paths.items()):
            if this_sub_path_info.mapped_records:
                self.observed_sbp_id_set.add(go_sp)
            else:
                logger.trace("Drop subpath without observation: {}: {}".format(go_sp, this_sub_path))

    def cover_all_observed_subpaths(self, variant_ids):
        if not self.observed_sbp_id_set:
            self.update_observed_sp_ids()
        model_sp_ids = set()
        for go_var in variant_ids:
            for sub_path in self.variant_subpath_counters[self.variant_paths[go_var]]:
                if sub_path in self.sbp_to_sbp_id:
                    # if sub_path was not dropped after the construction of self.variant_subpath_counters
                    model_sp_ids.add(self.sbp_to_sbp_id[sub_path])
        if self.observed_sbp_id_set.issubset(model_sp_ids):
            return True
        else:
            return False


