#!/usr/bin/env python
import math

from loguru import logger
from scipy import optimize
from collections import OrderedDict
from traversome.utils import (
    LogLikeFuncInfo, Criterion, aic, bic, run_dill_encoded, get_randint_by_exp_weights, setup_logger)
import numpy as np
import symengine
from multiprocessing import Manager, Pool
import threading
import sys
import traceback
import time
# import pickle
import dill
from typing import OrderedDict as typingODict
from typing import Union, Set, List, Tuple
import atexit
from bisect import bisect_left
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
        other_optimization_options = {"disp": verbose, "maxiter": 1000, "ftol": 1.0e-5, "eps": 1.0e-8}
        count_run = 0
        success_runs = []
        while count_run < 10000:
            # use dual annealing to find a good initial point
            # initials = np.random.random(num_variables)
            # initials /= sum(initials)
            # try dual_annealing several times to avoid bound ValueError # 2025-07-23
            for try_dual_annealing in range(3):
                try:
                    global_res = optimize.dual_annealing(
                        neg_loglike_func,
                        bounds=[(0., 1.0)] * num_variables,
                        maxiter=1000).x  # dual_annealing does not take constraints
                except ValueError as e:
                    if "violates bound constraints" in str(e):
                        continue
                    else:
                        raise e
                finally:
                    break
            initials = global_res / sum(global_res)
            # logger.debug("initials", initials)
            # TODO: provide the Jacobian of the objective function may help speed up the optimization and potentially find better solutions
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
            logger.trace(f"Found {len(success_runs)} successful runs with loglikes {[x.fun for x in success_runs]}")
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
                 variant_readpath_counters,
                 sbp_to_sbp_id,
                 repr_to_merged_variants,
                 be_unidentifiable_to,
                 loglevel="INFO",
                 logfile=None,
                 bootstrap_mode=False):
        self.model = model
        self.variant_paths = variant_paths
        self.num_put_variants = len(variant_paths)
        self.all_sub_paths = model.all_sub_paths
        self.variant_readpath_counters = variant_readpath_counters
        self.sbp_to_sbp_id = sbp_to_sbp_id
        self.repr_to_merged_variants = repr_to_merged_variants
        self.be_unidentifiable_to = be_unidentifiable_to
        self.loglevel = loglevel
        self.logfile = logfile
        self.bootstrap_mode = bootstrap_mode
        # self.graph = traversome_obj.graph
        # self.variant_sizes = traversome_obj.variant_sizes
        # self.align_len_at_path_sorted = traversome_obj.align_len_at_path_sorted

        # to be generated
        self.variant_percents = None
        self.observed_sbp_id_set = set()

        # res without model selection
        self.pe_neg_loglike_obj = None
        self.pe_best_proportions = None

        # 
        self.__warning_sent = False

    def point_estimate(self,
                       chosen_ids: set = None,
                       criterion=Criterion.BIC):
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
        if self.bootstrap_mode:
            logger.debug("Generating the likelihood function .. ")
        else:
            logger.info("Generating the likelihood function .. ")
        self.pe_neg_loglike_obj = self.get_neg_likelihood_of_var_freq(
            within_variant_ids=chosen_ids)
        if self.bootstrap_mode:
            logger.debug("Maximizing the likelihood function .. ")
        else:
            logger.info("Maximizing the likelihood function .. ")
        success_run = minimize_neg_likelihood(
            neg_loglike_func=self.pe_neg_loglike_obj.loglike_func,
            num_variables=len(chosen_ids),
            verbose=self.loglevel in ("TRACE", "ALL"))
        # TODO: we added chosen_ids_set at 2022-11-15, the result may be need to be checked
        if success_run:
            use_prop, echo_prop, this_like, this_criteria = \
                self.__summarize_like_and_criteria(success_run, sorted(chosen_ids), criterion, self.pe_neg_loglike_obj)
            # self.pe_best_proportions, echo_prop = self.__summarize_run_prop(success_run, within_var_ids=chosen_ids)
            # logger.info("Proportions: " + ", ".join(["%s:%.4f" % (_id, _p) for _id, _p, in echo_prop.items()]))
            # logger.info("Log-likelihood: %s" % (-success_run.fun))
            return use_prop, echo_prop, this_like, this_criteria
        else:
            raise Exception("Likelihood maximization failed.")

    def genetic_algorithm_search(self,
                                 n_proc,
                                 criterion=Criterion.AIC,
                                 chosen_ids: Union[typingODict[int, bool], Set] = None,
                                 user_fixed_ids: Union[list, tuple, set, None] = None,
                                 equal_tol=1e-5,
                                 population_size=300,
                                 max_generations=200,
                                 crossover_prob=0.8,
                                 mutation_prob=0.05,
                                 tournament_size=3,
                                 num_elites=2,
                                 patience=5):
        """
        :param criterion:
        :param chosen_ids:
            Only apply reverse model selection on chosen ids.
        :param user_fixed_ids:
            user fixed variant ids that will not be dropped during model selection.
        :param equal_tol:
            The tolerance for determining equivalence.
        :param population_size:
            The number of individuals in the population.
        :param max_generations:
            The maximum number of generations to run the algorithm.
        :param crossover_prob:
            The probability of crossover between two parents.
        :param mutation_prob:
            The probability of mutation for each locus.
        :param tournament_size:
            The number of individuals to compete in the tournament. Should be less than population_size and an even number.
        :param num_elites:
            The number of elite individuals to keep.
        :param patience:
            The number of generations to wait before stopping the algorithm if no improvement is found.
        """
        if chosen_ids:
            # chosen_ids = OrderedDict([(self.be_unidentifiable_to[variant_id], True)
            #                           for variant_id in chosen_ids])
            chosen_ids = {self.be_unidentifiable_to[variant_id] for variant_id in chosen_ids}
        else:
            # chosen_ids = OrderedDict([(variant_id, True) for variant_id in self.repr_to_merged_variants])
            chosen_ids = {variant_id for variant_id in self.repr_to_merged_variants}
        chosen_ids = set(chosen_ids)
        self.variant_percents = [symengine.Symbol("P" + str(variant_id)) if variant_id in chosen_ids else False
                                 for variant_id in range(self.num_put_variants)]
        sorted_chosen_ids = np.array(sorted(chosen_ids))  # use array so that we can filter it faster later
        if user_fixed_ids:
            # find idx for user_fixed_ids in sorted_chosen_ids, then use it (fixed_mask) to turn all individuals true at the fixed idx
            fixed_mask = np.isin(sorted_chosen_ids, user_fixed_ids)  # boolean mask for fixed ids
        else:
            fixed_mask = np.zeros(len(sorted_chosen_ids), dtype=np.bool_)  # all individuals are fixed
        
        self.update_observed_sp_ids()  # update the observed subpath ids so that self.observed_sbp_id_set does not to be updated every time
        # 1. data representation & population initialization
        # code the population as binary strings
        # only code the indices of the sorted chosen_ids
        cache_cover = {}
        population = self.__ga_initialize_population(population_size, sorted_chosen_ids, cache_cover=cache_cover)
        population |= fixed_mask
        logger.info(f"Initilized population size: {len(population)}")
        
        # 2. initialize values
        no_improvement = 0
        best_criterion_val = float("inf")
        best_models = {}  # only store the best models

        if n_proc == 1:
            cache_estimated = {}  # cache the estimated results
        else:
            manager = Manager()
            error_queue = manager.Queue()
            event = manager.Event()
            lock = manager.Lock()
            global_vars = manager.Namespace()
            global_vars.cache_estimated = manager.dict()

        for generation in range(max_generations):
            real_pop_size = len(population)
            logger.info(f"<The {generation}th Generation: {real_pop_size} individuals>")
            # 3. fitness evaluation
            current_criteria = []
            current_models = []
            if n_proc == 1:
                for individual in population:
                    use_ids = set(sorted_chosen_ids[individual].tolist())
                    # if self.cover_all_observed_subpaths(use_ids):  # this should not happen
                    chosen_id_tuple = tuple(sorted(use_ids))
                    if chosen_id_tuple in cache_estimated:
                        logger.debug("Using cached result for {} variants".format(len(use_ids)))
                        res_prop, echo_prop, this_like, this_criteria = cache_estimated[chosen_id_tuple]
                    else:
                        cache_estimated[chosen_id_tuple] = \
                            res_prop, echo_prop, this_like, this_criteria = \
                            self.__compute_like_and_criteria(chosen_id_list=chosen_id_tuple, criteria=criterion, quiet=True)
                    # else:
                    #     # if the individual does not cover all observed subpaths, set the criterion to inf
                    #     logger.warning(f"Individual {use_ids} does not cover all observed subpaths after controlled filtering.")
                    #     this_criteria = float("inf")
                    #     res_prop = echo_prop = this_like = None
                    current_criteria.append(this_criteria)
                    current_models.append([use_ids, res_prop, echo_prop, this_like])
            else:
                # use multiprocessing to speed up the fitness evaluation 
                 # the order of the prop, echo, loglike and criterion may be randomized by the async, initialize them ahead and use w_id to track
                global_vars.w_id = 0
                global_vars.used_ids = manager.list([None] * real_pop_size)
                global_vars.prop = manager.list([None] * real_pop_size)
                global_vars.echo = manager.list([None] * real_pop_size)
                global_vars.loglike = manager.list([None] * real_pop_size)
                global_vars.criterion = manager.list([None] * real_pop_size)
                global_vars.finished_w = 0 # finished worker id, used to check if all workers are finished
                event.clear()
                if self.bootstrap_mode:
                    logger.debug("Serializing traversome for multiprocessing ..")
                else:
                    logger.info("Serializing traversome for multiprocessing ..")
                payload = dill.dumps((self.__compute_like_worker,
                                      (population, sorted_chosen_ids, real_pop_size, criterion, global_vars, lock, event, error_queue)))
                pool_obj = Pool(processes=n_proc)
                job_list = []
                for go_w in range(real_pop_size):
                    logger.trace("assigning job to worker {}".format(go_w + 1))
                    job_list.append(pool_obj.apply_async(run_dill_encoded, (payload,)))
                    logger.trace("assigned job to worker {}".format(go_w + 1))
                pool_obj.close()
                event.wait()
                pool_obj.terminate()
                while not error_queue.empty():  # TODO: check if this is working
                    e, tb, location = error_queue.get()
                    logger.error("\n" + "".join(tb))
                    sys.exit(1)
               
                # sorted_w_id_list = np.argsort(list(global_vars.w_id_list))
                logger.debug("All workers finished, collecting results ..")
                # logger.debug("Criterion values: {}".format(global_vars.criterion))
                current_criteria = np.array(global_vars.criterion)
                used_ids_ = np.array(global_vars.used_ids)
                prop_ = np.array(global_vars.prop)
                echo_ = np.array(global_vars.echo)
                loglike_ = np.array(global_vars.loglike)
                current_models = list(zip(used_ids_, prop_, echo_, loglike_))

            current_criteria = np.array(current_criteria)
            current_min_criterion = np.min(current_criteria)
            # 4. keep the elites
            elites = self.__ga_keep_elitism(population, current_criteria, num_elites)
            # 5. selection & crossover & mutation
            offspring = self.__ga_generate_offspring(
                population, current_criteria, population_size - num_elites, tournament_size,
                crossover_prob, mutation_prob, sorted_chosen_ids,
                cache_cover=cache_cover)
            offspring |= fixed_mask
            # 6. create the new population
            population = np.vstack((elites, offspring))
            # some algorithms will deduplicate the population then supply randomly-generated individuals
            # this will be computational expensive and slowing down the entire process; 
            # if we do deduplication, we probably need to reduce the population size and/or patience
            population = np.unique(population, axis=0)
            logger.info(f"  add {population_size - len(population)} more individuals after dedup")
            population = np.vstack(
                (population, 
                 self.__ga_initialize_population(population_size - len(population), sorted_chosen_ids, cache_cover=cache_cover)))
            population |= fixed_mask
            
            if current_min_criterion < best_criterion_val - equal_tol:  # improved
                no_improvement = 0
                # 7. update the best criterion value
                best_criterion_val = current_min_criterion
                # only keep the best model selection 
                best_models = {}
            elif abs(current_min_criterion - best_criterion_val) < equal_tol:  # no improvement
                no_improvement += 1
            else:  # worse
                logger.info("Elites should have been kept. Deterioration is unexpected, please check the code.")
                # logger.info("Current minimum criterion: {}".format(current_min_criterion))
                logger.info("Best criterion value: {}".format(best_criterion_val))
                no_improvement += 1
            
            if current_min_criterion < best_criterion_val - equal_tol \
                    or abs(current_min_criterion - best_criterion_val) < equal_tol:  # not getting worse
                # 8. update the best models
                best_ind_idx = [go_id 
                                for go_id, this_criterion in enumerate(current_criteria)
                                if abs(this_criterion - current_min_criterion) < equal_tol]
                for go_id in best_ind_idx:
                    use_ids, res_prop, echo_prop, this_like = current_models[go_id]
                    this_criteria = current_criteria[go_id]
                    sorted_used_ids = tuple(sorted(use_ids))
                    if sorted_used_ids not in best_models:
                        best_models[sorted_used_ids] = [res_prop, echo_prop, this_like, this_criteria]
            
            # 9. check for stopping criteria
            if no_improvement >= patience:
                logger.info("No improvement for {} generations, stopping the algorithm.".format(patience))
                break
            if not self.bootstrap_mode:
                logger.info("Generation {}: best criterion value: {}".format(generation, best_criterion_val))
                # logger.info("Current minimum criterion: {}".format(current_min_criterion))
                logger.info("Best models: {}".format(sorted(best_models)))

        # 10. return the best models
        # sorted by (criterion value, variant ids)
        # best_model = sorted(best_models.items(), key=lambda x: (x[1][3], x[0]))[0]
        self.__echo_comb_res(best_models)
        return best_models
    
    def __compute_like_worker(self, population, sorted_chosen_ids, real_pop_size, criterion, global_vars, lock, event, error_queue):
        """
        Worker function to compute the likelihood and criteria for a given set of chosen ids.
        """
        # TODO check
        try:
            # logger.debug("Worker {} started.".format(global_vars.w_id))
            with lock:
                ind_id = global_vars.w_id
                global_vars.w_id += 1
            use_id_set = set(sorted_chosen_ids[population[ind_id]].tolist())
            use_id_tuple = tuple(sorted(use_id_set))
            if use_id_tuple in global_vars.cache_estimated:
                logger.debug("Using cached result for {} variants".format(len(use_id_set)))
                res_prop, echo_prop, this_like, this_criteria = global_vars.cache_estimated[use_id_tuple]
            else:
                res_prop, echo_prop, this_like, this_criteria = \
                    self.__compute_like_and_criteria(chosen_id_list=use_id_tuple, criteria=criterion, quiet=True)
                with lock:
                    global_vars.cache_estimated[use_id_tuple] = [res_prop, echo_prop, this_like, this_criteria]
            with lock:
                logger.debug("Worker {} finished with criterion {}".format(ind_id + 1, this_criteria))
                global_vars.used_ids[ind_id] = use_id_set
                global_vars.prop[ind_id] = res_prop
                global_vars.echo[ind_id] = echo_prop
                global_vars.loglike[ind_id] = this_like
                global_vars.criterion[ind_id] = this_criteria
                global_vars.finished_w += 1
            if global_vars.finished_w == real_pop_size:
                event.set()
        except Exception as e:
            # e, tb, location
            error_queue.put((e, traceback.format_exc(), sys._getframe().f_code.co_name))
            event.set()
    
    def __ga_initialize_population(self, population_size, sorted_chosen_ids, num_tries=100, cache_cover=None):
        """
        Initialize the population with random individuals.
        :param population_size:
            The number of individuals in the population.
        :param sorted_chosen_ids:
            The sorted ids of the variants to be used. Used to get the size and to filter out invalid individuals.
        :return:
            The initialized population.
        """
        individual_size = len(sorted_chosen_ids)
        population = np.random.randint(2, size=(population_size, individual_size), dtype=np.bool_)
        # try many times to discard the failure ones and supply randomly generated individuals
        for supply_go in range(num_tries):
            # create a mask to filter out the invalid individuals
            # check if an individual is valid (cover all observed subpaths)
            valid_mask = np.array([self.cover_all_observed_subpaths(set(sorted_chosen_ids[individual].tolist()), cache_res=cache_cover)
                                   for individual in population])
            # if all individuals are valid
            if np.all(valid_mask):
                break
            # if not, filter out the valid individuals
            valid_individuals = population[valid_mask]
            logger.debug("Valid individuals: {}".format(len(valid_individuals)))
            if supply_go == num_tries - 1:  
                # go ahead and use the valid individuals (less than population_size)
                logger.debug("Using valid individuals: {}".format(len(valid_individuals)))
                population = valid_individuals
                break
            # generate new individuals to fill the population
            new_individuals = np.random.randint(2, size=(population_size - len(valid_individuals), individual_size), dtype=np.bool_)
            population = np.vstack((valid_individuals, new_individuals))
        return population

    def __ga_generate_offspring(self, 
                                population, 
                                inv_fitness_scores, 
                                num_offspring, 
                                tournament_size, 
                                crossover_prob, 
                                mutation_prob,
                                sorted_chosen_ids,
                                num_tries=20,
                                cache_cover=None):
        """
        Generate offspring using tournament selection, single-point crossover, and mutation.
        """
        # similar to self.__ga_initialize_population, we need to try many times to discard the failure ones
        valid_offspring = np.empty((0, len(sorted_chosen_ids)), dtype=np.bool_)
        num_to_gen = num_offspring
        for supply_go in range(num_tries):
            parents1 = self.__ga_tournament_selection(population, inv_fitness_scores, num_to_gen // 2, tournament_size)
            parents2 = self.__ga_tournament_selection(population, inv_fitness_scores, num_to_gen // 2, tournament_size)
            child1, child2 = self.__ga_single_point_crossover(parents1, parents2, crossover_prob)
            child1 = self.__ga_mutation(child1, mutation_prob)
            child2 = self.__ga_mutation(child2, mutation_prob)
            offspring = np.vstack((valid_offspring, child1, child2))
            # check if the offspring is valid (cover all observed subpaths)
            # create a mask to filter out the valid offspring
            valid_mask = np.array([self.cover_all_observed_subpaths(set(sorted_chosen_ids[individual]), cache_res=cache_cover) for individual in offspring]) 
            if np.all(valid_mask):
                break
            else:
                # filter out the valid offspring
                valid_offspring = offspring[valid_mask]
                logger.debug("Valid offspring: {}".format(len(valid_offspring)))
                if supply_go == num_tries - 1:  
                    # go ahead and use the valid offspring (less than num_offspring)
                    logger.debug("Using valid offspring: {}".format(len(valid_offspring)))
                    offspring = valid_offspring
                    break
                num_to_gen = num_offspring - len(valid_offspring)
                if num_to_gen <= 1:  # odd number of offspring or negative number of offspring to generate
                    offspring = valid_offspring
                    break
        return offspring

    def __ga_tournament_selection(self, population, inv_fitness_scores, num_tournaments, tournament_size=3):
        """
        :param population:
            The population of individuals.
        :param inv_fitness_scores:
            The criteria scores of the individuals, ie. the inverted fitness scores.
        :param num_tournaments:
            The number of tournaments to run.
        :param tournament_size:
            The number of individuals to compete in the tournament.
        :return:
            The selected individual.
        """
        # use the permutation to do the no replacement sampling
        tournament_indices = np.array([np.random.permutation(len(population))[:tournament_size] for _ in range(num_tournaments)])
        best_indices = np.argmin(inv_fitness_scores[tournament_indices], axis=1)
        return np.array(population)[tournament_indices[np.arange(len(best_indices)), best_indices]]

    def __ga_mutation(self, individuals, mutation_prob=0.01):
        """
        :param individuals:
            The individuals to mutate.
        :param mutation_prob:
            The probability of mutation for each gene.
        :return:
            The mutated individuals.
        """
        individuals = np.array(individuals)
        mutation_mask = np.random.rand(*individuals.shape) < mutation_prob
        return np.logical_xor(individuals, mutation_mask)

    def __ga_single_point_crossover(self, parents1, parents2, crossover_prob):
        """
        :param parents1:
            The first set of parents.
        :param parents2:
            The second set of parents.
        :param crossover_prob:
            The probability of crossover.
        :return:
            The children.
        """
        crossover_points = np.random.randint(1, len(parents1[0]), size=len(parents1))
        crossover_mask = np.random.rand(len(parents1)) < crossover_prob
        children1 = np.where(np.arange(len(parents1[0])) < crossover_points[:, np.newaxis], parents1, parents2)
        children2 = np.where(np.arange(len(parents1[0])) < crossover_points[:, np.newaxis], parents2, parents1)
        children1[~crossover_mask] = parents1[~crossover_mask]
        children2[~crossover_mask] = parents2[~crossover_mask]
        return children1, children2

    def __ga_keep_elitism(self, population, inv_fitness_scores, num_elites=2):
        """
        :param population:
            The population of individuals.
        :param inv_fitness_scores:
            The criteria scores of the individuals, ie. the inverted fitness scores.
        :param num_elites:
            The number of elite individuals to keep.
        :return:
            The elite individuals.
        """
        elite_indices = np.argsort(inv_fitness_scores)[:num_elites]
        return population[elite_indices]

    def reverse_model_selection(self,
                                n_proc,
                                criterion=Criterion.AIC,
                                chosen_ids: Union[typingODict[int, bool], Set] = None,
                                user_fixed_ids: Union[list, tuple, set, None] = None,
                                max_queue_size: int = None,
                                max_end_hits: int = None,
                                max_unchanged: int = None,
                                random_size=15,
                                c_diff_tolerance=1e-3,
                                p_diff_tolerance=1e-7,
                                parent_event=None,
                                ):
        """
        :param n_proc: number of processes
        :param criterion:
        :param chosen_ids:
            Only apply reverse model selection on chosen ids.
            The value of the OrderedDict has no use. We actually just want an OrderedSet.
        :param user_fixed_ids:
            user fixed variant ids that will not be dropped during model selection.
        :param max_queue_size:
            The maximum size of the candidate queue.
        :param max_end_hits:
            The maximum number of times a model can hit the end of a searching branch before stopping the algorithm.
        :param max_unchanged:
            The maximum number of times a model can be tested without changing the best model before stopping the algorithm.
        :param random_size: [0, INT)
            Each time randomly subsampling X models from the (N-1)-dim models
            instead of testing all of them like.
            Choose 0 to disable this process.
            # not good for large number of variants when there are local optima
        :param c_diff_tolerance:
            The tolerance for determining equivalence of the criteria.
        :param p_diff_tolerance:
            The tolerance for determining whether proportions are zero.
        :param parent_event: Manager.Event, if provided along with n_proc > 1,
            the parent_event will be used to call all subprocesses to stop
        """
        if chosen_ids:
            # chosen_ids = OrderedDict([(self.be_unidentifiable_to[variant_id], True)
            #                           for variant_id in chosen_ids])
            chosen_ids = {self.be_unidentifiable_to[variant_id] for variant_id in chosen_ids}
        else:
            # chosen_ids = OrderedDict([(variant_id, True) for variant_id in self.repr_to_merged_variants])
            chosen_ids = {variant_id for variant_id in self.repr_to_merged_variants}
        chosen_ids = set(chosen_ids)

        len_chosen_ids = len(chosen_ids)
        # arbitrarily set the max_queue_size, max_end_hits and max_unchanged
        if max_queue_size is None:
            max_queue_size = len_chosen_ids * 50  # default to 40 times the number of chosen ids
        if max_end_hits is None:
            max_end_hits = len_chosen_ids // 2 + 20
        if max_unchanged is None:
            max_unchanged = len_chosen_ids * 10
        if self.bootstrap_mode:
            logger.debug("Reverse model selection with max_queue_size={}, max_end_hits={}, max_unchanged={}"
                         .format(max_queue_size, max_end_hits, max_unchanged))
        else:
            logger.info("Reverse model selection with max_queue_size={}, max_end_hits={}, max_unchanged={}"
                        .format(max_queue_size, max_end_hits, max_unchanged))
        
        # Because many traversome attributes including subpath information were created using the original variant
        # ids, so here we prefer not making traversome.get_multinomial_like_formula complicated. Instead, we create
        # variant_percents with foo values inserted when that variant id is not in chosen_ids_set.
        self.variant_percents = [symengine.Symbol("P" + str(variant_id)) if variant_id in chosen_ids else False
                                 for variant_id in range(self.num_put_variants)]
        sampled_ids_sorted_tuple = tuple(sorted(chosen_ids))

        if len(sampled_ids_sorted_tuple) < 5:
            n_proc = 1  # if the number of variants is too small, use single process to avoid overhead

        # do multiple rounds of reverse model selection then summarize the results, in case there are local optima and multiple equivalent models
        if n_proc == 1:
            # record the ongoing node that not hit an end yet
            # each element should contain [var_ids_tuple, indispensable_ids, previous_criteria]
            #  - var_ids_tuple is the tuple of sorted variant ids that used for testing
            #  - indispensable_ids is a dict of indispensable variant ids inherited from the previous selection step
            #  - previous_criteria is the criteria of the previous selection step for sorting the queue
            candidate_queue = []
        else:
            manager = Manager()
            atexit.register(manager.shutdown)  # ensure the manager is shutdown at the end of the program
            error_queue = manager.Queue()
            job_id_queue = manager.Queue()  # used to track the job ids
            for job_id in range(n_proc):
                job_id_queue.put(job_id)
            event = manager.Event()
            lock = manager.Lock()  # global lock
            global_vars = manager.Namespace()
            # stop criteria 1
            global_vars.running_status = manager.list([1] * n_proc)  # record the running status of the workers: 1 for running, 0 for idle
            # stop criteria 2
            global_vars.count_hit_end = manager.Value("i", 0)  # record the number of tests that hit the end
            global_vars.count_hit_end_lock = manager.Lock()  # lock for the count_hit_end
            # stop criteria 3
            global_vars.count_unchanged = manager.Value("i", 0)  # record the number of tests that did not change the best model
            global_vars.count_unchanged_lock = manager.Lock()  # lock for the count_unchanged
            #
            global_vars.candidate_queue = manager.list()
            global_vars.candidate_queue_lock = manager.Lock()  # lock for the candidate queue
            global_vars.explored_tuples = manager.dict()  # record the explored tuples
            global_vars.explored_tuples_lock = manager.Lock()  # lock for the explored tuples
            global_vars.best_lock = manager.Lock()  # lock for the best models and criteria
            global_vars.best_models = manager.dict()  # only store the best models
            global_vars.best_criteria = manager.Value("f", float("inf"))  # best criteria for the reverse model selection
            global_vars.cache_cover = manager.dict()  # cache the cover results

        # if one component is identified as indispensable in the n-dimension model,
        # it will be indispensable for subsequent (n-m)-dimension models
        if user_fixed_ids:  # in the case of user assigned fixed 'indispensable' variant id(s)
            indispensable_ids = set(user_fixed_ids)
        else:
            indispensable_ids = set()

        previous_criteria = 0
        if n_proc == 1:
            # if n_proc == 1, we can use the candidate_queue to store the ongoing nodes
            candidate_queue.append([sampled_ids_sorted_tuple, indispensable_ids, previous_criteria])
            best_models = self.__reverse_model_search(criterion=criterion,
                                                      max_queue_size=max_queue_size,
                                                      max_end_hits=max_end_hits,
                                                      max_unchanged=max_unchanged,
                                                      random_size=random_size,
                                                      c_diff_tolerance=c_diff_tolerance,
                                                      p_diff_tolerance=p_diff_tolerance,
                                                      candidate_queue=candidate_queue)
        else:
            # Monitoring the parent event to stop the workers when the parent event is set
            # if parent_event is not None:
            #     def parent_event_monitor():
            #         parent_event.wait()
            #         event.set()
            #     # start a thread to monitor the parent event
            #     monitor_thread = threading.Thread(target=parent_event_monitor, daemon=True)
            #     monitor_thread.start()

            # if n_proc > 1, we can use the global_vars.candidate_queue to store the ongoing nodes
            global_vars.candidate_queue.append([sampled_ids_sorted_tuple, indispensable_ids, previous_criteria])
            # global_vars.explored_tuples[sampled_ids_sorted_tuple] = True

            if self.bootstrap_mode:
                logger.debug("Serializing traversome for multiprocessing ..")
            else:
                logger.info("Serializing traversome for multiprocessing ..")
            payload = dill.dumps((self.__reverse_model_search_worker,
                                 (criterion, max_queue_size, max_end_hits, max_unchanged,
                                  random_size, c_diff_tolerance, p_diff_tolerance, global_vars,
                                  event, lock, job_id_queue, error_queue, parent_event)))
            pool_obj = Pool(processes=n_proc)
            job_list = []
            # for go_w in range(len(this_rd_ids)):
            for go_w in range(n_proc):
                # TODO: to fix the issue that the behaviour of the logger in the worker become be different
                #       sim.alignment.new.100k.300k.100x.traversome-bic-N1000-user-p
                logger.trace("assigning job to worker {}".format(go_w + 1))
                job_list.append(pool_obj.apply_async(run_dill_encoded, (payload,)))
                logger.trace("assigned job to worker {}".format(go_w + 1))
            pool_obj.close()
            # event.wait()
            # use a loop to consider both the event and the parent_event
            while True:
                if event.is_set():
                    break
                if parent_event is not None and parent_event.is_set():
                    event.set()
                    break
                time.sleep(0.3)
            pool_obj.terminate()
            # pool_obj.join()  will make the main process wait for the pool to finish, which is not what we want here
            while not error_queue.empty():
                e, tb, location = error_queue.get()
                logger.error("\n" + "".join(tb))  # + "\n" + str(location) + "\n" + str(e))
                sys.exit(1)
            best_models = dict(global_vars.best_models)

        self.__echo_comb_res(best_models)
        return best_models

    def reverse_model_selection_using_reps(self,
                                           n_proc,
                                           criterion=Criterion.AIC,
                                           chosen_ids: Union[typingODict[int, bool], Set] = None,
                                           # random_size: int = 0,
                                           user_fixed_ids: Union[list, tuple, set, None] = None,
                                           n_repeats: int = 10,
                                           # n_random_start_repeats: int = 5
                                           ):
        """
        :param n_proc: number of processes
        :param criterion:
        :param chosen_ids:
            Only apply reverse model selection on chosen ids.
            The value of the OrderedDict has no use. We actually just want an OrderedSet.
        # :param random_size: [0, INT)
        #     Each time randomly subsampling X models from the (N-1)-dim models
        #     instead of testing all of them like.
        #     Choose 0 to disable this process.
        #     # not good for large number of variants when there are local optima
        :param user_fixed_ids:
            user fixed variant ids that will not be dropped during model selection.
        """
        # TODO: tolerance can be larger
        # if random_size != 0 and n_proc > random_size and not self.__warning_sent:
        #     # logger.warning(f"random size {random_size} is smaller than the num of processes {n_proc}, "
        #     #                f"which is limited by the former.")
        #     self.__warning_sent = True

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
        
        # do multiple rounds of reverse model selection then summarize the results, in case there are local optima and multiple equivalent models
        if n_proc == 1:
            cache_estimated = {}  # cache the estimated results
            cache_cover = {}
        else:
            manager = Manager()
            error_queue = manager.Queue()
            event = manager.Event()
            lock = manager.Lock()
            global_vars = manager.Namespace()
            global_vars.cache_estimated = manager.dict()
            global_vars.cache_cover = manager.dict()
        best_models = {}
        best_criteria = float("inf")
        # there should not be the n_random_start_repeats, because this will work against the reverse model selection
        # total_repeats = n_repeats + n_random_start_repeats
        # if n_random_start_repeats:
        #     # use self.__ga_initialize_population to generate random selections so that the random ones are guaranteed to cover all observed subpaths
        #     if n_proc == 1:
        #         random_selections = self.__ga_initialize_population(population_size=n_random_start_repeats, 
        #                                                             sorted_chosen_ids=np.array(sorted(chosen_ids)),
        #                                                             cache_cover=cache_cover)
        #     else:
        #         random_selections = self.__ga_initialize_population(population_size=n_random_start_repeats, 
        #                                                             sorted_chosen_ids=np.array(sorted(chosen_ids)),
        #                                                             cache_cover=global_vars.cache_cover)
        for go_repeat in range(n_repeats):
            sampled_ids = set(chosen_ids)

            # NO random sampling for reverse model selection
            # if go_repeat < n_repeats:
            #     # if we are in the first n_repeats, we will use the original chosen_ids
            #     # otherwise, we will randomly sample from the chosen_ids
            #     sampled_ids = set(chosen_ids)
            #     if self.bootstrap_mode:
            #         logger.debug("Reverse model selection repeat {}/{}".format(go_repeat + 1, total_repeats))
            #     else:
            #         logger.info("Reverse model selection repeat {}/{}".format(go_repeat + 1, total_repeats))
            # else:
            #     sampled_ids = set(np.array(sorted(chosen_ids))[random_selections[go_repeat - n_repeats]])
            #     if self.bootstrap_mode:
            #         logger.debug("Reverse model selection random-repeat {}/{}".format(go_repeat + 1, total_repeats))
            #     else:
            #         logger.info("Reverse model selection random-repeat {}/{}".format(go_repeat + 1, total_repeats))
        
            # if one component is identified as indispensable in the n-dimension model,
            # it will be indispensable for subsequent (n-m)-dimension models
            if user_fixed_ids:  # in the case of user assigned fixed 'indispensable' variant id(s)
                indispensable_ids = {u_id: True for u_id in user_fixed_ids}
            else:
                indispensable_ids = {}
            
            logger.debug("Test variants {}".format(list(sampled_ids)))
            tuple_sampled_ids = tuple(sorted(sampled_ids))
            if n_proc == 1:
                here_cache_est = cache_estimated
            else:
                here_cache_est = global_vars.cache_estimated
            if tuple_sampled_ids in here_cache_est:
                logger.debug("Using cached result for initial {} variants".format(len(sampled_ids)))
                previous_prop, previous_echo, previous_like, previous_criteria = here_cache_est[tuple_sampled_ids]
            else:
                # compute the initial likelihood and criteria
                logger.debug("Computing initial likelihood and criteria ..")
                here_cache_est[tuple_sampled_ids] = previous_prop, previous_echo, previous_like, previous_criteria = \
                    self.__compute_like_and_criteria(chosen_id_list=tuple_sampled_ids, criteria=criterion)

            # logger.info("Proportions: %s " % {_iid: previous_prop[_gid] for _gid, _iid in enumerate(chosen_ids_set)})
            # logger.info("Log-likelihood: %s" % previous_like)
            # drop zero prob variant
            sampled_ids = self.__drop_zero_variants(sampled_ids, previous_prop, previous_echo, diff_tolerance, indispensable_ids)

            # stepwise
            while len(sampled_ids) > 1:
                logger.debug("Trying dropping {} variant(s) ..".format(self.num_put_variants - len(sampled_ids) + 1))
                chosen_ids_sorted = sorted(sampled_ids)
                # chosen_rd_list = list(range(len(chosen_ids_sorted)))
                # np.random.shuffle(chosen_rd_list)
                changed = False
                # rs = random_size if random_size > 0 else len(chosen_rd_list)
                # for go_rd_sp in range(0, len(chosen_rd_list), rs):
                #     this_rd_ids = chosen_rd_list[go_rd_sp: go_rd_sp+rs]
                test_id_res = OrderedDict()
                if n_proc == 1:
                    # for rd_id in this_rd_ids:
                    for try_id in chosen_ids_sorted:
                        self.__test_one_drop(
                            # var_id=chosen_ids_sorted[rd_id],
                            var_id=try_id,
                            chosen_ids=chosen_ids,
                            sorted_chosen_ids=chosen_ids_sorted,
                            criterion=criterion,
                            test_id_res=test_id_res,
                            indispensable_ids=indispensable_ids,
                            cache_cover=cache_cover,
                            cache_estimated=cache_estimated)
                else:
                    # TODO
                    # manager = Manager()
                    # error_queue = manager.Queue()
                    # event = manager.Event()
                    # lock = manager.Lock()
                    # global_vars = manager.Namespace()
                    global_vars.w_id = 0  # worker id, looks not used
                    global_vars.recorded_ids = manager.list()  # g_vars.recorded_ids is used to record the tested ids
                    global_vars.prop = manager.list()
                    global_vars.echo = manager.list()
                    global_vars.loglike = manager.list()
                    global_vars.criterion = manager.list()
                    global_vars.finished_w = 0  # finished worker id, used to check if all workers are finished
                    event.clear()  
                    # The indispensable_ids is a shortcut for the __test_one_drop to skip the indispensable ones.
                    # These are the ones which cannot be dropped in the reverse model selection process.
                    # So that we don't need to call cover_all_observed_subpaths() for all tests.
                    # So the indispensable_ids can only be applied in the reverse model selection algorithm, not in other randomization algorithms, e.g. genetic algorithm.
                    global_vars.indispensable_ids = manager.dict()
                    global_vars.indispensable_ids.update(indispensable_ids)
                    if self.bootstrap_mode:
                        logger.debug("Serializing traversome for multiprocessing ..")
                    else:
                        logger.info("Serializing traversome for multiprocessing ..")
                    # payload = dill.dumps((self.__test_one_drop_worker,
                    #                     (this_rd_ids, sampled_ids, criterion, global_vars, lock, event, error_queue)))
                    payload = dill.dumps((self.__test_one_drop_worker,
                                        (sampled_ids, criterion, global_vars, lock, event, error_queue)))
                    pool_obj = Pool(processes=n_proc)
                    job_list = []
                    # for go_w in range(len(this_rd_ids)):
                    for go_w in range(len(chosen_ids_sorted)):
                        # TODO: to fix the issue that the behaviour of the logger in the worker become be different
                        #       sim.alignment.new.100k.300k.100x.traversome-bic-N1000-user-p
                        logger.trace("assigning job to worker {}".format(go_w + 1))
                        job_list.append(pool_obj.apply_async(run_dill_encoded, (payload,)))
                        logger.trace("assigned job to worker {}".format(go_w + 1))
                    pool_obj.close()
                    event.wait()
                    pool_obj.terminate()
                    while not error_queue.empty():
                        e, tb, location = error_queue.get()
                        logger.error("\n" + "".join(tb))  # + "\n" + str(location) + "\n" + str(e))
                        sys.exit(1)
                    # pool.join()
                    # use the global_vars.recorded_ids to collect the corresponding resultI am
                    for go_r, var_id in enumerate(list(global_vars.recorded_ids)):
                        test_id_res[var_id] = \
                            {"prop": global_vars.prop[go_r], "echo": global_vars.echo[go_r],
                            "loglike": global_vars.loglike[go_r], criterion: global_vars.criterion[go_r]}
                    indispensable_ids.update(dict(global_vars.indispensable_ids))
                if test_id_res:
                    # modify the code to allow random selection from equivalent models that has similar criteria (within diff_tolerance)
                    # find the best drop id that minimize the criteria
                    sorted_res = sorted([[_go_var_, test_id_res[_go_var_][criterion]]
                                            for _go_var_ in test_id_res],
                                            key=lambda x: x[1])
                    best_drop_id, best_val = sorted_res[0]
                    # if the best_val is not better than the previous_criteria, we will not drop any variant
                    if best_val >= previous_criteria + diff_tolerance:
                        continue
                    else:
                        num_equivalent = len([x for x in sorted_res if abs(x[1] - best_val) < diff_tolerance])
                        # add equivalent to the previous_criteria as one case to draw
                        equivalent_to_previous = abs(best_val - previous_criteria) < diff_tolerance
                        draw_num = num_equivalent + 1 if equivalent_to_previous else num_equivalent
                        select_drop_id = np.random.randint(0, draw_num)
                        if equivalent_to_previous and select_drop_id == draw_num - 1:
                            # if we select the last one, we will not drop any variant
                            continue
                        # update the best drop id and value
                        best_drop_id, best_val = sorted_res[select_drop_id]
                        #
                        previous_criteria = best_val
                        previous_like = test_id_res[best_drop_id]["loglike"]
                        previous_prop = test_id_res[best_drop_id]["prop"]
                        previous_echo = test_id_res[best_drop_id]["echo"]
                        sampled_ids.remove(best_drop_id)
                        # drop candidate id that minimize criteria
                        if self.bootstrap_mode:
                            logger.debug("Drop {}".format(self.__str_rep_id(best_drop_id)))
                            logger.debug("  intermediate proportions: " +
                                        ", ".join(["%s:%.4f" % (_id, _p) for _id, _p, in previous_echo.items()]))
                            logger.debug("  intermediate log-likelihood: %s" % previous_like)
                        else:
                            logger.info("Drop {}".format(self.__str_rep_id(best_drop_id)))
                            logger.info("  intermediate Proportions: " +
                                        ", ".join(["%s:%.4f" % (_id, _p) for _id, _p, in previous_echo.items()]))
                            logger.info("  intermediate log-likelihood: %s" % previous_like)
                        sampled_ids = self.__drop_zero_variants(
                            sampled_ids, previous_prop, previous_echo, diff_tolerance, indispensable_ids)
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
                    # if self.bootstrap_str:
                    #     logger.info(f"{self.bootstrap_str} Proportions: " +
                    #                 ", ".join(["%s:%.4f" % (_id, _p) for _id, _p, in previous_echo.items()]))
                    #     logger.debug("Log-likelihood: %s" % previous_like)
                    # else:
                    #     logger.info("Proportions: " +
                    #                 ", ".join(["%s:%.4f" % (_id, _p) for _id, _p, in previous_echo.items()]))
                    #     logger.info("Log-likelihood: %s" % previous_like)
                    # self.__echo_res(previous_echo, previous_like)
                    # return previous_prop, previous_like, previous_criteria
                    break  # no more variants can be dropped, stop the stepwise process
            # use a slightly higher log level
            # logger.log("RES", "Proportions: " + ", ".join(["%s:%.4f" % (_id, _p) for _id, _p, in previous_echo.items()]))
            # if self.bootstrap_str:
            #     logger.info(f"{self.bootstrap_str} Proportions: " +
            #                 ", ".join(["%s:%.4f" % (_id, _p) for _id, _p, in previous_echo.items()]))
            #     logger.debug("Log-likelihood: %s" % previous_like)
            # else:
            #     logger.info("Proportions: " + ", ".join(["%s:%.4f" % (_id, _p) for _id, _p, in previous_echo.items()]))
            #     logger.info("Log-likelihood: %s" % previous_like)

            if previous_criteria < best_criteria - diff_tolerance:  # improved
                best_criteria = previous_criteria
                # only keep the best model selection
                best_models = {tuple(sorted(sampled_ids)): [previous_prop, previous_echo, previous_like, previous_criteria]}
            elif abs(previous_criteria - best_criteria) < diff_tolerance:  # no improvement
                # add the current model to the best models
                best_models[tuple(sorted(sampled_ids))] = [previous_prop, previous_echo, previous_like, previous_criteria]
            else:  # worse
                pass
        # self.__echo_res(previous_echo, previous_like)
        # if len(best_models) > 1:
        self.__echo_comb_res(best_models)
        # else:
        #     self.__echo_res(previous_echo, previous_like)
        # return previous_prop, previous_like, previous_criteria
        return best_models
    
    def __echo_comb_res(self, best_models):
        # sorted by (variant ids)
        sorted_best_models = sorted(best_models.items(), key=lambda x: x[0])
        # similar to self.__echo_res but use ; to join the proportions from different equally-good models
        proportion_str_list = []
        like_str_list = []
        for model_id, model_res in sorted_best_models:
            proportion_str_list.append(", ".join(["%s:%.4f" % (_id, _p) for _id, _p, in model_res[1].items()]))
            like_str_list.append(str(model_res[2]))
        if self.bootstrap_mode:
            logger.info(f"{self.bootstrap_mode} Proportions: " + "; ".join(proportion_str_list))
            logger.debug(f"{self.bootstrap_mode} Log-likelihood: %s" % "; ".join(like_str_list))
        else:
            logger.info("Proportions: " + "; ".join(proportion_str_list))
            logger.info("Log-likelihood: %s" % "; ".join(like_str_list))

    # def __echo_res(self, previous_echo, previous_like):
    #     if self.bootstrap_mode:
    #         logger.info(f"{self.bootstrap_mode} Proportions: " +
    #                     ", ".join(["%s:%.4f" % (_id, _p) for _id, _p, in previous_echo.items()]))
    #         logger.debug(f"{self.bootstrap_mode} Log-likelihood: %s" % previous_like)
    #     else:
    #         logger.info("Proportions: " + ", ".join(["%s:%.4f" % (_id, _p) for _id, _p, in previous_echo.items()]))
    #         logger.info("Log-likelihood: %s" % previous_like)

    def __drop_zero_variants_new(self, sorted_chosen_ids: Union[List, Tuple], representative_props, diff_tolerance, indispensable_ids, work_id=None):
        """
        Generating new model by dropping in the variants with estimated proportions of zero in one round of model selection, 
        designed to speed up the searching process)"""
        log_var_ids = []
        new_chosen_ids = list(sorted_chosen_ids)
        del_list = []
        for go_v, var_id in enumerate(sorted_chosen_ids):
            # do not drop a candidate variant if it is fixed either by the user or by a read path
            if var_id in indispensable_ids:
                continue
            if abs(representative_props[var_id] - 0.) < diff_tolerance:
                del_list.append(go_v)
                log_var_ids.append(self.__str_rep_id(var_id))
                # drop candidate id that has estimated proportion of zero
        work_id_str = f"Worker {work_id}: " if work_id is not None else ""
        if log_var_ids:
            logger.debug("{}Drop {}".format(work_id_str, ", ".join(log_var_ids)))
        else:
            logger.trace("{}No zero-prop variants dropped".format(work_id_str))
        # remove the variants from the chosen_ids
        for go_v in reversed(del_list):
            del new_chosen_ids[go_v]
        return tuple(new_chosen_ids)


    def __drop_zero_variants_list(self, sorted_chosen_ids: Union[List, Tuple], representative_props, diff_tolerance, indispensable_ids):
        """
        # deprecated: may potentially drop the real components when there are equivalent models

        Drop the variants with estimated proportions of zero in one round of model selection, 
        designed to speed up the dropping process)"""
        log_var_ids = []
        new_chosen_ids = list(sorted_chosen_ids)
        del_list = []
        for go_v, var_id in enumerate(sorted_chosen_ids):
            # do not drop a candidate variant if it is fixed either by the user or by a read path
            if var_id in indispensable_ids:
                continue
            if abs(representative_props[var_id] - 0.) < diff_tolerance:
                del_list.append(go_v)
                for cid_var_id in self.repr_to_merged_variants[var_id]:
                    del representative_props[cid_var_id]
                log_var_ids.append(self.__str_rep_id(var_id))
                # drop candidate id that has estimated proportion of zero
        if log_var_ids:
            if self.bootstrap_mode:
                logger.debug("Drop {}".format(", ".join(log_var_ids)))
            else:
                logger.info("Drop {}".format(", ".join(log_var_ids)))
        else:
            if self.bootstrap_mode:
                logger.trace("No zero-prop variants dropped")
            else:
                logger.debug("No zero-prop variants dropped")
        # remove the variants from the chosen_ids
        for go_v in reversed(del_list):
            del new_chosen_ids[go_v]
        return tuple(new_chosen_ids)

    def __drop_zero_variants(self, chosen_ids: Union[List, Tuple, Set], representative_props, echo_props, diff_tolerance, indispensable_ids):
        """
        # deprecated: may potentially drop the real components when there are equivalent models

        Drop the variants with estimated proportions of zero in one round of model selection, 
        designed to speed up the dropping process)"""
        log_var_ids = []
        new_chosen_ids = set(chosen_ids)
        for var_id in list(chosen_ids):
            # do not drop a candidate variant if it is fixed either by the user or by a read path
            if var_id in indispensable_ids:
                continue
            if abs(representative_props[var_id] - 0.) < diff_tolerance:
                new_chosen_ids.remove(var_id)
                for cid_var_id in self.repr_to_merged_variants[var_id]:
                    del representative_props[cid_var_id]
                del echo_props[self.__str_rep_id(var_id)]
                log_var_ids.append(self.__str_rep_id(var_id))
                # drop candidate id that has estimated proportion of zero
        if log_var_ids:
            if self.bootstrap_mode:
                logger.debug("Drop {}".format(", ".join(log_var_ids)))
            else:
                logger.info("Drop {}".format(", ".join(log_var_ids)))
        return new_chosen_ids

    def __test_one_drop(self, var_id, chosen_ids: set, sorted_chosen_ids, criterion, test_id_res, indispensable_ids, cache_cover=None, cache_estimated=None):
        """
        """
        if cache_estimated is None:
            cache_estimated = {}
        if var_id not in indispensable_ids:
            testing_ids = chosen_ids - {var_id}
            if self.cover_all_observed_subpaths(testing_ids, cache_res=cache_cover):
                # logger.debug(
                #     "Test variants [{}] - {}".
                #         format(", ".join([self.__str_rep_id(_c_i)
                #                           for _c_i in sorted_chosen_ids]), self.__str_rep_id(var_id)))
                tuple_testing_ids = tuple(sorted(testing_ids))
                if tuple_testing_ids in cache_estimated:
                    logger.debug("Using cached result for {} variants".format(len(testing_ids)))
                    res_list = cache_estimated[tuple_testing_ids]
                else:
                    # compute the likelihood and criteria
                    res_list = self.__compute_like_and_criteria(chosen_id_list=tuple_testing_ids, criteria=criterion)
                    # store the result in the cache_estimated
                    cache_estimated[tuple_testing_ids] = res_list
                test_id_res[var_id] = \
                    {"prop": res_list[0], "echo": res_list[1], "loglike": res_list[2], criterion: res_list[3]}
                logger.debug(
                    "Test variants [{}] - {}: criterion={}, loglike={}"
                        .format(", ".join([self.__str_rep_id(_c_i)
                                          for _c_i in sorted_chosen_ids]), self.__str_rep_id(var_id),
                                res_list[3], res_list[2]))
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

    def __reverse_model_search(
            self,
            criterion,
            max_queue_size,
            max_end_hits,
            max_unchanged,
            random_size,
            c_diff_tolerance,
            p_diff_tolerance,
            candidate_queue,
    ):
        """
        continuously calculate the likelihood and criteria for the variants using single process.

        :param criterion: the criterion to use for model selection
        :param max_queue_size: the maximum size of the candidate queue
        :param max_end_hits: the maximum number of times we can hit the end of the search
        :param max_unchanged: patience of the search, the maximum number of times we can have unchanged results
        :param random_size: randomly drop variants from the candidate model to generate new candidates to speed up the search. Use 0 to disable random dropping and use exhaustive search instead.
        :param c_diff_tolerance: the tolerance for the difference between criteria
        :param p_diff_tolerance: the tolerance for the difference between proportions
        :param candidate_queue: list of (var_ids_tuple, indispensable_ids, previous_criteria)
        :param cache_cover: optional cache for cover_all_observed_subpaths
        """
        cache_cover={}  # cache the cover results
        explored_tuples = set()  # whenever hit the explored_tuple, stop futher steps
        best_criteria = float('inf')
        best_models = dict()
        count_unchanged = 0
        count_hit_end = 0
        count_n_searches = 0

        logger.debug("Searching started")
        while True:
            if not candidate_queue:
                if self.bootstrap_mode:
                    logger.debug("No candidates left, terminating")
                else:
                    logger.info("No candidates left, terminating")
                break

            q_size = len(candidate_queue)
            chose_m = get_randint_by_exp_weights(q_size)
            var_ids_tuple, indispensable_ids, previous_criteria = candidate_queue.pop(chose_m)
            logger.debug(f"Testing variants {var_ids_tuple}")

            if var_ids_tuple in explored_tuples:
                logger.debug(f"{var_ids_tuple} already explored")
                continue
            explored_tuples.add(var_ids_tuple)
            count_n_searches += 1
            if count_n_searches % 10 == 0:
                if self.bootstrap_mode:
                    logger.trace(f"NSearches={count_n_searches}, Queue size={len(candidate_queue)}, End hits={count_hit_end}, Unchanged={count_unchanged}")
                else:
                    logger.debug(f"NSearches={count_n_searches}, Queue size={len(candidate_queue)}, End hits={count_hit_end}, Unchanged={count_unchanged}")

            prop, info, loglike, criteria_val = self.__compute_like_and_criteria(
                chosen_id_list=var_ids_tuple, criteria=criterion)

            # Drop zero-prob variants and add the new model to the queue to speed up the search
            if len(var_ids_tuple) > 1:
                new_ids_tuple = self.__drop_zero_variants_new(
                    sorted_chosen_ids=var_ids_tuple,
                    representative_props=prop,
                    diff_tolerance=p_diff_tolerance,
                    indispensable_ids=indispensable_ids,
                )
                if new_ids_tuple != var_ids_tuple and new_ids_tuple not in explored_tuples:
                    idx = bisect_left([item[2] for item in candidate_queue], criteria_val)
                    candidate_queue[idx:idx] = [(new_ids_tuple, set(indispensable_ids), criteria_val)]
                    if len(candidate_queue) > max_queue_size:
                        del candidate_queue[max_queue_size:]

            logger.debug(
                "Test variants [{}]: criterion={}, loglike={}".format(
                    ", ".join([self.__str_rep_id(_c_i) for _c_i in var_ids_tuple]),
                    criteria_val, loglike
                )
            )

            # Best criteria management
            if criteria_val < best_criteria - c_diff_tolerance:
                best_criteria = criteria_val
                best_models.clear()
                best_models[var_ids_tuple] = [prop, info, loglike, criteria_val]
                if self.bootstrap_mode:
                    logger.debug(f"Current best criteria {criteria_val}; best model: {[var_ids_tuple]}")
                else:
                    logger.info(f"Current best criteria {criteria_val}; best model: {[var_ids_tuple]}")
                count_unchanged = 0
            elif abs(criteria_val - best_criteria) < c_diff_tolerance:
                best_models[var_ids_tuple] = [prop, info, loglike, criteria_val]
                if self.bootstrap_mode:
                    logger.debug(f"Current best criteria {best_criteria}; best models: {list(best_models.keys())}")
                else:
                    logger.info(f"Current best criteria {best_criteria}; best models: {list(best_models.keys())}")
                count_unchanged = 0
            else:
                count_unchanged += 1
                if count_unchanged >= max_unchanged:
                    if self.bootstrap_mode:
                        logger.debug(f"Hit unchanged {count_unchanged} times, stopping")
                    else:
                        logger.info(f"Hit unchanged {count_unchanged} times, stopping")
                    break

            # generate new candidates for next step
            var_ids_size = len(var_ids_tuple)
            candidate_tuples_here = []
            if var_ids_size == 1:
                count_new_indispensable = 1
            else:
                indispensable_ids = set(indispensable_ids)
                count_new_indispensable = 0
                if var_ids_size > random_size > 0:
                    drop_pool = np.random.choice(range(var_ids_size), size=random_size, replace=False).tolist()
                else:
                    drop_pool = range(var_ids_size)
                for go_drop in drop_pool:
                    new_var_ids_tuple = var_ids_tuple[:go_drop] + var_ids_tuple[go_drop+1:]
                    if self.cover_all_observed_subpaths(new_var_ids_tuple, cache_res=cache_cover):
                        if new_var_ids_tuple in explored_tuples:
                            continue
                        candidate_tuples_here.append(new_var_ids_tuple)
                    else:
                        indispensable_ids.add(var_ids_tuple[go_drop])
                        count_new_indispensable += 1
                        explored_tuples.add(new_var_ids_tuple)
            if count_new_indispensable == var_ids_size:
                count_hit_end += 1
                logger.debug("All variants are indispensable, no new candidates generated")
                if count_hit_end >= max_end_hits:
                    if self.bootstrap_mode:
                        logger.debug(f"Hit end of search {count_hit_end} times, stopping")
                    else:
                        logger.info(f"Hit end of search {count_hit_end} times, stopping")
                    break
                continue
            if not candidate_tuples_here:
                logger.debug("No new candidates generated")
                continue
            logger.debug(f"{len(candidate_tuples_here)} new candidates generated")
            candidate_tuples_here.sort()
            idx = bisect_left([item[2] for item in candidate_queue], criteria_val)
            candidate_queue[idx:idx] = [
                (new_var_ids_tuple, set(indispensable_ids), criteria_val)
                for new_var_ids_tuple in candidate_tuples_here
            ]
            if len(candidate_queue) > max_queue_size:
                del candidate_queue[max_queue_size:]
        return best_models

    def __reverse_model_search_worker(
            self,
            criterion,
            max_queue_size,
            max_end_hits,
            max_unchanged,
            random_size,
            c_diff_tolerance,
            p_diff_tolerance,
            g_vars,
            event,
            lock,
            job_id_queue,
            error_queue,
            parent_event
            # logger  # ?not working for multiprocessing
            ):
        """
        continuously calculate the likelihood and criteria for the variants

        :param criterion: the criterion to use for model selection
        :param max_queue_size: the maximum size of the candidate queue
        :param max_end_hits: the maximum number of times we can hit the end of the search
        :param max_unchanged: patience of the search, the maximum number of times we can have unchanged results
        :param random_size: randomly drop variants from the candidate model to generate new candidates to speed up the search. Use 0 to disable random dropping and use exhaustive search instead.
        :param c_diff_tolerance: the tolerance for the difference between criteria
        :param p_diff_tolerance: the tolerance for the difference between proportions
        :param g_vars: the global variables for the worker
        :param event: the event to signal the termination of the worker
        :param job_id_queue: the queue for the worker id
        :param error_queue: the queue for the error messages
        # :param logger: the logger for the worker, for compatibility with windows multiprocessing # but cannot work on windows due to it's not under __main__
        """
        try:
            work_id = job_id_queue.get()
            with lock:
                # explicitly reset logger, for consistent log levels across processes on both OS
                setup_logger(loglevel=self.loglevel, timed=True, log_file=self.logfile)
            logger.debug("Worker {} started".format(work_id))
            while not event.is_set() and (parent_event is None or not parent_event.is_set()):
                with g_vars.candidate_queue_lock:
                    if len(g_vars.candidate_queue) == 0:
                        g_vars.running_status[work_id] = 0
                        if sum(g_vars.running_status) == 0:
                            # if all workers are not running, we can terminate the event
                            if self.bootstrap_mode:
                                # TODO this can be sub-worker of a worker, so the work_id should be more complicated
                                logger.debug("Worker {}: no candidates left, terminating".format(work_id))
                            else:
                                logger.info("Worker {}: no candidates left, terminating".format(work_id))
                            event.set()
                            return
                        time.sleep(0.5)  # wait for new candidates
                        continue
                    else:
                        g_vars.running_status[work_id] = 1
                        q_size = len(g_vars.candidate_queue)
                        chose_m = get_randint_by_exp_weights(q_size)
                        logger.debug("Worker {}: choose candidate {} from queue of size {}".format(work_id, chose_m, q_size))
                        var_ids_tuple, indispensable_ids, previous_criteria = g_vars.candidate_queue.pop(chose_m)
                        logger.debug("Worker {}: testing variants {}".format(work_id, var_ids_tuple))
                # only when the node is assigned can we know if another worker has already tested this,
                # so we have to include this in the worker function rather than the main function
                with g_vars.explored_tuples_lock:
                    if var_ids_tuple in g_vars.explored_tuples:
                        # this usually will not happend because we already check the explored_tuples when adding candidates
                        # to the queue, but we still need to check
                        logger.trace("  {} already explored".format(var_ids_tuple))
                        continue
                    else:
                        g_vars.explored_tuples[var_ids_tuple] = True
                
                prop, info, loglike, criteria_val = self.__compute_like_and_criteria(
                    chosen_id_list=var_ids_tuple, criteria=criterion)
                
                # !! deprecated: may drop the real components when there are equivalent models
                # drop zero prob variants
                # while len(var_ids_tuple) > 1:
                #     prev_var_ids = var_ids_tuple
                #     var_ids_tuple = self.__drop_zero_variants_new(
                #         sorted_chosen_ids=var_ids_tuple,
                #         representative_props=prop,
                #         diff_tolerance=diff_tolerance,
                #         indispensable_ids=indispensable_ids)
                #     if prev_var_ids != var_ids_tuple:
                #         # if the var_ids_tuple is changed, we need to recalculate the likelihood and criteria
                #         logger.debug("Dropped zero prob variant(s), now testing variants {}".format(var_ids_tuple))
                #         prop, info, loglike, criteria_val = self.__compute_like_and_criteria(
                #             chosen_id_list=var_ids_tuple, criteria=criterion)
                #     else:
                #         logger.trace("No zero prob variant dropped, using previous results")
                #         break
                
                # instead we add the drop_zero_variants() to add candidates to the queue
                if len(var_ids_tuple) > 1:
                    new_ids_tuple = self.__drop_zero_variants_new(
                        sorted_chosen_ids=var_ids_tuple,
                        representative_props=prop,
                        diff_tolerance=p_diff_tolerance,
                        indispensable_ids=indispensable_ids,
                        work_id=work_id)
                    if new_ids_tuple != var_ids_tuple and new_ids_tuple not in g_vars.explored_tuples:
                        with g_vars.candidate_queue_lock:
                            localized_queue = list(g_vars.candidate_queue)
                            # find insertion point using binary search
                            idx = bisect_left([item[2] for item in localized_queue], criteria_val)
                            # insert the new candidates at the right position
                            localized_queue[idx:idx] = [(new_ids_tuple, set(indispensable_ids), criteria_val)]
                            # make sure the queue size does not exceed the max_queue_size
                            if len(localized_queue) > max_queue_size:
                                g_vars.candidate_queue[:] = localized_queue[:max_queue_size]
                            else:
                                g_vars.candidate_queue[:] = localized_queue

                # compare the criteria with the best and record the results if it is better or equal to the best
                logger.debug(
                    "Worker {}: Test variants [{}]: criterion={}, loglike={}"
                        .format(work_id, ", ".join([self.__str_rep_id(_c_i)
                                                    for _c_i in var_ids_tuple]),
                                criteria_val, loglike))
                with g_vars.best_lock:
                    if criteria_val < g_vars.best_criteria.value - c_diff_tolerance:
                        # if the criteria is better than the best, update the best
                        g_vars.best_criteria.value = criteria_val
                        g_vars.best_models.clear()
                        g_vars.best_models[var_ids_tuple] = [prop, info, loglike, criteria_val]
                        if self.bootstrap_mode:
                            # just report the current best var_ids_tuple, not the whole model result
                            logger.debug("Worker {}: Current best criteria {}; best model: {}".format(work_id, criteria_val, [var_ids_tuple]))
                        else:
                            logger.info("Worker {}: Current best criteria {}; best model: {}".format(work_id, criteria_val, [var_ids_tuple]))
                        g_vars.count_unchanged.value = 0
                    elif abs(criteria_val - g_vars.best_criteria.value) < c_diff_tolerance:
                        # if the criteria is equal to the best, add it to the best models
                        # logger.info("Worker {}: Previous best criteria {}; best models: {}".format(work_id, g_vars.best_criteria.value, g_vars.best_models))
                        g_vars.best_models[var_ids_tuple] = [prop, info, loglike, criteria_val]
                        # logger.info("==============")
                        # logger.info("Worker {}: current var_ids_tuple: {} added to best models".format(work_id, var_ids_tuple))
                        # logger.info("Worker {}: Current best models: {}".format(work_id, list(g_vars.best_models.keys())))
                        if self.bootstrap_mode:
                            # just report the current best var_ids_tuple, not the whole model result
                            logger.debug("Worker {}: Current best criteria {}; best models: {}".format(work_id, g_vars.best_criteria.value, list(g_vars.best_models.keys())))
                        else:
                            logger.info("Worker {}: Current best criteria {}; best models: {}".format(work_id, g_vars.best_criteria.value, list(g_vars.best_models.keys())))
                        g_vars.count_unchanged.value = 0
                    else:
                        # if the criteria is worse than the best, skip the current result
                        with g_vars.count_unchanged_lock:
                            g_vars.count_unchanged.value += 1
                            if g_vars.count_unchanged.value >= max_unchanged:
                                # if we did not improve the criteria for too many times, stop
                                if self.bootstrap_mode:
                                    logger.debug("Worker {}: Hit unchanged {} times, stopping".format(work_id, g_vars.count_unchanged.value))
                                else:
                                    logger.info("Worker {}: Hit unchanged {} times, stopping".format(work_id, g_vars.count_unchanged.value))
                                event.set()
                                return

                # generate new candidates for the next step
                # if len(g_vars.candidate_queue) < max_queue_size or criteria_val < g_vars.candidate_queue[-1][2]:
                    
                # if the queue is not full or the new criteria is better than the currently-worset one,
                # we can generate new candidates
                # we will generate new candidates by dropping one variant at a time
                # and check if the remaining variants cover all observed subpaths
                # if so, we can add the new candidate to the queue
                # otherwise, we will not add it to the queue
                var_ids_size = len(var_ids_tuple)
                candidate_tuples_here = []
                if var_ids_size == 1:
                    count_new_indispensable = 1
                else:
                    indispensable_ids = set(indispensable_ids)  # make a copy of the indispensable ids
                    count_new_indispensable = 0
                    if var_ids_size > random_size > 0:
                        drop_pool = np.random.choice(range(var_ids_size), size=random_size, replace=False).tolist()
                    else:
                        drop_pool = range(var_ids_size)
                    for go_drop in drop_pool:
                        # generate a new candidate by dropping one variant at a time
                        new_var_ids_tuple = var_ids_tuple[:go_drop] + var_ids_tuple[go_drop+1:]
                        # check if the new candidate covers all observed subpaths
                        if self.cover_all_observed_subpaths(new_var_ids_tuple, cache_res=g_vars.cache_cover):
                            if new_var_ids_tuple in g_vars.explored_tuples:
                                # check if it was already explored
                                # do it after cover_all_observed_subpaths() to count the new indispensable ids
                                continue
                            else:
                                # if the new candidate is not explored, we can add it to the queue
                                candidate_tuples_here.append(new_var_ids_tuple)
                        else:                        
                            # the negative result of the cover_all_observed_subpaths()
                            # meaning that a var_id is indispensable,
                            # will be hard to be transferred to the previous node 
                            # and impossible to be applied to the indispensable_ids of sibling nodes
                            # so we have to check it when we are trying to create a candidate for next step, rather than starting a new test
                            indispensable_ids.add(var_ids_tuple[go_drop])
                            count_new_indispensable += 1
                            g_vars.explored_tuples[new_var_ids_tuple] = True
                if count_new_indispensable == var_ids_size:
                    # if all variants in the current candidate are indispensable,
                    # it means we cannot generate any new candidates, meaning hit the end of the search
                    with g_vars.count_hit_end_lock:
                        g_vars.count_hit_end.value += 1
                    logger.trace("  all variants are indispensable, no new candidates generated")
                    if g_vars.count_hit_end.value >= max_end_hits:
                        # if we hit the end of the search for too many times, we will stop the search
                        logger.info("Hit end of search {} times, stopping".format(g_vars.count_hit_end.value))
                        event.set()
                    continue
                # if there are no new candidates, we will not add it to the queue
                if not candidate_tuples_here:
                    logger.trace("  no new candidates generated")
                    continue
                # otherwise, we will add the new candidates to the queue
                logger.trace("  {} new candidates generated".format(len(candidate_tuples_here)))
                candidate_tuples_here.sort()  # sort the candidates to make sure the order is consistent
                with g_vars.candidate_queue_lock:
                    localized_queue = list(g_vars.candidate_queue)
                    # find insertion point using binary search
                    idx = bisect_left([item[2] for item in localized_queue], criteria_val)
                    # insert the new candidates at the right position
                    localized_queue[idx:idx] = [
                        (new_var_ids_tuple, set(indispensable_ids), criteria_val)
                        for new_var_ids_tuple in candidate_tuples_here]
                    # make sure the queue size does not exceed the max_queue_size
                    if len(localized_queue) > max_queue_size:
                        g_vars.candidate_queue[:] = localized_queue[:max_queue_size]
                    else:
                        g_vars.candidate_queue[:] = localized_queue

                    with g_vars.count_unchanged_lock:
                        if self.bootstrap_mode:
                            logger.trace("Worker {}: Queue size={}, End hits={}, Unchanged={}".format(
                                work_id, len(g_vars.candidate_queue),
                                g_vars.count_hit_end.value, g_vars.count_unchanged.value))
                        else:
                            logger.debug("Worker {}: Queue size={}, End hits={}, Unchanged={}".format(
                                work_id, len(g_vars.candidate_queue),
                                g_vars.count_hit_end.value, g_vars.count_unchanged.value))

        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            tb = traceback.format_exception(exc_type, exc_value, exc_traceback)
            location = traceback.extract_tb(exc_traceback)[-1]
            error_queue.put((e, tb, location))
            event.set()
            return
    
    def __test_one_drop_worker(
            self,
            # this_rd_ids,
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
            #### - discard
            # use the worker id to pick a unique rd_id for each job,
            # then use the unique rd_id to randomly pick a var_id from the sorted_chosen_ids
            # var_id = sorted_chosen_ids[this_rd_ids[w_id]]
            ####
            var_id = sorted_chosen_ids[w_id]
            if var_id not in g_vars.indispensable_ids:
                testing_ids = chosen_ids_set - {var_id}
                if self.cover_all_observed_subpaths(testing_ids, g_vars.cache_cover):
                    # logger.debug(
                    #     "Test variants [{}] - {}".
                    #         format(", ".join([self.__str_rep_id(_c_i)
                    #                           for _c_i in sorted_chosen_ids]), self.__str_rep_id(var_id)))
                    tuple_testing_ids = tuple(sorted(testing_ids))
                    if tuple_testing_ids in g_vars.cache_estimated:
                        logger.debug("Using cached result for {} variants".format(len(testing_ids)))
                        res_list = g_vars.cache_estimated[tuple_testing_ids]
                    else:
                        # compute the likelihood and criteria
                        res_list = self.__compute_like_and_criteria(chosen_id_list=tuple_testing_ids, criteria=criterion)
                        # store the result in the global_vars
                        g_vars.cache_estimated[tuple_testing_ids] = res_list
                    lock.acquire()
                    g_vars.recorded_ids.append(var_id)
                    g_vars.prop.append(res_list[0])
                    g_vars.echo.append(res_list[1])
                    g_vars.loglike.append(res_list[2])
                    g_vars.criterion.append(res_list[3])
                    lock.release()
                    logger.debug(
                        "Test variants [{}] - {}: criterion = {}, loglike = {}"
                            .format(", ".join([self.__str_rep_id(_c_i)
                                               for _c_i in sorted_chosen_ids]), self.__str_rep_id(var_id),
                                    res_list[3], res_list[2]))
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
        if len(sorted_chosen_ids) == g_vars.finished_w:
            event.set()

    def __compute_like_and_criteria(self, chosen_id_list: Union[List, Tuple], criteria, quiet=False):
        # logger.debug("Generating the likelihood function .. ")
        neg_loglike_func_obj = self.get_neg_likelihood_of_var_freq(within_variant_ids=set(chosen_id_list))
        # if self.bootstrap_mode or quiet:
        #     logger.debug("Maximizing the likelihood function for {} variants".format(len(chosen_id_set)))
        # else:
        #     logger.info("Maximizing the likelihood function for {} variants".format(len(chosen_id_set)))
        success_run = minimize_neg_likelihood(
            neg_loglike_func=neg_loglike_func_obj.loglike_func,
            num_variables=len(chosen_id_list),
            verbose=self.loglevel in ("TRACE", "ALL"))
        return self.__summarize_like_and_criteria(success_run, chosen_id_list, criteria, neg_loglike_func_obj)

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

    def __summarize_like_and_criteria(self, success_run, sorted_chosen_id:Union[Tuple, List], criteria, neg_loglike_func_obj):
        """
        ---
        Returns
        : use_prop: OrderedDict
        : echo_prop: OrderedDict
            echo_prop can be different from use_prop in values because it uses the sum of unidentifiable variants
        : this_like: float
        : this_criteria: float
        """
        if success_run:
            # this_prop = list(success_run.x)
            this_like = -success_run.fun
            use_prop, echo_prop = self.__summarize_run_prop(success_run, sorted_chosen_id)
            logger.debug("Proportions: " + ", ".join(["%s:%.4f" % (_id, _p) for _id, _p, in echo_prop.items()]))
            logger.debug("Log-likelihood: %s" % this_like)
            if criteria == "AIC":
                logger.debug("len_param: %s" % neg_loglike_func_obj.variable_size)
                this_criteria = aic(
                    loglike=this_like,
                    len_param=neg_loglike_func_obj.variable_size)
                logger.debug("%s: %s" % (criteria, this_criteria))
            elif criteria == "BIC":
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

    # def __summarize_run_prop(self, success_run, sorted_var_ids: Union[List, Tuple]):
    #     prop_dict = {}
    #     representatives = [rep_id for rep_id in sorted_var_ids if rep_id in self.repr_to_merged_variants]
    #     echo_prop = OrderedDict()
    #     for go, this_prop in enumerate(success_run.x):
    #         echo_prop[self.__str_rep_id(representatives[go])] = this_prop
    #         unidentifiable_var_ids = self.repr_to_merged_variants[representatives[go]]
    #         this_prop /= len(unidentifiable_var_ids)
    #         for cid_var_id in unidentifiable_var_ids:
    #             prop_dict[cid_var_id] = this_prop
    #     use_prop = OrderedDict([(_id, prop_dict[_id]) for _id in sorted(prop_dict)])
    #     return use_prop, echo_prop
    # update 20250704, do not evenly split unidentifiable variants, leave them combined in the representative
    def __summarize_run_prop(self, success_run, sorted_var_ids: Union[List, Tuple]):
        prop_dict = {}
        representatives = [rep_id for rep_id in sorted_var_ids if rep_id in self.repr_to_merged_variants]
        echo_prop = OrderedDict()
        for go, this_prop in enumerate(success_run.x):
            # TODO: now echo_prop can be skipped to return, only use_prop is needed
            #       echo_prop can be directly generated upon using
            echo_prop[self.__str_rep_id(representatives[go])] = this_prop
            prop_dict[representatives[go]] = this_prop
            # unidentifiable_var_ids = self.repr_to_merged_variants[representatives[go]]
            # this_prop /= len(unidentifiable_var_ids)
            # for cid_var_id in unidentifiable_var_ids:
            #     prop_dict[cid_var_id] = this_prop
        use_prop = OrderedDict([(_id, prop_dict[_id]) for _id in sorted(prop_dict)])
        return use_prop, echo_prop

    def __str_rep_id(self, rep_id):
        return "+".join([f"cid_{_cid_var_id}" for _cid_var_id in self.repr_to_merged_variants[rep_id]])

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

    def cover_all_observed_subpaths(self, sorted_variant_id: Tuple, cache_res=None):
        if not self.observed_sbp_id_set:
            self.update_observed_sp_ids()

        if cache_res is not None and sorted_variant_id in cache_res:
            return cache_res[sorted_variant_id]
        else:
            model_sp_ids = set()
            for go_var in sorted_variant_id:
                for sub_path in self.variant_readpath_counters[self.variant_paths[go_var]]:
                    if sub_path in self.sbp_to_sbp_id:
                        # if sub_path was not dropped after the construction of self.variant_readpath_counters
                        model_sp_ids.add(self.sbp_to_sbp_id[sub_path])
            if self.observed_sbp_id_set.issubset(model_sp_ids):
                if cache_res is not None:
                    cache_res[sorted_variant_id] = True
                return True
            else:
                if cache_res is not None:
                    cache_res[sorted_variant_id] = False
                return False
            
    # def cover_all_observed_subpaths(self, variant_ids):
    #     if not self.observed_sbp_id_set:
    #         self.update_observed_sp_ids()
    #     model_sp_ids = set()
    #     for go_var in variant_ids:
    #         for sub_path in self.variant_readpath_counters[self.variant_paths[go_var]]:
    #             if sub_path in self.sbp_to_sbp_id:
    #                 # if sub_path was not dropped after the construction of self.variant_readpath_counters
    #                 model_sp_ids.add(self.sbp_to_sbp_id[sub_path])
    #     return self.observed_sbp_id_set.issubset(model_sp_ids)
            


