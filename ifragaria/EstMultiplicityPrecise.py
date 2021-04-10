#!/usr/bin/env python

"""
Maximum Likelihood Optimization
"""

from loguru import logger
from itertools import product
from ifragaria.utils import ProcessingGraphFailed, INF, reduce_list_with_gcd, weighted_mean_and_std
from scipy import optimize
from copy import deepcopy
import sympy
import numpy as np
import random



ECHO_DIRECTION = ["_tail", "_head"]



class EstMultiplicityPrecise(object):
    """

    """
    def __init__(
        self, 
        graph,
        maximum_copy_num=8,
        broken_graph_allowed=False, 
        return_new_graphs=False,
        do_least_square=True,
        label="target",
        debug=False):

        # link to data from the graph
        self.graph = graph
        self.vertex_info = self.graph.vertex_info

        # store params
        self.maximum_copy_num = maximum_copy_num
        self.broken_graph_allowed = broken_graph_allowed
        self.return_new_graphs = return_new_graphs
        self.do_least_square = do_least_square
        self.label = label
        self.debug = debug  # pass to optimizer

        # attrs to fill
        self.vertlist = sorted(self.vertex_info)
        self.vertex_to_symbols = {i: sympy.Symbol("V{}".format(i), integer=True) for i in self.vertlist}
        self.symbols_to_vertex = {self.vertex_to_symbols[i]: i for i in self.vertlist}
        self.free_copy_variables = []
        self.extra_str_to_symbol = {}
        self.extra_symbol_to_str = {}
        self.formulae = []
        self.all_v_symbols = []
        self.all_symbols = []
        self.copy_solution = {}



    def run(self):
        """
        ...        

        Returns:
        --------
        None if 'return_new_graphs' = False, else it returns a list of dicts.
        """
        # bail out if vert list len is 1
        if len(self.vertlist) == 1:
            cov_ = self.vertex_info[self.vertlist[0]].cov
            logger.debug("Avg {} kmer-coverage = {}".format(self.label, round(cov_, 2)))
            if self.return_new_graphs:
                return [{"graph": deepcopy(self.graph), "cov": cov_}]
            return

        # the core function calls 
        self.get_max_multiplicity()
        self.build_formulae()
        self.add_self_loop_formulae()
        self.add_limit_formulae()

        self.sympy_solve_equations()
        if self.return_new_graphs:
            return self.optimize_model()
        else:
            self.optimize_model()



    def get_formula(self, from_vertex, from_end, back_to_vertex, back_to_end, here_record_ends):
        """
        Dynamic function, very cool!
        Used to get a formula for ...

        from_vertex (str): vertex name at start
        from_end (bool): head or tail 
        back_to_vertex (str): vertex name at end
        back_to_end (bool): head or tail
        here_record_ends (set): ...
        """
        # start with the symbol for this vertex (e.g., "1")
        result_form = self.vertex_to_symbols[from_vertex]

        # this set contains tuples of (vname, bool) for where edges end.
        # start by adding self to it: (from_vertex, from_end)
        here_record_ends.add((from_vertex, from_end))
        
        # if looped so that from_vertex == back_to_vertex skip it
        if from_vertex != back_to_vertex:
            
            # iterate over all connections from 'from_vertex'
            for (next_v, next_e) in self.vertex_info[from_vertex].connections[from_end]:
                
                # if next vertex is a loop back to 'from_vertex' then also skip
                if (next_v, next_e) == (from_vertex, not from_end):
                    pass              

                # next vertex, edge is new, so incorporate to formula
                elif (next_v, next_e) not in here_record_ends:
                    form = self.get_formula(next_v, next_e, from_vertex, from_end, here_record_ends)
                    result_form -= form

        return result_form



    def path_without_leakage(self, start_v, start_e, terminating_end_set, terminator):
        """
        Called within self.is_sequential_repeat()

        ... searches for path from starting vertex to terminating end
        by following a path different from starting edge?          
        """
        in_pipe_leak = False
        circle_in_between = []
        in_vertex_ends = set()
        in_vertex_ends.add((start_v, start_e))
        in_searching_con = [(start_v, not start_e)]
        
        # search for connection between starting vertex and not starting edge
        while in_searching_con:
            
            # vertex name, bool
            in_search_v, in_search_e = in_searching_con.pop(0)
            
            # terminating end set is either connection_set_t or _f depending...
            if (in_search_v, in_search_e) in terminating_end_set:
                # start from the same (next_t_v, next_t_e), merging to two different ends of connection_set_f
                if circle_in_between:
                    in_pipe_leak = True
                    break
                else:
                    circle_in_between.append(((start_v, start_e), (in_search_v, in_search_e)))
        
            # terminator is currently always connection_set_t
            elif (in_search_v, in_search_e) in terminator:
                in_pipe_leak = True
                break
        
            # 
            else:
                for n_in_search_v, n_in_search_e in self.vertex_info[in_search_v].connections[in_search_e]:
                    if (n_in_search_v, n_in_search_e) in in_vertex_ends:
                        pass
                    else:
                        in_vertex_ends.add((n_in_search_v, n_in_search_e))
                        in_searching_con.append((n_in_search_v, not n_in_search_e))

        # return empty if there is a leak, else the circle in between x
        if not in_pipe_leak:
            return circle_in_between
        else:
            return []



    def is_sequential_repeat(self, search_vertex_name, return_pair_in_the_trunk_path=True):
        """
        Called in several places...
        Returns an iterable
        """
        # search vertex name
        if search_vertex_name not in self.vertex_info:
            raise ProcessingGraphFailed("Vertex name {} not found!".format(search_vertex_name))

        # get connections leading from this vertex
        connection_set_t = self.vertex_info[search_vertex_name].connections[True]
        connection_set_f = self.vertex_info[search_vertex_name].connections[False]

        # if both connection sets do not have 2 then return empty
        if not (len(connection_set_t) == len(connection_set_f) == 2):
            return []

        # if both connection sets have 2 then store inner circles
        all_pairs_of_inner_circles = []

        # 
        for next_t_v, next_t_e in list(connection_set_t):

            # check leakage in forward direction
            this_inner_circle = self.path_without_leakage(next_t_v, next_t_e, connection_set_f, connection_set_t)
            if this_inner_circle:

                # check leakage in reverse direction
                reverse_v, reverse_e = this_inner_circle[0][1]
                not_leak = self.path_without_leakage(reverse_v, reverse_e, connection_set_t, connection_set_t)
                if not_leak:
                    all_pairs_of_inner_circles.extend(this_inner_circle)

        # sort pairs by average depths(?)
        all_pairs_of_inner_circles.sort(
            key=lambda x: (self.vertex_info[x[0][0]].cov + self.vertex_info[x[1][0]].cov))

        # 
        if all_pairs_of_inner_circles and return_pair_in_the_trunk_path:
            # switch nearby vertices
            # keep those prone to be located in the "trunk road" of the repeat
            single_pair_in_main_path = []
            if len(all_pairs_of_inner_circles) == 1:
                for next_v, next_e in list(connection_set_t) + list(connection_set_f):
                    if (next_v, next_e) not in all_pairs_of_inner_circles[0]:
                        single_pair_in_main_path.append((next_v, next_e))
                single_pair_in_main_path = tuple(single_pair_in_main_path)
            else:
                # two circles share this sequential repeat,
                # return the one with a smaller average depth(?)
                single_pair_in_main_path = tuple(all_pairs_of_inner_circles[0])
            return single_pair_in_main_path
        return all_pairs_of_inner_circles



    def get_max_multiplicity(self):
        """
        Finds the maximum copy number based on all contig coverages
        """
        # reduce maximum_copy_num to reduce computational burden
        all_coverages = [self.vertex_info[i].cov for i in self.vertlist]
        self.maximum_copy_num = min(
            self.maximum_copy_num, 
            int(2 * np.ceil(max(all_coverages) / min(all_coverages)))
            # add the latter one to avoid the influence of bloated self.maximum_copy_num
        )
        logger.info("Maximum multiplicity: {}".format(self.maximum_copy_num))



    def build_formulae(self):
        """
        Traverses edges between vertices to build formulae for relationship between contig copies
        which will then reduced by sympy
        """

        # list to store formulae
        formulae = []

        # keep track of which verts ends have been visited
        recorded_ends = set()

        # iterate over vertices and ends
        for vname in self.vertlist:
            for this_end in (True, False):

                # if this combination of (name, end) has not yet been processed
                if (vname, this_end) not in recorded_ends:
                    
                    # store this combo
                    recorded_ends.add((vname, this_end))
                    
                    # if this vert connects at this end to something...
                    if self.vertex_info[vname].connections[this_end]:

                        # get the symbol for this vertex not yet formulized
                        this_formula = self.vertex_to_symbols[vname]
                        formulized = False
                        
                        # get vertex, edge
                        for n_v, n_e in self.vertex_info[vname].connections[this_end]:
                            if (n_v, n_e) not in recorded_ends:

                                # try to formulize symbol 
                                try:
                                    form = self.get_formula(n_v, n_e, vname, this_end, recorded_ends)
                                    logger.warning(form)
                                    this_formula -= form
                                    formulized = True
                                    logger.debug('formula: {} - {} = {}'.format(this_formula, this_formula, this_formula))

                                except RecursionError:
                                    logger.exception(
                                        "formulating for: {}{} -> {}{} failed"
                                        .format(n_v, ECHO_DIRECTION[n_e], vname, ECHO_DIRECTION[this_end])
                                    )
                                    raise ProcessingGraphFailed("RecursionError!")
                        

                        logger.info("formula for {}{} = {}"
                            .format(vname, ECHO_DIRECTION[this_end], this_formula))

                        if formulized:
                            formulae.append(this_formula)

                    elif self.broken_graph_allowed:
                        # Extra limitation to force terminal vertex to have only one copy, to avoid over-estimation
                        # Under-estimation would not be a problem here,
                        # because the True-multiple-copy vertex would simply have no other connections,
                        # or failed in the following estimation if it does
                        formulae.append(self.vertex_to_symbols[vname] - 1)

        # store formulae to self
        self.formulae = formulae
        logger.info('formulae step 1: {}'.format(self.formulae))



    def add_self_loop_formulae(self):
        """

        """
        # add self-loop formulae
        for vname in self.vertlist:
            if self.vertex_info[vname].is_self_loop():
                logger.warning("Self-loop contig detected: Vertex_{}".format(vname))

                # create pseudo self loop vertex
                pseudo_self_loop_str = "P" + vname

                # check if it is already in extra dict
                if pseudo_self_loop_str not in self.extra_str_to_symbol:

                    # store symbol and psuedo vertex name in dict and revdict
                    self.extra_str_to_symbol[pseudo_self_loop_str] = sympy.Symbol(pseudo_self_loop_str, integer=True)
                    self.extra_symbol_to_str[self.extra_str_to_symbol[pseudo_self_loop_str]] = pseudo_self_loop_str

                # subtract psuedovertex from this vertex
                this_formula = self.vertex_to_symbols[vname] - self.extra_str_to_symbol[pseudo_self_loop_str]
                
                # store this to formulae
                self.formulae.append(this_formula)
                logger.info("formulating for: {}{}:{}".format(vname, ECHO_DIRECTION[True], this_formula))



    def add_limit_formulae(self):
        """

        """
        # add following extra limitation
        # set cov_sequential_repeat = x*near_by_cov, x is an integer
        for vname in self.vertlist:

            # if this is a 
            single_pair_in_the_trunk_path = self.is_sequential_repeat(vname)
            logger.warning(single_pair_in_the_trunk_path)

            # if it is a sequential repeat
            if single_pair_in_the_trunk_path:

                # get the vertex, edge pairs for both sequential things
                (from_v, from_e), (to_v, to_e) = single_pair_in_the_trunk_path

                # from_v and to_v are already in the "trunk path", if they are the same,
                # the graph is like two circles sharing the same sequential repeat, no need to add this limitation
                if from_v != to_v:

                    # create a new extra symbol
                    new_str = "E" + str(len(self.extra_str_to_symbol))
                    self.extra_str_to_symbol[new_str] = sympy.Symbol(new_str, integer=True)
                    self.extra_symbol_to_str[self.extra_str_to_symbol[new_str]] = new_str

                    # write as a formula and add to formulae list
                    this_formula = (
                        self.vertex_to_symbols[vname] -
                        self.vertex_to_symbols[from_v] * self.extra_str_to_symbol[new_str]
                    )
                    self.formulae.append(this_formula)
                    logger.info("formulating for: {}:{}".format(vname, this_formula))

        # get all symbols
        self.all_v_symbols = list(self.symbols_to_vertex)
        self.all_symbols = self.all_v_symbols + list(self.extra_symbol_to_str)
        logger.info("formulae: " + str(self.formulae))



    def sympy_solve_equations(self):
        """
        Uses sympy algebra to reduce the equation to solve.
        """
        # solve the equations or replace with empty dict
        self.copy_solution = sympy.solve(self.formulae, self.all_v_symbols)
        self.copy_solution = self.copy_solution if self.copy_solution else {}
     
        # delete 0 containing set, even for self-loop vertex
        if isinstance(self.copy_solution, list):
            go_solution = 0
            while go_solution < len(self.copy_solution):
                if 0 in set(self.copy_solution[go_solution].values()):
                    del self.copy_solution[go_solution]
                else:
                    go_solution += 1
        logger.info("solved: {}".format(self.copy_solution))

        # check if anything is left after removing 0 sets
        if not self.copy_solution:
            raise ProcessingGraphFailed(
                "Incomplete/Complicated/Unsolvable {} graph (1)!"
                .format(self.label))

        # check if too many solutions
        elif isinstance(self.copy_solution, list):
            if len(self.copy_solution) > 2:
                raise ProcessingGraphFailed(
                    "Incomplete/Complicated/Unsolvable {} graph (2)!"
                    .format(self.label))
            else:
                self.copy_solution = self.copy_solution[0]

        # check for all variables in solution
        self.free_copy_variables = []
        for symbol_used in self.all_symbols:
            if symbol_used not in self.copy_solution:
                self.free_copy_variables.append(symbol_used)
                self.copy_solution[symbol_used] = symbol_used

        logger.debug("copy equations: " + str(self.copy_solution))
        logger.debug("free variables: " + str(self.free_copy_variables))

        if self.do_least_square:
            # minimizing equation-based copy's deviations from coverage-based copy values.
            least_square_expr = 0
            for symbol_used in self.all_v_symbols:

                # get equation based copy and coverage based copy
                this_vertex = self.symbols_to_vertex[symbol_used]
                this_copy = self.graph.vertex_to_float_copy[this_vertex]

                # get the sum of squared differences
                least_square_expr += (self.copy_solution[symbol_used] - this_copy) ** 2  # * self.vertex_info[this_vertex]["len"]

            # make this into a sympy equation
            core_function = sympy.lambdify(args=self.free_copy_variables, expr=least_square_expr)
            logger.debug("least squares equation: {}".format(least_square_expr))

        else:
            # do normal likelihood
            # approximately calculate normal distribution scale
            sample_means = []
            sample_weights = []
            for symbol_used in self.all_v_symbols:
                this_vertex = self.symbols_to_vertex[symbol_used]
                sample_means.append(self.graph.vertex_to_float_copy[this_vertex])
                sample_weights.append(self.vertex_info[this_vertex].len)
            sample_mean_mean, sample_mean_var = weighted_mean_and_std(sample_means, sample_weights)
            global_scale = sample_mean_var * np.average(sample_weights) ** 0.5

            normal_neg_loglike_expr = 0
            for symbol_used in self.all_v_symbols:
                # get equation based copy and coverage based copy
                this_vertex = self.symbols_to_vertex[symbol_used]
                this_copy = self.graph.vertex_to_float_copy[this_vertex]

                # get the sum of log likes
                loc = self.copy_solution[symbol_used]
                scale = global_scale / (self.copy_solution[symbol_used] * self.vertex_info[this_vertex].len) ** 0.5
                normal_neg_loglike_expr += 1/(2 * scale) * (loc - this_copy) ** 2 + 1/2 * np.log(scale)
                # + 0.5 * np.log(2 * np.pi)

            # make this into a sympy equation
            core_function = sympy.lambdify(args=self.free_copy_variables, expr=normal_neg_loglike_expr)
            logger.debug("normal neg-log-likelihood equation: {}".format(normal_neg_loglike_expr))

        # for safe running
        if len(self.free_copy_variables) > 10:
            raise ProcessingGraphFailed("Free variable > 10 is not accepted yet!")

        # for compatibility between scipy and sympy
        def core_function_v(x):
            return core_function(*tuple(x))

        # store these for now
        self.core_function = core_function
        self.core_function_v = core_function_v


    def optimize_model(self):
        """
        Uses scipy to optimize parameters of the model to fit the data.

        model with discrete parameters are difficult to fit
        """
        # If the number of free variables is small then use brute force
        max_cn = self.maximum_copy_num
        if self.maximum_copy_num ** len(self.free_copy_variables) < 5E6:
            # sometimes, SLSQP ignores bounds and constraints
            copy_results = self.minimize_brute_force(
                func=self.core_function_v,
                range_list=[range(1, max_cn + 1)] * len(self.free_copy_variables),
                constraint_list=({'type': 'ineq', 'fun': self.__constraint_min_function_for_customized_brute},
                                 {'type': 'eq', 'fun': self.__constraint_int_function},
                                 {'type': 'ineq', 'fun': self.__constraint_max_function}),
            )
        elif 3 ** len(self.free_copy_variables) < 1E7:

        # large number of variables so we will use SLSQP, which sometimes have abnormal behaviours...
        else:
            constraints = ({'type': 'ineq', 'fun': self.__constraint_min_function},
                           {'type': 'eq', 'fun': self.__constraint_int_function},
                           {'type': 'ineq', 'fun': self.__constraint_max_function})
            copy_results = set()
            best_fun = INF

            # fit iteratively 
            for initial_copy in range(max_cn * 2 + 1):

                # select initial values based relative to max copy num
                if initial_copy < max_cn:
                    initials = np.array(
                        [initial_copy + 1] * len(self.free_copy_variables)
                    )
                elif initial_copy < max_cn * 2:
                    initials = np.array([
                        random.randint(1, max_cn)
                        ] * len(self.free_copy_variables)
                    )
                else:
                    initials = np.array([
                        self.graph.vertex_to_copy.get(
                            self.symbols_to_vertex.get(symb, False), 2)
                        for symb in self.free_copy_variables]
                    )

                # get bounds on parameters between (1, maxcopy)
                bounds = [(1, max_cn) for _ in range(len(self.free_copy_variables))]
                
                # try to fit the model but allow it to fail
                try:
                    copy_result = optimize.minimize(
                        fun=self.core_function_v,
                        x0=initials, 
                        jac=False,      # we could perhaps try to solve this?
                        method='SLSQP', 
                        bounds=bounds, 
                        constraints=constraints, 
                        options={'disp': self.debug, "maxiter": 100},
                    )
                except Exception:
                    continue

                # 
                if copy_result.fun < best_fun:
                    best_fun = round(copy_result.fun, 2)
                    copy_results = {tuple(copy_result.x)}
                elif copy_result.fun == best_fun:
                    copy_results.add(tuple(copy_result.x))
                else:
                    pass

            # report model fitting 
            logger.info("Best function value: " + str(best_fun))        
        
        # report results
        logger.info("Copy results: " + str(copy_results))
        
        # get results into a sorted list
        if len(copy_results) == 1:
            copy_results = list(copy_results)
        elif len(copy_results) > 1:
            # draftly sort results by freedom vertices
            copy_results = sorted(copy_results, key=lambda
                x: sum([(x[go_sym] - self.graph.vertex_to_float_copy[self.symbols_to_vertex[symb_used]]) ** 2
                        for go_sym, symb_used in enumerate(self.free_copy_variables)
                        if symb_used in self.symbols_to_vertex]))
        else:
            raise ProcessingGraphFailed(
                "Incomplete/Complicated/Unsolvable {} graph (3)!"
                .format(self.label)
            )

        # optionally return the new graph 
        if self.return_new_graphs:

            # produce all possible vertex copy combinations
            final_results = []
            all_copy_sets = set()

            # iterate over each ...
            for go_res, copy_result in enumerate(copy_results):
                
                # ...
                free_copy_variables_dict = {
                    self.free_copy_variables[i]: int(this_copy)
                    for i, this_copy in enumerate(copy_result)
                }

                # simplify copy values # 2020-02-22 added to avoid multiplicities res such as: [4, 8, 4]
                all_copies = []
                for this_symbol in self.all_v_symbols:
                    vertex_name = self.symbols_to_vertex[this_symbol]
                    this_copy = int(self.copy_solution[this_symbol].evalf(subs=free_copy_variables_dict, chop=True))
                    if this_copy <= 0:
                        raise ProcessingGraphFailed("Cannot identify copy number of " + vertex_name + "!")
                    all_copies.append(this_copy)
                
                # ...
                if len(all_copies) == 0:
                    raise ProcessingGraphFailed(
                        "Incomplete/Complicated/Unsolvable " + self.label + " graph (4)!")
                elif len(all_copies) == 1:
                    all_copies = [1]
                elif min(all_copies) == 1:
                    pass
                else:
                    new_all_copies = reduce_list_with_gcd(all_copies)
                    if self.debug and new_all_copies != all_copies:
                        logger.debug("Estimated copies: " + str(all_copies))
                        logger.debug("Reduced copies: " + str(new_all_copies))
                    all_copies = new_all_copies
                all_copies = tuple(all_copies)
                if all_copies not in all_copy_sets:
                    all_copy_sets.add(all_copies)
                else:
                    continue

                # record new copy values
                final_results.append({"graph": deepcopy(self.graph)})
                for go_s, this_symbol in enumerate(self.all_v_symbols):
                    vertex_name = self.symbols_to_vertex[this_symbol]
                    if vertex_name in final_results[go_res]["graph"].vertex_to_copy:
                        old_copy = final_results[go_res]["graph"].vertex_to_copy[vertex_name]
                        final_results[go_res]["graph"].copy_to_vertex[old_copy].remove(vertex_name)
                        if not final_results[go_res]["graph"].copy_to_vertex[old_copy]:
                            del final_results[go_res]["graph"].copy_to_vertex[old_copy]
                    this_copy = all_copies[go_s]
                    final_results[go_res]["graph"].vertex_to_copy[vertex_name] = this_copy
                    if this_copy not in final_results[go_res]["graph"].copy_to_vertex:
                        final_results[go_res]["graph"].copy_to_vertex[this_copy] = set()
                    final_results[go_res]["graph"].copy_to_vertex[this_copy].add(vertex_name)

                # re-estimate baseline depth
                total_product = 0.
                total_len = 0
                for vertex_name in self.vertlist:
                    this_len = (self.vertex_info[vertex_name].len - self.graph.overlap() + 1) \
                               * final_results[go_res]["graph"].vertex_to_copy.get(vertex_name, 1)
                    this_cov = self.vertex_info[vertex_name].cov \
                               / final_results[go_res]["graph"].vertex_to_copy.get(vertex_name, 1)
                    total_len += this_len
                    total_product += this_len * this_cov
                final_results[go_res]["cov"] = total_product / total_len
            return final_results


        # not returning the graph, so record new values to the original graph object
        else:
            # produce the first-ranked copy combination
            free_copy_variables_dict = {
                self.free_copy_variables[i]: int(this_copy)
                for i, this_copy in enumerate(copy_results[0])
            }

            # simplify copy values; avoid multiplicities res such as: [4, 8, 4]
            all_copies = []
            for this_symbol in self.all_v_symbols:
                vertex_name = self.symbols_to_vertex[this_symbol]
                this_copy = int(self.copy_solution[this_symbol].evalf(subs=free_copy_variables_dict, chop=True))
                if this_copy <= 0:
                    raise ProcessingGraphFailed("Cannot identify copy number of " + vertex_name + "!")
                all_copies.append(this_copy)

            # 
            if len(all_copies) == 0:
                raise ProcessingGraphFailed(
                    "Incomplete/Complicated/Unsolvable " + self.label + " graph (4)!")
            elif len(all_copies) == 1:
                all_copies = [1]
            elif min(all_copies) == 1:
                pass
            else:
                new_all_copies = reduce_list_with_gcd(all_copies)
                if self.debug and new_all_copies != all_copies:
                    logger.debug("Estimated copies: " + str(all_copies))
                    logger.debug("Reduced copies: " + str(new_all_copies))
                all_copies = new_all_copies

            # record new copy values
            for go_s, this_symbol in enumerate(self.all_v_symbols):
                vertex_name = self.symbols_to_vertex[this_symbol]
                if vertex_name in self.graph.vertex_to_copy:
                    old_copy = self.graph.vertex_to_copy[vertex_name]
                    self.graph.copy_to_vertex[old_copy].remove(vertex_name)
                    if not self.graph.copy_to_vertex[old_copy]:
                        del self.graph.copy_to_vertex[old_copy]
                this_copy = all_copies[go_s]
                self.graph.vertex_to_copy[vertex_name] = this_copy
                if this_copy not in self.graph.copy_to_vertex:
                    self.graph.copy_to_vertex[this_copy] = set()
                self.graph.copy_to_vertex[this_copy].add(vertex_name)


            # re-estimate baseline depth
            total_product = 0.
            total_len = 0
            overlap = self.graph.overlap() if self.graph.overlap() else 0
            for vertex_name in self.vertlist:
                this_len = (self.vertex_info[vertex_name].len - overlap + 1) \
                           * self.graph.vertex_to_copy.get(vertex_name, 1)
                this_cov = self.vertex_info[vertex_name].cov / self.graph.vertex_to_copy.get(vertex_name, 1)
                total_len += this_len
                total_product += this_len * this_cov
            new_val = total_product / total_len
            logger.debug("Average " + self.label + " kmer-coverage = " + str(round(new_val, 2)))






    def __constraint_min_function(self, x):
        """
        create constraints by creating inequations: the copy of every contig has to be >= 1
        """
        replacements = [(symbol_used, x[go_sym]) for go_sym, symbol_used in enumerate(self.free_copy_variables)]
        expression_array = np.array([self.copy_solution[this_sym].subs(replacements) for this_sym in self.all_symbols])
        min_copy = np.array([1.001] * len(self.all_v_symbols) + [2.001] * len(self.extra_symbol_to_str))
        # effect: expression_array >= int(min_copy)
        return expression_array - min_copy



    def __constraint_min_function_for_customized_brute(self, x):
        """

        """
        replacements = [
            (symbol_used, x[go_sym]) 
            for (go_sym, symbol_used) in enumerate(self.free_copy_variables)
        ]
        expression_array = np.array([
            self.copy_solution[this_sym].subs(replacements)
            for this_sym in self.all_symbols
        ])
        min_copy = np.array([1.0] * len(self.all_v_symbols) + [2.0] * len(self.extra_symbol_to_str))
        # effect: expression_array >= min_copy
        return expression_array - min_copy



    def __constraint_max_function(self, x):
        """

        """
        replacements = [(symbol_used, x[go_sym]) for go_sym, symbol_used in enumerate(self.free_copy_variables)]
        expression_array = np.array([self.copy_solution[this_sym].subs(replacements) for this_sym in self.all_symbols])
        max_copy = np.array([self.maximum_copy_num] * len(self.all_v_symbols) +
                            [self.maximum_copy_num * 2] * len(self.extra_symbol_to_str))
        # effect: expression_array <= max_copy
        return max_copy - expression_array



    def __constraint_int_function(self, x):
        replacements = [(symbol_used, x[go_sym]) for go_sym, symbol_used in enumerate(self.free_copy_variables)]
        expression_array = np.array([self.copy_solution[this_sym].subs(replacements) for this_sym in self.all_symbols])
        # diff = np.array([0] * len(all_symbols))
        return sum([abs(every_copy - int(every_copy)) for every_copy in expression_array])




    def minimize_brute_force(self,
        func, 
        range_list, 
        constraint_list, 
        round_digit=4, 
        ):

        best_fun_val = INF
        best_para_val = []
        count_round = 0
        count_valid = 0

        # 
        for value_set in product(*[list(this_range) for this_range in range_list]):
            count_round += 1
            is_valid_set = True
            for cons in constraint_list:
                if cons["type"] == "ineq":
                    try:
                        if (cons["fun"](value_set) < 0).any():
                            is_valid_set = False
                            break
                    except TypeError:
                        is_valid_set = False
                        break
                elif cons["type"] == "eq":
                    try:
                        if cons["fun"](value_set) != 0:
                            is_valid_set = False
                            break
                    except TypeError:
                        is_valid_set = False
                        break
            if not is_valid_set:
                continue
            count_valid += 1
            this_fun_val = round(func(value_set), round_digit)
            if this_fun_val < best_fun_val:
                best_para_val = [value_set]
                best_fun_val = this_fun_val
            elif this_fun_val == best_fun_val:
                best_para_val.append(value_set)
            else:
                pass

        logger.debug("Brute valid/candidate rounds: {}/{}".format(count_valid, count_round))
        logger.debug("Brute best function value: {}".format(best_fun_val))
        logger.debug("Best solution: {}".format(best_para_val))
        return best_para_val







