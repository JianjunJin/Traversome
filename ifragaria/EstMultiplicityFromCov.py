#!/usr/bin/env python

"""

"""

from loguru import logger


# mode defines the ... that will be used to set the 'max_majority_copy'
MODE = {
    "embplant_pt": 2,
    "other_pt": 10,
    "embplant_mt": 4,
    "embplant_nr": 2,
    "animal_mt": 4,
    "fungus_mt": 8,
    "all": 100,
}



class EstMultiplicityFromCov(object):
    """
    Use seq coverage data to estimate copy and depth.
    """
    def __init__(
        self, 
        graph, 
        verts=None,
        avgcov=None, 
        mode="all", 
        reinit=False, 
        ):

        # shorter variable names
        self.vinfo = graph.vertex_info
        self.vcopy = graph.vertex_to_copy
        self.copyv = graph.copy_to_vertex
        self.vfcopy = graph.vertex_to_float_copy
        self.overlap = (graph.overlap() if graph.overlap() else 0)

        # select either all or a subset of vertices and sort.
        if not verts:
            self.verts = sorted(graph.vertex_info)
        else:
            self.verts = sorted(self.verts)

        # control options
        self.avgcov = avgcov
        self.mode = mode
        self.reinit = reinit

        # datatype settings
        self.max_majority_copy = 100
        if self.mode in MODE:
            self.max_majority_copy = MODE[mode]

        # result is stored here
        self.copydepth = None
        

    def run(self):
        """
        Runs estimation functions and stores the result to .copydepth
        """
        # run functions        
        if self.reinit:
            self.initialize()
        if not self.avgcov:
            self.get_avg_cov()
        else:
            self.given_avg_cov()


    def initialize(self):
        """
        Resets the 'copy_to_vertex' and 'vertex_to_copy' dicts 
        of the Assembly object to empty dicts or to keys with vals=1
        """
        for vertex_name in self.verts:
            
            # check for vert in Assembly dict 'vertex_to_copy'
            if vertex_name in self.vcopy:
                
                # save the old copy
                old_copy = self.vcopy[vertex_name]

                # remove this vertex from dict 'copy_to_vertex'
                self.copyv[old_copy].remove(vertex_name)

                # set this vertex to 1 in 'vertex_to_copy'
                self.vcopy[vertex_name] = 1

                # same for float dict...
                self.vfcopy[vertex_name] = 1.

                # ensure that 'copy_to_vertex' has {1: {}} in it
                if 1 not in self.copyv:
                    self.copyv[1] = set()

                # add this vertex to as value to key 1 in 'copy_to_vertex' dict.
                self.copyv[1].add(vertex_name)



    def get_avg_cov(self):
        """
        If user provided avgcov then ... else ...
        """
        # store previous value in a set starting at zero
        previous_val = {0.}
        new_val = -1.
        min_average_depth = (
            0.9 * min([self.vinfo[vertex_n].cov for vertex_n in self.vinfo])
        )

        # get av...
        while 1:
            
            # store previous value
            previous_val.add(round(new_val, 5))
            
            # estimate baseline depth by iterating over all verts
            total_product = 0.
            total_len = 0
            for vertex_name in self.verts:

                # get values at this vertex
                this_len = (
                    (self.vinfo[vertex_name].len - self.overlap + 1) * \
                    self.vcopy.get(vertex_name, 1)
                )
                this_cov = self.vinfo[vertex_name].cov / self.vcopy.get(vertex_name, 1)

                # add to total product count
                total_len += this_len
                total_product += this_len * this_cov

            # calculate new value for average coverage
            new_val = max(total_product / total_len, min_average_depth)

            # adjust this_copy according to new baseline depth
            for vertex_name in self.vinfo:
                
                # if vertex is in 'vertex_to copy' dict then remove it.
                if vertex_name in self.vcopy:

                    # get value from vcopy, remove it from copyv values
                    old_copy = self.vcopy[vertex_name]
                    self.copyv[old_copy].remove(vertex_name)

                    # if copyv value is now empty at this key then remove it
                    if not self.copyv[old_copy]:
                        del self.copyv[old_copy]

                # get and store values inside min max bounds
                this_float_copy = self.vinfo[vertex_name].cov / new_val
                this_copy = min(max(1, int(round(this_float_copy, 0))), self.max_majority_copy)
                self.vfcopy[vertex_name] = this_float_copy
                self.vcopy[vertex_name] = this_copy

                # create empty set as value if this copy not in copyv
                if this_copy not in self.copyv:
                    self.copyv[this_copy] = set()
                self.copyv[this_copy].add(vertex_name)

            # if new_val is in previous_val then break
            if round(new_val, 5) in previous_val:
                break
        
        # log result
        logger.info(
            "updating average {} {} {}\n".format(
                self.mode, 
                ("kmer-coverage:" if bool(self.overlap) else "coverage:"),
                round(new_val, 2),
            )
        )
        self.copydepth = new_val



    def given_avg_cov(self):
        """
        User DID provide an average depth estimate so...
        """
        # iterate over vertices and 
        for vertex_name in self.verts:

            # if vertex is in 'to copy' then ...
            if vertex_name in self.vcopy:
                old_copy = self.vcopy[vertex_name]
                self.copyv[old_copy].remove(vertex_name)
                if not self.copyv[old_copy]:
                    del self.copyv[old_copy]

            # ...
            this_float_copy = self.verts[vertex_name].cov / self.avgcov
            this_copy = min(max(1, int(round(this_float_copy, 0))), self.max_majority_copy)

            self.vcopy[vertex_name] = this_float_copy
            self.vcopy[vertex_name] = this_copy
            if this_copy not in self.copyv:
                self.copyv[this_copy] = set()
            self.copyv[this_copy].add(vertex_name)

        self.copydepth = self.avgcov

