

# Traversome

Genomic structure frequency estimation from traversal paths of long reads mapped to genome assembly graphs.

### Assembly graphs
...

### Graph Alignment
...

### Traversal information
*Traversome* uses a heuristic approach to propose different genome
structures based on a seed and extend model using information from
the traversal paths of long reads mapped to an assembly graph. Given
the start and end position of a mapped read, and its length, the 
number of times that this read can traverse a repetitive region of
the assembly graph is contrained, and can be estimated...

### Structural variation frequency estimation
*Traversome* fits a statistical model to estimate the frequencies of 
different genome structural variants in a sample based on the ...

### Implementation
*traversome* is written as both a command line tool and an interactive
Python API. The former allows users to easily run and automate 
computationally intensive steps, while the latter offers a number 
of useful tools for interactively visualizing and assessing input 
data and results.

### Developers/Contributors
- [Jianjun Jin](https://github.com/JianjunJin), Columbia University
- [Deren Eaton](https://github.com/eaton-lab), Columbia University



