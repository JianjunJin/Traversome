

# Traversome (Underdevelopment)
Genomic isomer frequency estimation from genome assembly graphs and long reads.


### Install dependencies
```bash
conda install typer numpy pandas scipy pymc3 sympy matplotlib -c conda-forge
```

### Install devel version of traversome
```bash
git clone -b modular https://github.com/Kinggerm/Traversome
pip install -e . --no-deps
```

### Command line interface (CLI)

Temporary usage:
```bash
python traversome/estimate_isomer_frequencies.py -g graph.gfa -a align.gaf -o .
```

future: 
```bash
traversome -g graph.gfa -a align.gaf -o .
```

### Interpreting results
...


### (Advanced) Bayesian estimation
...


## Development

```
# workflow

|-- __main__.py
|-- traversome.py
    |-- __init__.py
    |-- SimpleAssembly.py
    |-- Assembly.py
    |-- GraphAlignRecords.py
    |-- CleanGraph.py (still working)
    |-- EstCopyDepthFromCov.py
    |-- EstCopyDepthPrecise.py (still working)
    |-- GraphAlignmentPathGenerator.py
    |-- GraphOnlyPathGenerator.py
    |-- ModelFitBayesian.py
    |-- ModelFitMaxLike.py
