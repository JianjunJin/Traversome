

# Traversome (Under active development)
Genomic isomer frequency estimation from genome assembly graphs and long reads.


### Install dependencies
```bash
conda install typer numpy pandas scipy pymc3 sympy loguru -c conda-forge
conda install python-symengine -c symengine -c conda-forge
```

### Install devel version of traversome
```bash
git clone --depth=1 https://github.com/Kinggerm/Traversome
pip install -e ./Traversome --no-deps
```

### Command line interface (CLI)

```bash
traversome ml -g graph.gfa -a align.gaf -o outdir
traversome mc -g graph.gfa -a align.gaf -o outdir (pymc3 crashes when the graph becomes complicated)
```

### Interpreting results
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
