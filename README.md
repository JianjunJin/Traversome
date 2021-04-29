

# Traversome (Underdevelopment)
Genomic isomer frequency estimation from genome assembly graphs and long reads.


### Install dependencies
```bash
conda install numpy pandas scipy pymc3 sympy -c conda-forge
```

### Install devel version of ifragaria
```bash
git clone -b modular https://github.com/Kinggerm/Traversome
pip install -e . --no-deps
```

### Command line interface (CLI)

Temporary usage:
```bash
python ifragaria/estimate_isomer_frequencies.py -g graph.gfa -a align.gaf -o .
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
    |-- Assembly.py
    |-- GraphAlignRecords.py
    |-- EstCopyDepthFromCov.py
    |-- EstCopyDepthPrecise.py (still working)
    |-- ...