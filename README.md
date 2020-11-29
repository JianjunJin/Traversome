

# ifragaria
Genomic isomer frequency estimation from genome assembly graphs and long reads.


### Install dependencies
```bash
conda install numpy pandas scipy pymc3 sympy -c conda-forge
```

### Install devel version of ifragaria
```bash
git clone -b modular https://github.com/dereneaton/Fragaria
pip install -e . --no-deps
```

### Command line interface (CLI)

Temporary usage:
```bash
python ifragaria/estimate_isomer_frequencies.py -g graph.gfa -a align.gaf -o .
```

future: 
```bash
ifragaria -g graph.gfa -a align.gaf -o .
```

### Interpreting results
...


### (Advanced) Bayesian estimation
...


## Development

I'm working through the code starting from the very beginning of `estimate_isomer_frequencies.py` and moving step-by-step forward. At each step
if I find a large contiguous chunk of code I break it out into a new class
module, often in a new file. As I work through the class I try to understand
what it is doing at each step by adding comments at each block if there are 
none. If I do not fully understand I usually insert a comment block but may 
just write "...". 


The `ifragaria.Fragaria` class object is the main object, and the `.run()` 
function of this object will perform the core functions of ifragaria. This 
is currently unfinished. It completes up to the step of using sympy to 
simplify the equation, but I am only starting now on the scipy estimation.


```
# workflow
|-- __main__.py
|-- ifragaria.py
    |-- Assembly.py
    |-- GraphAlignRecords.py
    |-- EstCopyDepthFromCov.py
    |-- EstCopyDepthPrecise.py (still working)
    |-- ...