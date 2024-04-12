

# Traversome
Genomic structure frequency estimation from genome assembly graphs and long reads.


### Installation

Install dependencies using conda. I recommend using the mamba version of conda.
```bash
mamba create -n traversome_env
mamba activate traversome_env
mamba install python numpy scipy sympy python-symengine dill typer loguru
```
<details><summary>[Optional] Install dependencies for running Bayesian MCMC.</summary>
If you want to run Bayesian mcmc with Traversome, you have to install pymc and pytensor. 
Due to the fast evolving of pymc, sometimes its installation may be unsuccessful and not seen during the installation.

```bash
mamba install pytensor pymc
```
</details>

Install Traversome using pip.
```bash
git clone --depth=1 https://github.com/JianjunJin/Traversome
pip install ./Traversome --no-deps
```

### Command line interface (CLI)

```bash
traversome thorough -g graph.gfa -a align.gaf -o outdir --topo circular --v-comp all
```

Important optional flags to finetune for achieving valid result (high bootstrap support):

```
--min-align-id        Threshold for alignment identity, below which the alignment will be discarded. [default: 0.992]
--min-align-len       Threshold for alignment length, below which the alignment will be discarded. [default: 5000]
--min-align-counts    Threshold for counts per path, below which the alignment(s) of that path will be discarded. [default: auto]
```

Use `traversome thorough -h` to see details for above flags and other flags.

### Interpreting results
```
|-- output_dir
    |-- traversome.log.txt          running log
    |-- variants.info.tab           information of survival variants after model selection and bootstrap
    |-- bootstrap.replicates.tab    bootstrap results
    |-- final.result.tab            summary of pangenome solutions
    |-- pangenome.gfa               pangenome graph of the best supported result
```
