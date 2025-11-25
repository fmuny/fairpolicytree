# Fairness-Aware and Interpretable Policy Learning

Welcome to the repository for the **fairpolicytree** package, developed as part of the research paper [_Fairness-Aware and Interpretable Policy Learning_](https://arxiv.org/abs/2509.12119).

This repository provides:

- The `fairpolicytree` R package for fair and interpretable policy learning.
- Replication files to reproduce the empirical results from the paper.

## 🔧 Installation

You can install the `fairpolicytree` package directly from GitHub using:

```R
devtools::install_github("fmuny/fairpolicytree")
````

## 📄 Documentation

* **Package documentation PDF** [Download here](https://github.com/fmuny/fairpolicytree/blob/main/fairpolicytree_0.1.0.pdf?raw=true)

## 🔁 Reproducing the Results

To replicate the empirical results from the paper:

Open the [replication notebook](https://fmuny.github.io/fairpolicytree/replication_paper/replication_notebook.html) for a step-by-step guide or

1. Navigate to the `replication_paper` folder.
2. Follow the instructions in the included scripts

The analysis has been carried out on a machine with 16 virtual CPUs, 64 GB RAM, 250 GB SSD Storage and the MS Windows Server 2022 Datacenter operating system. 
With these specifications the runtimes are as follows:

* `data_cleaning.py`: ~5 min 
* `mcf_estimation.py`: ~5 days, 22 hours, much faster with parallelization (`gen_mp_parallel=None`), but results will not be peferctly reproducible
* `replication_code.R`: ~1 hour, 15 min

Data preparation and estimation of scores has been carried out with Python 3.12.11, the detailed dependencies can be found [here](https://https://github.com/fmuny/fairpolicytree/blob/main/replication_paper/python_environment.yaml?raw=true). The main analysis is conducted using R version 4.4.2.


## 📫 Citation

If you use this package or replication materials, please cite the [paper](https://arxiv.org/abs/2509.12119):

> *Bearth, N., Lechner, M., Mareckova, J., Muny, F.* (2025). **Fairness-Aware and Interpretable Policy Learning**. *arXiv preprint
arXiv:2509.12119*.

## 📜 License

This project is licensed under the MIT License.

