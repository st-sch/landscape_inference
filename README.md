# landscape_inference

This repository contains the Python code for fitness landscape inference, validation and visualization used in our manuscript on a heterogeneously epistatic antibody fitness landscape promoting evolvability. It reproduces the plots shown in Figs. 1, S3 and S4.

## Contents

| file/directory      | description       |
|----------------|----------------|
| [`data`](data/) | sequence count data reported out of the experiment (see [this Github repo](https://github.com/nicwulab/COV107-23_fitness_landscape)) |
| [`output`](output/) | output data and plots generated by the Jupyter scripts below |
| [`src`](src/) | some functions I wrote (e.g. for landscape class, Eq. (1), γ statistics, etc.) |
| [`fig1B.ipynb`](fig1B.ipynb) | global epistasis model for sequence count data using maximum-likelihood estimation |
| [`fig1C.ipynb`](fig1C.ipynb) | specific epistasis model for sequence count data using maximum-likelihood estimation |
| [`fig1D_S4AB.ipynb`](fig1D_S4AB.ipynb) | low-dimensional visualization of fitness landscape using force-directed network layout |
| [`fig1E.ipynb`](fig1E.ipynb) | low-dimensional and comparable visualization of fitness (sub-) landscapes using force-directed network layout |
| [`figS3A.ipynb`](figS3A.ipynb) | validation of specific epistasis models obtained from maximum-likelihood estimation (using cross-validation) and from Walsh-Hadamard transform (using bandpass filters) |

## Code references

These are the code used in this project that were written by other people (besides standard Python libraries):

- [sparray](https://github.com/jesolem/sparray): used to deal with arrays of *a priori* unknown number of dimensions
- [mavenn](https://github.com/jbkinney/mavenn): used for fitting global epistasis models