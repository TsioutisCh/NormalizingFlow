# NormalizingFlow
Part of master thesis. A flow based generative model implemented to a Markov Chain Monte Carlo for Quantum Oscillators.

This repository contains the code used for the simulation of Metropolis and Hybrid Monte Carlo algorithms for the Quantum Anharmonic Oscillator. RealNVP.py contains the implementation of RealNVP (normalizing flow) to the same physical system. 


## Abstract

The core objective of this research was to explore the application of a flow-based Monte Carlo algorithm, specifically RealNVP, to mitigate autocorrelation issues that are present instate of the art algorithms such as the Metropolis and Hybrid Monte Carlo. Based on the path integral formulation, we initially implemented Metropolis-Hastings and Hybrid Monte Carlo methods for both harmonic and anharmonic oscillators, establishing a baseline for comparison with the RealNVP algorithm.

In our study we showcased that RealNVP can produce ensembles that accurately represent the theory, offering significant improvements in terms of autocorrelation times and accuracy. However the application of this method to more quantitative questions such as how the method scales with the physical volume, higher dimensions, and dependence on the model parameters such as the magnitude of the anharmonic potential requires further investigation. 
