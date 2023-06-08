# iHMM_GP
Contains relevant codes for the paper "Scalable nonparametric Bayesian learning for dynamic velocity fields"

From complex heterogenous spatio-temporal data like traffic or ocean/wind current, this model can efficiently extract spatial and temporal patterns without knowledge of number of components in an unsupervised manner. We view each such pattern as a distinct vector field, and this latent field changes across time where the temporal dynamics is assumed Markovian - the observations are believed to be noisy data from these latent fields at specific spatial locations, where we allow the number of observations (and their positions) at different time points to be different. We use Gaussian process for the spatial patterns and infinite Hidden Markov Model for the temporal dynamics. For inference, we propose a novel two pass algorithm, based on sequential greedy MAP estimation, to be able to analyze a large amount of data through our complex nonparametric model in reasonable time. For further speeding up the inference process, we use Sparse Gaussian Process with fixed inducing points in place of usual Gaussian process. Below is an example from NGSIM data - which demonstrates the abstraction we employ, starting from the actual intersection image (left) to real observations at a fixed time (right) - the third figure is one of the estimated vector fields, which we believe underlies the time point in the right-most figure.
![abstraction](https://github.com/sunritc/iHMM_GP_project/assets/62302566/c11e1abb-8804-4062-b65b-8e66e17c2d99)

Here are a few of the extracted patterns along with their temporal dynamics (100 step transition probabilities) - see paper for further details.
[ngsim_patterns.pdf](https://github.com/sunritc/iHMM_GP_project/files/11691354/ngsim_patterns.pdf)


The folder iHMM_GP contains all the relevant codes. In particular
1. data_setup and data_utils deal with creating data for simulations
2. gp_utils contain codes for gaussian process and sparse gaussian process needed in algorithm
3. step1_utils contain forward pass and refinement codes
4. main combines these -> main function here is: fit_model

See example.ipynb for few examples of how to use the codes. 

All codes run on python v3.10.7
Requirements
numpy 1.10.3
scipy 1.3.1
sklearn 1.0.2
matplotlib 3.4.3
for optimizing inducing points/further controls -> may use SVGP from GPflow (https://github.com/GPflow/GPflow/blob/develop/gpflow/models/svgp.py) (not used here)
