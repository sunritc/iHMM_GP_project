# iHMM_GP
Contains relevant codes for the paper "Scalable nonparametric Bayesian learning for dynamic velocity fields"

The folder iHMM_GP contains all the relevant codes. In particular
1. data_setup and data_utils deal with creating data for simulations
2. gp_utils contain codes for gaussian process and sparse gaussian process needed in algorithm
3. step1_utils contain forward pass and refinement codes
4. main combines these -> main function here is: fit_model

fit_model takes the following inputs:
1. data_train -> this is an array whose every element is a tuple (X_t, Y_t), each being np array
2. m0 -> block size, if m0 > T (length of data_train) then no blocks are used
3. Z -> inducing points (points in spatial domain of X)
4. n0 -> threshold parameter in refinement step
5. iHMM_params -> (alpha, beta, gamma)
6. Lmax -> max no of iterations for Step 2
7. n_jobs -> for using parallelize

Recommendations:
1. block size: higher block size is better but takes more time
2. inducing points -> set of points whihc best captures the spatial function (might consider optimizing over them later)
3. n0 -> if no knowledge of size of clusters/no of clusters, use 0. Higher values will remove small clusters.
4. iHMM parameters -> pretty robust, if the temporal process has lot of self-transition, increase alpha; otherwise better to search for more clusters by increasing gamma, beta
5. Lmax -> check convergence, higher value takes longer

See example.ipynb for few examples of how to use the codes. 

