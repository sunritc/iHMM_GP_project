from iHMM_GP.step1_utils import *


def fit_model(data_train, m0, Z, k_ls_bounds=(0.01, 10), k_var_bounds=(0.01, 10), noise_var_bounds=(1e-4, 10), n0=5, iHMM_params=(3,2,1), Lmax=30, n_jobs=-1, N_max=1000, update_kernel_every=0, type='SGP', verbose=True, high=20):
    '''
    INPUT:
        data_train: training data - array of length T, t-th object is (X_t, Y_t), X_t (n_t, D) and Y_t (n_t, P) shaped nparrays
        type: 'GP' / 'SGP' (default: 'SGP')
        hyperparameters:
            m0: number of blocks to use in Part (A) (no default)
            n0: minimum cluster size - tuning parameter in refinement step (default: 5)
            iHMM_params: (a,b,c) - tuple alpha, beta, gamma (default: (3,2,1))
            Z: set of inducing points (shape (L,D)) (no default)
            Lmax: max no. of final iterations (Step 2) (default: 30)
        k_ls_bounds: upper and lower bounds for kernel lengthscale
        k_var_bounds: upper and lower bounds for kernel variance
        noise_var_bounds: upper and lower bounds for likelihood noise sigma_j^2
        maxiter: maximum number of iterations in Part (B)
        n_jobs: no of cores to use (used in blocks, Step B) note: initial GP fit use all available cores
        N_max: max no of training points used to fit GP to estimate kernel
        update_kernel_every: how frequently to update kernel during forward pass
        verbose: if True, prints certain steps
        high: buffer parameter for selecting optimal K (default: 20), searches over these extra larger K to see best based on silhoutte index
        Note: For type='SGP', Z serves as inducing points, for 'GP', it serves as locations over which clustering is done for combining blocks
    OUTPUT:
        K_opt: number of latent states
        s_final_train: estimates of latent state variables for each time
        final_models: list of length K_opt, j-th element is P-tuple of SGP-models (j-th latent state, P-output dimensions)
        hmm_loglik_train: loglikelihood of model under HMM assumption, using estimates
        Pi_hat: estimate of transition matrix
        sigma2: estimate of noise variance (P-length, recall noise is diagonal)
        time: total time of execution
    '''
    # Fitting the model
    import time
    T = len(data_train)

    #### PART (A) #####
    if verbose:
        print('Starting Step A')

    # fit GPs
    time0_start = time.time()
    marginal_lls, sigma2, all_kernels = get_all_GPs(
        data_train, parallelize=True, k_ls_bounds=k_ls_bounds, k_var_bounds=k_var_bounds, noise_var_bounds=noise_var_bounds)

    if verbose:
        print('GPs done')

    # get block results
    results = get_block_results(
        data_train, marginal_lls, all_kernels, sigma2, iHMM_params, Z=Z, M=m0, min_cluster_size=n0, type=type, update_kernel_every=update_kernel_every, N_max=N_max, verbose=False)
    if verbose:
        print('blocks done')

    K_opt, means = get_K(results, high=high, Z=Z, plot=verbose)
    s1 = get_final_states(K_opt, means, results)
    if verbose:
        print('optimal K=', K_opt)

    # use block results to get models
    final_models_ = final_models(
        data_train, s1, all_kernels, sigma2, Z, n_jobs=n_jobs, N_max=N_max, type=type, fit_new=True)
    time0 = time.time() - time0_start

    #### PART (B) ####

    conv = 1
    count = 0
    time1 = time.time()
    # print('time till now:', time0)

    if verbose:
        print('Starting Step B')

    while (conv >= 1) and (count < Lmax):
        if verbose:
            print('iter:', count+1)

        # compute likelihoods of each observed points in data_train
        logliks_train = get_likelihoods(data_train, final_models_, sigma2) # uses all available cores
        logliks_train = np.array(logliks_train)
        # viterbi for train
        Pi_est, _ = get_nm(s1, np.zeros(T))
        row_sums = Pi_est.sum(axis=1)
        Pi_hat = Pi_est / row_sums[:, np.newaxis]
        s_final_train = viterbi(None, None, Pi_hat, logliks_train)

        # check if any cluster has 0 time points, can remove it
        s_final_train = rename_s(s_final_train)[1]
        K_opt = len(np.unique(s_final_train))

        final_models_ = get_final_results(
            data_train, s_final_train, sigma2, Z, n_jobs=n_jobs, N_max=N_max, type=type)  # new 2 lines
        conv = np.linalg.norm(s_final_train - s1)
        s1 = s_final_train.copy()
        count += 1

    if count < Lmax:
        print('converged in ', count)
    else:
        print('did not converge')
    # print(count)
    logliks_train = get_likelihoods(data_train, final_models_, sigma2)
    logliks_train = np.array(logliks_train)
    Pi_est, _ = get_nm(s_final_train, np.zeros(T))
    row_sums = Pi_est.sum(axis=1)
    Pi_hat = Pi_est / row_sums[:, np.newaxis]

    s_final_train = viterbi(None, None, Pi_hat, logliks_train)
    time1 = time.time() - time1

    # calculate likelihood
    hmm_loglik_train = compute_likelihood_HMM(Pi_hat, logliks_train)

    return K_opt, s_final_train, final_models_, hmm_loglik_train, Pi_hat, sigma2, time0+time1
