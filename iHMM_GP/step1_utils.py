from iHMM_GP.gp_utils import *
from iHMM_GP.other_utils import *

##############################################################################
#
#
#           FIRST PASS CODES
#
#
###############################################################################


def oracle_state(N, M, curr_data, s_last, models, marginal_ll_new, iHMM_params, sigma2):

    # estimates one step oracle and state variables

    alpha, beta, gamma = iHMM_params
    K = N.shape[0]  # this is current number of clusters

    Pi = np.zeros((2, K+1))

    for j in range(K):
        Pi[0, j] = Pi[1, j] = log_density(models[j], curr_data, sigma2)

    Pi[1, K] = marginal_ll_new  # extracted from fitted GPs at start

    for j in range(K):
        # oracle = 0
        if j != s_last:
            Pi[0, j] += new_log(N[s_last, j]) - \
                new_log(N.sum(axis=1)[s_last] + alpha + beta)
        else:
            Pi[0, j] += new_log(N[s_last, j] + alpha) - \
                new_log(N.sum(axis=1)[s_last] + alpha + beta)

        # oracle = 1
        Pi[1, j] += new_log(beta) - new_log(N.sum(axis=1)[s_last] + alpha + beta) \
            + new_log(M[j]) - new_log(M.sum() + gamma)

    Pi[0, K] = -np.inf
    Pi[1, K] += new_log(beta) - new_log(N.sum(axis=1)[s_last] + alpha + beta) + \
        new_log(gamma) - new_log(M.sum() + gamma)

    o, s = np.unravel_index(np.argmax(Pi), Pi.shape)

    return (s, o)


def forward_pass(full_data, iHMM_params, marginal_ll, all_kernels, sigma2, Z=None, type='SGP', update_kernel_every=0, N_max=500, verbose=False):
    '''
    Performs single forward pass through entire data
    full_data -> full data of length T, each obj is (X_t, Y_t)
    iHMM_params -> (alpha, beta, gamma)
    marginal_ll -> marginal loglikelihoods for each time point (using GP initialization)
    all_kernels -> kernel parameters for each time point (also from initialization)
    sigma2 -> P-length estimate of variance (from init)
    Z -> inducing points (not required if type is GP)
    type -> either 'GP' or 'SGP'
    update_kernel_every -> how often to update the kernels (default=0: never)
    N_max -> if update_kernel_every>0, then max no of points used to re-estimate kernel
    verbose -> if True, prints details
    '''

    T = len(full_data)
    P = full_data[0][1].shape[1]

    K = 0
    N = np.zeros((K, K), dtype=int)
    M = np.zeros(K, dtype=int)
    models = []

    s = np.zeros(T, dtype=int)
    o = np.zeros(T, dtype=int)

    for t in range(T):
        # for t in tqdm(range(T)):  # can use tqdm
        if verbose:
            print('t=', t, 'K=', K)
        if t == 0:
            s[t] = 0
            o[t] = 1
            N = np.array([[0]], dtype=int)
            M = np.array([1], dtype=int)
            models = [fit_GP_old_kernel(
                full_data[t], sigma2, all_kernels[t], Z, type)]
            K = 1

        else:
            s_new, o_new = oracle_state(
                N, M, full_data[t], s[t-1], models, marginal_ll[t], iHMM_params, sigma2)
            s[t] = s_new
            o[t] = o_new

            if s_new < K:
                N[s[t-1], s[t]] += 1
                M[s_new] += o_new
                models[s[t]] = add_data_models(
                    models[s[t]], full_data[t], sigma2)
            else:
                a = np.zeros((K, 1), dtype=int)
                b = np.zeros((1, K+1), dtype=int)
                a[s[t-1], 0] = 1
                N = np.block([[N, a], [b]])
                M = np.append(M, o_new)
                model_ = fit_GP_old_kernel(
                    full_data[t], sigma2, all_kernels[t], Z, type)
                models.append(model_)
                K += 1

            if (update_kernel_every > 0) and (t % update_kernel_every == 0):
                new_models = []
                #print('len models=', len(models))
                for k in range(len(models)):
                    #print(k)
                    X_ = models[k][0].X
                    Y_ = np.hstack([models[k][j].Y for j in range(P)])
                    if type == 'GP':
                        sigma2_ = np.array(
                            [models[k][j].sigma2 for j in range(P)])
                    else:
                        sigma2_ = np.array(
                            [1/models[k][j].precision for j in range(P)])
                    new_models.append(fit_GP_new_kernel(
                        (X_, Y_), sigma2_, Z, N_max=N_max, type=type))
                models = new_models

    return (s, o, models, N, M)

##############################################################################
#
#
#           SECOND PASS CODES
#
#
###############################################################################


def identify_runs(s, k):

    # identifies consecutive occurences of k in state sequence s

    idx = np.where(s == k)[0]
    if len(idx) == 0:
        return []
    t = 0
    my_list = []
    curr_list = [idx[t]]

    T = len(idx)

    # identify the runs
    for t in range(T-1):
        if idx[t+1] == (idx[t] + 1):
            curr_list.append(idx[t+1])
        else:
            my_list.append(np.array(curr_list, dtype=int))
            curr_list = [idx[t+1]]
    if len(curr_list) > 0:
        my_list.append(np.array(curr_list, dtype=int))

    # identify the before and after states for each run
    T = len(my_list)
    my_data = []

    for t in range(T):
        curr_list = my_list[t]
        if np.min(curr_list) == 0:
            z_start = None
        else:
            z_start = s[curr_list[0] - 1]
        if np.max(curr_list) == len(s) - 1:
            z_stop = None
        else:
            z_stop = s[curr_list[-1] + 1]

        my_data.append({'run': curr_list,
                       'z_start': z_start,
                        'z_stop': z_stop})

    return my_data


def get_new_Pi(k, N, M, models, iHMM_params):

    # returns Pi, models to be used in viterbi after removing cluster k

    alpha, beta, gamma = iHMM_params

    # remove the curr state k_small from n, m and Theta_posteriors
    n_new = N.copy()
    m_new = M.copy()
    models_new = models.copy()

    del models_new[k]
    n_new = np.delete(n_new, k, axis=0)
    n_new = np.delete(n_new, k, axis=1)
    m_new = np.delete(m_new, k, axis=0)
    K_new = len(m_new)

    # print(n_new)
    # print(m_new)

    Pi = np.zeros((K_new + 1, K_new + 1))
    for i in range(K_new):
        den1 = n_new[i].sum() + alpha + beta
        den2 = m_new.sum() + gamma
        p = np.array([n_new[i, j]/den1 + (beta/den1) * (m_new[j]/den2)
                     for j in range(K_new)])
        p[i] = p[i] + alpha/den1
        p = np.append(p, (beta/den1) * (gamma/den2))

        Pi[i] = p

    for j in range(K_new+1):
        den1 = alpha + beta
        den2 = m_new.sum() + gamma
        if j < K_new:
            Pi[-1, j] = (beta/den1) * (m_new[j]/den2)
        else:
            Pi[-1, j] = (alpha/den1) + (beta/den1) * (gamma/den2)

    return Pi, models_new


def check_cluster(s, o, N, M, k, models, iHMM_params, full_data, marginal_lls, sigma2, min_cluster_size=5):

    # second pass checking for kth cluster
    # remove cluster k from s, o, N, M, models
    # for each t with s[t]=k, reassign them using viterbi using runs
    # check how many times new cluster is needed
    # if this number > n0 -> need to consider it as a separate cluster
    # else -> remove cluster, assign these points to closest in existing clusters

    Pi_new, models_new = get_new_Pi(k, N, M, models, iHMM_params)
    total_occurence = np.sum(s == k)
    where_occurence = np.where(s == k)[0]
    runs = identify_runs(s, k)
    K = len(models_new)
    count_new = 0
    new_states = np.array([], dtype=int)

    if total_occurence <= min_cluster_size:

        count_new = total_occurence
        new_states = K * np.ones(total_occurence, dtype=int)

    else:
        L = len(runs)
        for l in range(L):
            run = runs[l]['run']
            z_start = runs[l]['z_start']
            z_stop = runs[l]['z_stop']

            if (z_start is None) and (z_stop is None):
                start = run[0]
                length = len(run)
            elif (z_start is None) and (z_stop is not None):
                start = run[0]
                length = len(run) + 1
            elif (z_start is not None) and (z_stop is None):
                start = run[0] - 1
                length = len(run) + 1
            elif (z_start is not None) and (z_stop is not None):
                start = run[0] - 1
                length = len(run) + 2
            data_run_idx = np.arange(start, start + length)
            # print(data_run_idx)

            # construct likelihoods
            T1 = len(data_run_idx)

            logliks1 = np.array([[log_density(
                models_new[k], full_data[data_run_idx[j]], sigma2) for k in range(K)] for j in range(T1)])
            # logliks1 -> (T1 x K)
            marginal_lls_1 = np.array([marginal_lls[j]
                                       for j in data_run_idx])  # T1-len list
            logliks = np.block([logliks1, marginal_lls_1[:, None]])

            # account for k has been removed -> adjust z_start, z_stop
            if (z_start is not None) and (z_start > k):
                z_start = z_start - 1
            if (z_stop is not None) and (z_stop > k):
                z_stop = z_stop - 1

            z_new = viterbi(z_start, z_stop, Pi_new, logliks)
            # print(z_new)

            if (z_start is None) and (z_stop is None):
                new_states = np.append(new_states, z_new)
            elif (z_start is None) and (z_stop is not None):
                new_states = np.append(new_states, z_new[:-1])
            elif (z_start is not None) and (z_stop is None):
                new_states = np.append(new_states, z_new[1:])
            else:
                new_states = np.append(new_states, z_new[1:-1])

            count_new += np.sum(z_new == K)

    # print('total occurences=', total_occurence)
    # print('new needed=', count_new)
    # print('z_new', new_states)
    # print(len(new_states))

    if count_new <= min_cluster_size:  # remove this cluster

        retain_cluster = False

        # reassign these to other clusters
        s_new = s.copy()
        s_new[s_new == k] = -1
        s_new[s_new > k] -= 1

        T = len(s)
        targets = np.zeros(T, dtype=int)
        new_needed = np.where(new_states == K)[0]

        if len(new_needed) > 0:
            targets[where_occurence[new_needed]] = 1
        new_runs = identify_runs(targets, 1)
        L = len(new_runs)

        if L == 0:
            s_new[where_occurence] = new_states
            o_new = o
            N, M = get_nm(s_new, o_new)

        else:
            Pi_new, models_new = get_new_Pi(
                k, N, M, models, (iHMM_params[0], iHMM_params[1], 0))
            Pi_new1 = Pi_new[:-1, :-1]  # exclude new state

            reassigned_states = np.array([], dtype=int)

            for l in range(L):
                run = new_runs[l]['run']
                z_start = new_runs[l]['z_start']
                z_stop = new_runs[l]['z_stop']

                if (z_start is None) and (z_stop is None):
                    start = run[0]
                    length = len(run)
                elif (z_start is None) and (z_stop is not None):
                    start = run[0]
                    length = len(run) + 1
                elif (z_start is not None) and (z_stop is None):
                    start = run[0] - 1
                    length = len(run) + 1
                elif (z_start is not None) and (z_stop is not None):
                    start = run[0] - 1
                    length = len(run) + 2
                data_run_idx = np.arange(start, start + length)

                T1 = len(data_run_idx)

                logliks1 = np.array([[log_density(
                    models_new[k], full_data[data_run_idx[j]], sigma2) for k in range(K)] for j in range(T1)])
                # logliks1 -> (T1 x K)

                if (z_start is not None) and (z_start > k):
                    z_start = z_start - 1
                if (z_stop is not None) and (z_stop > k):
                    z_stop = z_stop - 1

                z_new = viterbi(z_start, z_stop, Pi_new1, logliks1)

                if (z_start is None) and (z_stop is None):
                    reassigned_states = np.append(reassigned_states, z_new)
                elif (z_start is None) and (z_stop is not None):
                    reassigned_states = np.append(
                        reassigned_states, z_new[:-1])
                elif (z_start is not None) and (z_stop is None):
                    reassigned_states = np.append(reassigned_states, z_new[1:])
                else:
                    reassigned_states = np.append(
                        reassigned_states, z_new[1:-1])

            s_new[np.where(targets == 1)[0]] = reassigned_states
            new_states[new_needed] = reassigned_states

            o_new = o
            N, M = get_nm(s_new, o_new)
    else:

        # keep current cluster, easy case
        retain_cluster = True
        s_new = s
        o_new = o
        models_new = models
        N, M = get_nm(s_new, o_new)

        new_needed = []
        new_states = []

    return retain_cluster, where_occurence, new_states, s_new, o_new, N, M, models_new


def refinement(s, o, N, M, models, iHMM_params, sigma2, marginal_lls, full_data, min_cluster_size=5, verbose=True):
    '''
    second pass -> refinement to reduce no. of clusters (apply check_clusters starting from smallest cluster)
    input:
        s, o, M, N, models from forward pass
        iHMM_params -> set at first
        sigma2 -> estimated forward pass
        marginal_lls -> from initialization
        full_data -> length T
        min_cluster_size -> n0 passed to check_cluster function
    '''

    K = len(models)
    K0 = K

    cluster_sizes = np.array([np.sum(s == k) for k in range(K)])

    do_again = 1
    count = 0
    lowest_cluster_current = 0

    T = len(s)

    while do_again == 1:

        count += 1

        # pick the smallest unattended cluster
        k = arg_min_nth(cluster_sizes, lowest_cluster_current)

        retain_cluster, where_occurence, new_states, s1, o1, _, _, models1 = check_cluster(
            s, o, N, M, k, models, iHMM_params, full_data, marginal_lls, sigma2, min_cluster_size)

        if verbose is True:
            print('checking cluster:', k, ', retained=', str(retain_cluster))

        if retain_cluster == True:

            lowest_cluster_current += 1

        else:

            # update the changed models
            for k in np.unique(new_states):
                idx = where_occurence[np.where(new_states == k)]
                X = np.vstack([full_data[i][0] for i in idx])
                Y = np.vstack([full_data[i][1] for i in idx])
                models1[k] = add_data_models(models1[k], (X, Y), sigma2)

            # relabel clusters so that increasing
            relabel, s = rename_s(s1)
            K = K-1

            models = []
            o = o1
            for j in relabel.keys():
                models.append(models1[j])

            N, M = get_nm(s, o)
            cluster_sizes = np.array([np.sum(s == k) for k in range(K)])
            # print(cluster_sizes)

        if (len(cluster_sizes) == lowest_cluster_current) or (count > K0):
            do_again = 0

    return s, o, models, N, M


def get_block_results(full_data, marginal_lls, all_kernels, sigma2, iHMM_params, Z, M=100, min_cluster_size=5, type='SGP', update_kernel_every=0, N_max=500, verbose=False):
    '''
    function to use blocks
    for each block, runs forward pass followed by refinement
    '''

    T = len(full_data)
    P = full_data[0][1].shape[1]

    def block(t, M=M, Z=Z, full_data=full_data, marginal_lls=marginal_lls, all_kernels=all_kernels, sigma2=sigma2, iHMM_params=iHMM_params, min_cluster_size=min_cluster_size, type=type, update_kernel_every=update_kernel_every, N_max=N_max, verbose=verbose):
        dat = full_data[t*M:(t*M+M)]
        all_kernels_ = all_kernels[t*M:(t*M+M)]
        marginal_lls_ = marginal_lls[t*M:(t*M+M)]

        s1, o1, models1, N1, M1 = forward_pass(
            dat, iHMM_params, marginal_lls_, all_kernels_, sigma2, Z, type, update_kernel_every, N_max)
        s2, o2, models2, N2, M2 = refinement(
            s1, o1, N1, M1, models1, iHMM_params, sigma2, marginal_lls_, dat, min_cluster_size, verbose=verbose)

        return len(models1), len(models2), models2, s2, o2, N2, M2

    from joblib import Parallel, delayed
    times = T // M
    left = T - M * times
    results = Parallel(n_jobs=-1, timeout=99999)(delayed(block)(t)
                                                 for t in range(times))

    if left > 0:
        dat = full_data[-left:]
        all_kernels_ = all_kernels[-left:]
        marginal_lls_ = marginal_lls[-left:]

        s1, o1, models1, N1, M1 = forward_pass(
            dat, iHMM_params, marginal_lls_, all_kernels_, sigma2, Z)
        s2, o2, models2, N2, M2 = refinement(
            s1, o1, N1, M1, models1, iHMM_params, sigma2, marginal_lls_, dat)
        results.append((len(models1), len(models2), models2, s2, o2, N2, M2))

    return results

##############################################################################
#
#
#           COMBINING BLOCKS CODE
#
#
###############################################################################


def get_K(results_from_ref, high=20, Z=None, plot=False):
    '''
    collect data from blocks
    combine them using KMeans
    for SGP: use the mean of SGP over the inducing points
    for GP: use the mean of GP over the passed Z
    '''
    # collect dimensions
    P = len(results_from_ref[0][2][0])
    T1 = len(results_from_ref)

    # collect the means from all models

    all_models = [results_from_ref[t][2] for t in range(T1)]
    type = all_models[0][0][0].name
    if type == 'GP' and Z is None:
        raise ValueError('For GP type, need Z to compare means')

    means = []
    for t in range(T1):
        for n in range(len(all_models[t])):
            m = all_models[t][n]
            if type == 'SGP':
                x = m[0].mu_Z
                if P > 1:
                    for j in range(1, P):
                        x = np.append(x, m[j].mu_Z)
                if P == 1:
                    x = x.reshape(len(x),)
            elif type == 'GP':
                x = m[0].predict_f(Z)[0]
                if P > 1:
                    for j in range(1, P):
                        x = np.append(x, m[j].predict_f(Z)[0])
                if P == 1:
                    x = x.reshape(len(x),)
            means.append(x)
    means = np.array(means)

    # use KMeans to determine number of components
    K2 = [results_from_ref[t][1] for t in range(T1)]
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    K2 = np.array(K2)
    K_ = np.arange(np.max([K2.min(), 3]), np.min(
        [K2.max()+high, means.shape[0]]))

    silhouttes = []
    distortions = []
    for k in K_:
        kmeanModel = KMeans(n_clusters=k, n_init=3)
        kmeanModel.fit(means)
        distortions.append(kmeanModel.inertia_)
        labels = kmeanModel.labels_
        silhouttes.append(silhouette_score(means, labels, metric='euclidean'))

    if T1 > 1 and plot is True:
        fig = plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(K_, distortions)
        plt.title('distortions')

        plt.subplot(1, 2, 2)
        plt.plot(K_, silhouttes)
        plt.title('silhouttes')

        fig.suptitle('choice of K', fontsize=20)
        fig.tight_layout
        plt.show()
    if T1 > 1:
        K_opt = K_[np.argmax(silhouttes)]
    elif T1 == 1:
        K_opt = results_from_ref[0][1]

    return K_opt, means


def get_final_states(K, means, results_ref):
    '''
    based on K_opt (after combining blocks)
    get logliks for GP models (evaluated on Z) 
    use Viterbi at end with these logliks and estimated Pi_hat
    '''
    # get the states clustered by Kmeans
    from sklearn.cluster import KMeans
    kmeanModel = KMeans(n_clusters=K, n_init=3)
    kmeanModel.fit(means)
    labels = kmeanModel.labels_

    s = []
    count = 0
    for block_idx in range(len(results_ref)):
        s_ = results_ref[block_idx][3]
        curr_labels = labels[count:(count+results_ref[block_idx][1])]
        for i in range(len(s_)):
            s.append(curr_labels[s_[i]])
        count += results_ref[block_idx][1]

    s_final = np.array(s, dtype=int)

    return s_final


def final_models(full_data, s_final, all_kernels, sigma2, Z, n_jobs=-1, N_max=500, type='SGP', fit_new=False):
    '''
    re-fit models based on s_final
    if fit_new is False: use weighted means for the kernel parameters (combining across time points)
    if fit_new is True: use N_max obs to estimate kernel params and use it to get model
    type is GP/SGP
    '''
    K = len(np.unique(s_final))

    def get_model(k, data=full_data, s_final=s_final, sigma2=sigma2, all_kernels=all_kernels, Z=Z, fit_new=fit_new):
        P = data[0][0].shape[1]
        idx = np.where(s_final == k)[0]

        ns = np.array([data[i][0].shape[0] for i in idx])
        X = np.vstack([full_data[i][0] for i in idx])
        Y = np.vstack([full_data[i][1] for i in idx])

        if fit_new is True:
            final_model = fit_GP_new_kernel((X, Y), sigma2, Z, N_max, type)
        else:
            L = len(idx)
            k_ls = []
            k_var = []
            final_model = []
            for j in range(P):
                ls = np.array([all_kernels[idx[l]][1][j] for l in range(L)])
                var = np.array([all_kernels[idx[l]][0][j] for l in range(L)])
                k_var = np.sum(var * ns) / np.sum(ns)
                k_ls = np.sum(ls * ns) / np.sum(ns)
                model_ = SGP((X, Y[:, j][:, None]), (k_var, k_ls), Z, sigma2)
            final_model.append(model_)

        return final_model

    from joblib import Parallel, delayed
    results = Parallel(n_jobs=n_jobs, timeout=99999)(delayed(get_model)(k)
                                                     for k in range(K))
    return results


##############################################################################
#
#
#           OTHER RELATED FUNCTIONS
#
#
###############################################################################

def get_likelihoods(data, models, sigma2, n_jobs=-1):

    def f(t, data=data, models=models, sigma2=sigma2):

        K = len(models)
        X, Y = data[t]
        logliks = np.array([log_density(models[k], (X, Y), sigma2)
                           for k in range(K)])
        return logliks

    T = len(data)
    from joblib import Parallel, delayed
    results = Parallel(n_jobs=n_jobs, backend="loky", timeout=99999)(
        delayed(f)(t) for t in range(T))
    return results


def compute_likelihood_HMM(transition_matrix, log_liks):
    # assume initial distribution is uniform
    from scipy.special import logsumexp
    A = new_log(transition_matrix)  # K by K : log P(s_1=j | s_0=i)
    B = log_liks  # T by K : log p(D_t|theta_k)
    K = A.shape[0]
    T = B.shape[0]

    alpha = np.zeros((T, K))

    # initialize
    alpha[0, :] = B[0, :] + new_log(1/K)

    # recursion
    for t in range(T-1):
        alpha[t+1, :] = np.array([(B[t+1, j] + logsumexp(alpha[t, :] + A[:, j]))
                                 for j in range(K)])

    # end
    return logsumexp(alpha[-1, :])


def get_final_results(data_train, s_final, sigma2, Z, n_jobs=-1, N_max=500, type='SGP'):

    K = len(np.unique(s_final))
    P = data_train[0][1].shape[1]

    def f(k, data=data_train, sigma2=sigma2, Z=Z, N_max=N_max):
        idx = np.where(s_final == k)[0]
        X_ = np.vstack([data[i][0] for i in idx])
        Y_ = np.vstack([data[i][1] for i in idx])

        models = fit_GP_new_kernel((X_, Y_), sigma2, Z, N_max, type)
        return models

    from joblib import Parallel, delayed
    results = Parallel(n_jobs=n_jobs, backend="loky", timeout=99999)(
        delayed(f)(k) for k in range(K))
    return results
