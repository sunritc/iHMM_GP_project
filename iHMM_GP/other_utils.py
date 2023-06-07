import numpy as np


def log1(x):
    if x <= 0:
        return -np.inf
    else:
        return np.log(x)


new_log = np.vectorize(log1)

def rename_s(s):
    # renames states so that increasing in the order of appearance
    old_new = {}
    curr_max_label = 0

    T = len(s)
    for t in range(T):
        if s[t] not in old_new.keys():
            old_new[s[t]] = curr_max_label
            curr_max_label += 1

    s_new = s.copy()
    for t in range(T):
        s_new[t] = old_new[s[t]]
    return old_new, np.array(s_new)

def get_nm(s, o):
    # constructs N, M from s, o
    K = len(np.unique(s))
    n = np.zeros((K, K), dtype=int)
    m = np.array([np.sum((s == k)*(o == 1)) for k in range(K)], dtype=int)
    for t in range(len(s)-1):
        n[s[t], s[t+1]] += 1
    return n, m

def arg_min_nth(a, n):
    # returns the nth smallest element's position

    a_sort = np.sort(np.unique(a))
    id = np.array([], dtype=int)
    for i in a_sort:
        id = np.append(id, np.where(a == i)[0])
    return id[n]

def viterbi(z_start, z_stop, Pi, logliks):
    
    # z_start -> z[0] (or None)
    # z_stop -> z[T-1] (or None)
    # Pi -> transition matrix
    # logliks -> (TxK) log likelihoods for each data for each cluster
    # output: z[0], z[1], ... z[T-1]

    T = len(logliks)
    K = np.shape(Pi)[0]
    logPi = new_log(Pi)

    z = np.ones(T, dtype=int) * -1

    if (z_start is None) and (z_stop is None):  # nothing given - CASE 1

        # initialize
        xi = np.zeros((T, K))
        Psi = np.zeros((T-1, K), dtype=int)
        xi[0] = np.log(np.ones(K)/K) + logliks[0]

        # induction
        for t in range(T-1):
            B = logPi + xi[t][:, None]
            Psi[t] = np.argmax(B, axis=0)
            b = np.max(B, axis=0)
            xi[t+1] = b + logliks[t+1]

        # backtrack
        z[-1] = np.argmax(xi[-1])
        for t in range(T-2, -1, -1):
            z[t] = Psi[t][z[t+1]]

    # z[T-1] is given only - CASE 2
    elif (z_start is None) and (z_stop is not None):

        # initialize
        xi = np.zeros((T, K))
        Psi = np.zeros((T-1, K), dtype=int)
        xi[0] = np.log(np.ones(K)/K) + logliks[0]

        # induction
        for t in range(T-1):
            B = logPi + xi[t][:, None]
            Psi[t] = np.argmax(B, axis=0)
            b = np.max(B, axis=0)
            xi[t+1] = b + logliks[t+1]

        # backtrack
        z[T-1] = z_stop
        for t in range(T-2, -1, -1):
            z[t] = Psi[t][z[t+1]]

    # z[0] is given only - CASE 3
    elif (z_start is not None) and (z_stop is None):

        # initialize
        xi = np.zeros((T-1, K))
        Psi = np.zeros((T-2, K), dtype=int)
        z[0] = z_start
        xi[0] = logPi[z_start] + logliks[1]

        # induction
        for t in range(T-2):
            B = logPi + xi[t][:, None]
            Psi[t] = np.argmax(B, axis=0)
            b = np.max(B, axis=0)
            xi[t+1] = b + logliks[t+2]

        # backtrack
        z[T-1] = np.argmax(xi[T-2])
        for t in range(T-3, -1, -1):
            z[t+1] = Psi[t][z[t+2]]

    else:  # both z[0] and z[T-1] given - CASE 4

        # initialize
        xi = np.zeros((T-1, K))
        Psi = np.zeros((T-2, K), dtype=int)
        xi[0] = logPi[z_start] + logliks[1]

        # induction step
        for t in range(T-2):
            B = logPi + xi[t][:, None]
            Psi[t] = np.argmax(B, axis=0)
            b = np.max(B, axis=0)
            xi[t+1] = b + logliks[t+2]

        # backtrack
        z[T-1] = z_stop
        for t in range(T-3, -1, -1):
            z[t+1] = Psi[t][z[t+2]]
        z[0] = z_start

    return z

