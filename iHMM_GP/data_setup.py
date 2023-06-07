from iHMM_GP.data_utils import *
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


f_true = []

f0 = generate_function_spline(s=13, k=80, seed=0)


def new_f0(x):
    return f0(x)


f_true.append(new_f0)

for j in range(3):
    f_true.append(generate_function_spline(s=10, k=50, seed=15*(j+1)))

f1 = generate_random_function(D=2, P=2, seed=60)
f2 = generate_random_function(D=2, P=2, seed=75)


def newf1(x):
    return f1(x)/3


def newf2(x):
    return f2(x)/3.3


f_true = f_true + [newf1, newf2]

f_new1 = merge(f_true[2], f_true[3], type=1)
f_true.append(f_new1)

f_new2 = merge(f_true[4], f_true[1], type=2)
f_true.append(f_new2)

g1 = f_true[1]
g2 = f_true[2]
g3 = f_true[3]
g5 = f_true[5]


def f_new3(x):
    eps1 = sigmoid(x[0])
    eps2 = sigmoid(x[1])
    return eps1*eps2*g1(x) + eps1*(1-eps2)*g2(x) + (1-eps1)*eps2*g3(x) + (1-eps1)*(1-eps2)*g5(x)


f_true.append(f_new3)

# Pi_true = np.array([[2, 1, 1, 0, 0, 0, 0, 1, 1],
#                   [1, 2, 1, 1, 0, 0, 0, 0, 1],
#                   [1, 1, 2, 1, 1, 0, 0, 0, 0],
#                   [0, 1, 1, 2, 1, 1, 0, 0, 0],
#                   [0, 0, 1, 1, 2, 1, 1, 0, 0],
#                   [0, 0, 0, 1, 1, 2, 1, 1, 0],
#                   [0, 0, 0, 0, 1, 1, 2, 1, 1],
#                   [1, 0, 0, 0, 0, 1, 1, 2, 1],
#                   [1, 1, 0, 0, 0, 0, 1, 1, 2]])
# Pi_true = Pi_true/Pi_true.sum(axis=1)[:,None]

Pi_true = np.eye(len(f_true)) * 0.1
for i in range(len(f_true)):
    Pi_true[i, (i+1) % (len(f_true))] = 0.9


# helper function
def is_in(x, xlims, ylims):
    # x is 2dim (x0, x1); returns True or False
    if (xlims[0] < x[0] < xlims[1]) and (ylims[0] < x[1] < ylims[1]):
        return True
    else:
        return False


# gridding system
def remove_data(X, Y):

    breaks = np.linspace(-1, 1, 4)

    grid_id_remove = np.random.choice(9, size=4, replace=False)

    rows = grid_id_remove//3
    cols = grid_id_remove % 3

    grid_positions = [(rows[i], cols[i]) for i in range(len(rows))]
    xlims = [(breaks[row], breaks[row+1]) for row in rows]
    ylims = [(breaks[col], breaks[col+1]) for col in cols]

    idx_to_remove = []

    for i in range(len(X)):
        x = X[i]
        if np.any([is_in(x, xlims[j], ylims[j]) for j in range(4)]):
            idx_to_remove.append(i)

    X_new = np.delete(X, idx_to_remove, axis=0)
    Y_new = np.delete(Y, idx_to_remove, axis=0)
    X_removed = X[idx_to_remove]
    Y_removed = Y[idx_to_remove]

    return (X_new, Y_new), (X_removed, Y_removed)


def sim_new_data(f0, Pi0, T=200, n=70, sigma2=2):

    n_ = int(n * 9/5)
    K = len(f0)
    sigma = np.sqrt(sigma2)

    # generate the hidden states
    s = np.zeros(T, dtype=int)
    for t in range(T-1):
        s[t+1] = np.random.choice(K, p=Pi0[s[t], :])

    # get full data
    full_data = []
    for t in range(T):
        X = np.random.uniform(low=-1, high=1, size=(n_, 2))
        f = f0[s[t]]
        Y_ = np.vstack([f(X[i]) for i in range(n_)])
        Y = Y_ + np.random.normal(scale=sigma, size=(n_, 2))
        full_data.append((X, Y))

    # use gridding system to remove some data
    final_data = []
    test_data = []
    for t in range(T):
        X, Y = full_data[t]
        (X_, Y_), (X_rem, Y_rem) = remove_data(X, Y)
        final_data.append((X_, Y_))
        test_data.append((X_rem, Y_rem))

    return final_data, s, test_data


def sim_new_data2(f0, Pi0, T=200, n=70, sigma2=2):

    n_ = int(n)
    K = len(f0)
    sigma = np.sqrt(sigma2)

    # generate the hidden states
    s = np.zeros(T, dtype=int)
    for t in range(T-1):
        s[t+1] = np.random.choice(K, p=Pi0[s[t], :])

    # get full data
    full_data = []
    for t in range(T):
        X = np.random.uniform(low=-1, high=1, size=(n_, 2))
        f = f0[s[t]]
        Y_ = np.vstack([f(X[i]) for i in range(n_)])
        Y = Y_ + np.random.normal(scale=sigma, size=(n_, 2))
        full_data.append((X, Y))

    return full_data, s


## equivalent functions in 1d

def in_region(x, c=0):
    if c == 0:
        if x<1/3 or x>2/3:
            return 1
        else:
            return 0
    elif c == 1:
        if x < 2/3:
            return 1
        else:
            return 0
    elif c == 2:
        if x > 1/3:
            return 1
        else:
            return 0

def censor1d(data):
    new_data = []
    c = np.random.choice(3, size=len(data))
    
    for t in range(len(data)):
        x, y = data[t]
        x = x.flatten()
        y = y.flatten()
        c_ = c[t]
        x_ = []
        y_ = []
        for i in range(len(x)):
            if in_region(x[i], c_) == 1:
                x_.append(x[i])
                y_.append(y[i])
        x_ = np.array(x_)
        y_ = np.array(y_)
        new_data.append((x_[:,None], y_[:,None]))
    return new_data
        
def sim_new_data_1d(f0, Pi0, T=200, n=70, sigma2=2):

    n_ = int(n)
    K = len(f0)
    sigma = np.sqrt(sigma2)

    # generate the hidden states
    s = np.zeros(T, dtype=int)
    for t in range(T-1):
        s[t+1] = np.random.choice(K, p=Pi0[s[t], :])

    # get full data
    full_data = []
    for t in range(T):
        X = np.random.uniform(low=0, high=1, size=n_)
        f = f0[s[t]]
        Y_ = f(X)
        Y = Y_ + np.random.normal(scale=sigma, size=n_)
        full_data.append((X[:,None], Y[:,None]))

    return full_data, s