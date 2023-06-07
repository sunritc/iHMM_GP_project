from scipy.stats import multivariate_normal
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
import numpy as np

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


@ignore_warnings(category=ConvergenceWarning)
def fit_GP(data, sigma2=None, k_ls_bounds=(0.01, 10), k_var_bounds=(0.01, 10), noise_var_bounds=(1e-4, 10), return_model=False):

    # fits a NEW GP to given data
    # fits independent GPs to each of output dimensions
    # returns overall marginal loglikelihood and models
    # models[j] is GP model for (X,Y[:,j])
    # estimates all parameters

    X, Y = data
    n, D = X.shape
    _, P = Y.shape

    kernels_var = []
    kernels_ls = []
    vars = []
    marginal_loglik = 0.0

    gprs = []

    for j in range(P):
        if sigma2 is None:
            kernel = ConstantKernel(constant_value_bounds=k_var_bounds) * RBF(
                length_scale_bounds=k_ls_bounds) + WhiteKernel(noise_level_bounds=noise_var_bounds)
        else:
            kernel = ConstantKernel(constant_value_bounds=k_var_bounds) * RBF(
                length_scale_bounds=k_ls_bounds) + WhiteKernel(noise_level=sigma2[j], noise_level_bounds="fixed")
        y = Y[:, j][:, None]
        gpr = GaussianProcessRegressor(kernel=kernel, random_state=0).fit(X, y)
        gprs.append(gpr)

        kernel_params = np.exp(gpr.kernel_.theta)

        # adjust kernel variance and lengthscales so that bounded between 1e-5 and 1000
        k_var = np.clip(kernel_params[0], 1e-5, 1000)
        k_ls = np.clip(kernel_params[1], 1e-5, 1000)

        if sigma2 is None:
            lik_var = np.clip(kernel_params[2], 1e-5, 1000)
        else:
            lik_var = sigma2

        kernels_var.append(k_var)
        kernels_ls.append(k_ls)
        vars.append(lik_var)
        marginal_loglik += gpr.log_marginal_likelihood_value_

    if return_model is False:
        return marginal_loglik, vars, (kernels_var, kernels_ls)
    else:
        return gprs


def get_all_GPs(full_data, parallelize=True, k_ls_bounds=(0.01, 10), k_var_bounds=(0.01, 10), noise_var_bounds=(1e-3, 1e3), plot_var_dist=False):
    '''
    parallelize over fit_GP to fit GP at each time points
    uses all available cores
    returns marginal_logliks (T-array), estimated variance (P-array), all kernels (T-array of objects - (kernel_variances, kernel_lengthscales))
    '''
    T = len(full_data)
    X, Y = full_data[0]

    def f(t, k_ls_bounds=k_ls_bounds, k_var_bounds=k_var_bounds, noise_var_bounds=noise_var_bounds):
        marg_ll, vars, kernels = fit_GP(
            full_data[t], k_ls_bounds=k_ls_bounds, k_var_bounds=k_var_bounds, noise_var_bounds=noise_var_bounds)
        return marg_ll, vars, kernels

    if parallelize is True:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=-1, timeout=99999)(delayed(f)(t)
                                                     for t in range(T))
    else:
        results = [f(t) for t in range(T)]  # can use tqdm

    marginal_lls = [results[i][0] for i in range(T)]
    variances = np.array([results[i][1] for i in range(T)])
    all_kernels = [(results[i][2][0], results[i][2][1])
                   for i in range(T)]  # all_kernels[t] -> (ker_var, ker_ls)
    ns = np.array([full_data[t][0].shape[0] for t in range(T)])

    P = Y.shape[1]
    est_var = []
    for j in range(P):
        est_var.append(np.sum(ns * variances[:, j]) / np.sum(ns))

    if plot_var_dist is True:
        fig = plt.figure(figsize=(9, 4))
        plt.subplot(1, 2, 1).set_title("dim 1")
        plt.hist(variances[:, 0], alpha=0.4)
        plt.subplot(1, 2, 2).set_title("dim 2")
        plt.hist(variances[:, 1], alpha=0.4)
        plt.show()

    # est_var = np.sum(ns * variances) / np.sum(ns)
    return marginal_lls, est_var, all_kernels


class SGP:
    '''
    implements SGP 
    used for R^D->R (single dimension of output)
    '''

    def __init__(self, data, kernel_params, inducing_variables, sigma2, Kmm=None, Kmm_inv=None):
        self.name = 'SGP'
        from sklearn.gaussian_process.kernels import ConstantKernel, RBF
        self.X, self.Y = data
        k_var, k_ls = kernel_params
        k = ConstantKernel(constant_value=k_var) * RBF(length_scale=k_ls)
        self.kernel = k
        self.precision = 1. / sigma2
        self.Z = inducing_variables
        self.k_ls = k_ls
        self.k_var = k_var

        if Kmm is None:
            Kmm = k(self.Z) + 1e-8 * np.eye(self.Z.shape[0])
            self.Kmm = Kmm
        else:
            self.Kmm = Kmm
        if Kmm_inv is None:
            self.Kmm_inv = np.linalg.solve(Kmm, np.eye(self.Z.shape[0]))
        else:
            self.Kmm_inv = Kmm_inv

        Kmn = k(self.Z, self.X)
        Kmn2 = Kmn @ Kmn.T
        Sigma = Kmm + (self.precision) * Kmn2
        A_mu = np.linalg.solve(Sigma, Kmn)
        A_var = np.linalg.solve(Sigma, Kmm)

        mu_Z = (self.precision) * (Kmm @ A_mu @ self.Y)
        Sig_Z = Kmm @ A_var
        self.mu_Z = mu_Z
        self.Sig_Z = Sig_Z

        self.Kmn2 = Kmn2
        self.Kmn = Kmn

    def predict_f(self, X_new):
        Kmm = self.Kmm
        K_starZ = self.kernel(X_new, self.Z)
        K_star = self.kernel(X_new, X_new)

        Kmm_inv = self.Kmm_inv
        mu_star = K_starZ @ Kmm_inv @ self.mu_Z
        A = Kmm_inv @ K_starZ.T
        B = Kmm_inv @ self.Sig_Z
        Sigma_star = K_star - K_starZ @ A + K_starZ @ B @ A

        return (mu_star, (Sigma_star + Sigma_star.T)/2)


class GP:
    '''
    implements GP (similar to SGP)
    used for R^D->R (single dimension of output)
    '''

    def __init__(self, data, kernel_params, sigma2):
        self.name = 'GP'
        from sklearn.gaussian_process.kernels import ConstantKernel, RBF
        self.X, self.Y = data
        k_var, k_ls = kernel_params
        k = ConstantKernel(constant_value=k_var) * RBF(length_scale=k_ls)
        self.kernel = k
        self.sigma2 = sigma2
        self.k_ls = k_ls
        self.k_var = k_var

        K_train = k(self.X, self.X)
        self.A = np.linalg.inv(K_train + sigma2 * np.eye(self.X.shape[0]))

    def predict_f(self, X_new):
        A = self.A
        K_new = self.kernel(X_new, X_new)
        K_new_data = self.kernel(X_new, self.X)

        mu_star = K_new_data @ A @ self.Y
        Sigma_star = K_new - K_new_data @ A @ K_new_data.T

        return (mu_star, (Sigma_star + Sigma_star.T)/2)


def log_density(models, data, sigma2):
    X, Y = data
    P = len(models)
    ll = 0.0

    for j in range(P):
        model = models[j]
        pred_mean, pred_var = model.predict_f(X)
        pred_var = pred_var + sigma2[j] * np.eye(X.shape[0])
        try:
            a = multivariate_normal.logpdf(
                Y[:, j].flatten(),  mean=pred_mean.flatten(), cov=pred_var)
        except:
            q = Y[:, j].flatten() - pred_mean.flatten()
            D = X.shape[0]
            q1 = q.T @ np.linalg.solve(pred_var, q)
            a = - 0.5 * (D * np.log(2*np.pi) +
                         np.linalg.slogdet(pred_var)[1] + q1)
        ll += a
    return ll


def add_data_models(models, new_data, sigma2):
    '''
    models is a list of models (length P - for each output dimension)
    updates these using new_data and passed sigma2 (typically one estimated during Step1)
    returns list of models
    NEW: if input models are of type SGP/GP, return models are of same type
    '''

    X, Y = new_data
    new_models = []
    P = len(models)

    if models[0].name == 'SGP':
        for j in range(P):
            m = models[j]
            y = Y[:, j][:, None]
            m.X = np.vstack([m.X, X])
            m.Y = np.vstack([m.Y, y])
            m.precision = 1.0 / sigma2[j]

            new_K = m.kernel(m.Z, X)
            Kmn = np.hstack([m.Kmn, new_K])
            Kmn2 = m.Kmn2 + new_K @ new_K.T
            Sigma = m.Kmm + (m.precision) * Kmn2
            A_mu = np.linalg.solve(Sigma, Kmn)
            A_var = np.linalg.solve(Sigma, m.Kmm)

            mu_Z = (m.precision) * (m.Kmm @ A_mu @ m.Y)
            Sig_Z = m.Kmm @ A_var
            m.mu_Z = mu_Z
            m.Sig_Z = Sig_Z
            m.Kmn = Kmn
            m.Kmn2 = Kmn2

            new_models.append(m)

    elif models[0].name == 'GP':
        for j in range(P):
            m = models[j]
            y = Y[:, j][:, None]
            X_ = np.vstack([m.X, X])
            Y_ = np.vstack([m.Y, y])

            m = GP((X_, Y_), (m.k_var, m.k_ls), sigma2[j])
            new_models.append(m)

    return new_models


def fit_GP_new_kernel(data, sigma2, Z=None, N_max=500, type='SGP'):

    # function used to update kernel parameters (can be used without a previous model)
    # data = (X,Y) for all output dim
    # fits GP on random N_max points and uses the kernel to get SGP models
    # return (SGP1, SGP2) for 2 output dim

    if type == 'SGP' and Z is None:
        raise ValueError('Need inducing points for SGP')

    N = np.min([data[0].shape[0], N_max])
    X, Y = data

    idx = np.random.choice(data[0].shape[0], size=N, replace=False)
    X_ = X[idx]
    Y_ = Y[idx]

    # fit GP
    data_ = (X_, Y_)
    _, _, k_params = fit_GP(data_, sigma2=sigma2, return_model=False)

    m_new = []
    P = Y.shape[1]
    for j in range(P):
        k_params1 = (k_params[0][j], k_params[1][j])
        if type == 'SGP':
            m_new1 = SGP((X, Y[:, j][:, None]), k_params1, Z, sigma2[j])
        elif type == 'GP':
            m_new1 = GP((X, Y[:, j][:, None]), k_params1, sigma2[j])
        m_new.append(m_new1)

    return m_new


def fit_GP_old_kernel(data, sigma2, kernel_params, Z=None, type='SGP'):
    '''
    return GP/SGP (based on type) models on data (X,Y)
    one for each output dimension
    use (kernel_variances, kernel_lengthscales) = kernel_params to get the kernels
    '''
    k_var, k_ls = kernel_params
    X, Y = data
    P = Y.shape[1]
    if type == 'SGP' and Z is None:
        raise ValueError('Need Z when type is SGP')
    models = []
    for j in range(P):
        y = Y[:, j][:, None]
        if type == 'SGP':
            m = SGP((X, y), (k_var[j], k_ls[j]), Z, sigma2[j])
        elif type == 'GP':
            m = GP((X, y), (k_var[j], k_ls[j]), sigma2[j])
        models.append(m)
    return models
