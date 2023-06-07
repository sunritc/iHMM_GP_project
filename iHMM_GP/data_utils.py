import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")

K_true = 10  # true number of clusters
D = 2       # input location dimension
P = 2       # output field dimension


def generate_random_function(D=2, P=2, seed=None):

    # generates a random R_D -> R_P function

    rng = np.random.RandomState(seed)

    A = rng.uniform(-3, 3, size=(2, P))
    sin_coef = rng.uniform(0.1, 4, size=(P, D))
    cos_coef = rng.uniform(0.1, 4, size=(P, D))

    c = rng.uniform(-2, 2, size=P)

    def f(x):
        m1 = np.sin(sin_coef @ x)
        m2 = np.cos(cos_coef @ x)
        m = np.vstack((m1, m2))
        B = A * m
        y = B.sum(axis=0)
        y = y + c
        return y

    return f


def sum_functions(f1, f2, eps):

    def f(x):
        return (1 - eps) * f1(x) + eps * f2(x)

    return f


# code to make quiver plots
'''
f = generate_random_function(D=2, P=2, seed=2)
x,y = np.meshgrid(np.linspace(-1.05,1.05,10),np.linspace(-1.05,1.05,10))
u = np.zeros((10, 10))
v = np.zeros((10, 10))
for i in range(10):
    for j in range(10):
        val = f([x[i,j], y[i,j]])
        u[i,j] = val[0]
        v[i,j] = val[1]
plt.quiver(x,y, u,v)        
'''


def get_data(T, D=2, P=2, n=100, K_true=10, sigma=1, eps=0.5, f_true=None, Pi=None):

    # get true functions
    if f_true is None:
        f_true = []
        for k in range(K_true - 1):
            f_true.append(generate_random_function(D, P, seed=15*k))

        f_true.append(sum_functions(f_true[0], f_true[1], eps=eps))
    else:
        K = len(f_true)
        if K != K_true:
            print('Warning: given only ', K, ' functions')

    # get transition kernel
    if Pi is None:
        Pi = np.random.dirichlet(alpha=[5]*K_true, size=K_true)
    # else check it is a proper size transition matrix

    data = []

    # get states
    s = np.zeros(T, dtype=int)
    for t in range(T-1):
        s[t+1] = np.random.choice(K_true, p=Pi[s[t]])

    # starting
    for t in range(T):
        X = np.random.rand(n, D) * 2 - 1
        g = f_true[s[t]]
        Y = np.array([g(X[i]) for i in range(n)])
        Y = Y + np.random.normal(0, sigma, (n, P))
        data.append((X, Y))

    return data, s, f_true, Pi


'''# example to visualize data 
data, s, f_true, Pi = get_data(T=1000, n=200)

# visualize the functions
x,y = np.meshgrid(np.linspace(-1.05,1.05,10),np.linspace(-1.05,1.05,10))
u = np.zeros((10, 10))
v = np.zeros((10, 10))

fig = plt.figure(figsize=(22, 8))
for row in range(2):
    for col in range(5):
        idx = row * 5 + col
        f = f_true[idx]
        for i in range(10):
            for j in range(10):
                val = f([x[i,j], y[i,j]])
                u[i,j] = val[0]
                v[i,j] = val[1]
        plt.subplot(2, 5, idx+1).set_title('number'+str(idx))
        plt.quiver(x,y, u,v) 

fig.suptitle('True vector fields', fontsize=20)
fig.tight_layout()
plt.show()
'''
# visualize a particular time point

'''X1, Y1 = data[1]
x = X1[:, 0]
y = X1[:, 1]
u = Y1[:, 0]
v = Y1[:, 1]
plt.quiver(x, y, u, v)
'''


def get_ablation(data, tau, m):

    dataset_training = []
    dataset_test = []

    data_train = data.copy()
    data_test = []
    for t in tau:
        X, Y = data[t]
        idx = np.random.choice(len(X), m, replace=False)
        X_test = X[idx]
        Y_test = Y[idx]
        X_new = np.delete(X, idx, axis=0)
        Y_new = np.delete(Y, idx, axis=0)
        data_train[t] = (X_new, Y_new)
        data_test.append((t, X_test, Y_test))

    return (data_train, data_test)


def get_1params():
    def f1(x):
        x1 = x[0]
        return np.array([1.6*x1**2 - 0.8])

    def f2(x):
        x1 = x[0]
        return np.array([(x1-0.3)**2/3])

    def f3(x):
        x1 = x[0]
        return np.array([np.sin(10*x1) - 0.2 * np.cos(5*x1)])

    def f4(x):
        x1 = x[0]
        return np.array([0.3 + 0.6 * np.exp(x1) * np.sin(6*x1)])

    def f5(x):
        x1 = x[0]
        return np.array([- 0.6 * np.sin(10*x1) + 0.7 * np.cos(5*x1)])

    def f6(x):
        x1 = x[0]
        return np.array([-0.8 + 2 * np.exp(-4*x1**2)])

    f_true = [f1, f2, f3, f4, f5, f6]
    Pi = np.array([[0.3, 0.3, 0.2, 0.2, 0, 0],
                   [0, 0.3, 0.3, 0.2, 0.2, 0],
                   [0, 0, 0.3, 0.3, 0.2, 0.2],
                   [0.2, 0, 0, 0.3, 0.3, 0.2],
                   [0.2, 0.2, 0, 0, 0.3, 0.3],
                   [0.3, 0, 0, 0.2, 0.2, 0.3]])
    return f_true, Pi


def sigmoid(x, alpha=10):
    return 1 / (1 + np.exp(-alpha * x))


def merge(f1, f2, type=1):
    ''' type: 1 (merge left/right), 2 (up/down) '''
    if type == 1:
        def f(x):
            eps = sigmoid(x[0])
            return eps * f1(x) + (1-eps) * f2(x)
    else:
        def f(x):
            eps = sigmoid(x[1])
            return eps * f1(x) + (1-eps) * f2(x)

    return f


def region(z):

    group = []
    for i in range(len(z)):
        x = z[i][0]
        y = z[i][1]
        if (x >= 0) and (y >= 0):
            group.append(1)
        elif (x >= 0) and (y < 0):
            group.append(2)
        elif (x < 0) and (y >= 0):
            group.append(3)
        else:
            group.append(4)
    return np.array(group, dtype=int)


def get_data_new(T, f_true, Pi, n, sigma):

    K = len(f_true)
    s = np.zeros(T, dtype=int)
    data = []
    for t in range(T-1):
        s[t+1] = np.random.choice(K, p=Pi[s[t]])

    rand_regions = np.random.choice([1, 2, 3, 4], size=T)

    for t in range(T):
        X = np.random.rand(n, 2) * 2 - 1
        g = f_true[s[t]]
        Y = np.array([g(X[i]) for i in range(n)])
        Y[:, 0] = Y[:, 0] + np.random.normal(0, sigma[0], n)
        Y[:, 1] = Y[:, 1] + np.random.normal(0, sigma[1], n)

        labels = region(X)
        idx = np.where(labels == rand_regions[t])[0]
        X = np.delete(X, idx, axis=0)
        Y = np.delete(Y, idx, axis=0)
        data.append((X, Y))
    return data, s

# spline new section
# generate new functions using splines


def multi_f(f1, f2):
    # f1, f2 are spline objects
    def f(x):
        return np.array([f1.ev(x[0], x[1]), f2.ev(x[0], x[1])])

    return f


def generate_function_spline(s=10, k=50, seed=10):
    from scipy.interpolate import SmoothBivariateSpline as spln
    np.random.seed(seed)
    Z = np.random.rand(k, 2) * 2 - 1

    # boundary points - do not behave erratic near boundary
    bd = np.linspace(-1, 1, 10)
    Z1 = np.array([[1.01, bd[j]] for j in range(len(bd))])
    Z2 = np.array([[-1.01, bd[j]] for j in range(len(bd))])
    Z3 = np.array([[bd[j], 1.01] for j in range(len(bd))])
    Z4 = np.array([[bd[j], -1.01] for j in range(len(bd))])

    Z = np.vstack((Z, Z1, Z2, Z3, Z4))
    V1 = np.random.uniform(low=-1.5, high=1.5, size=k)
    V2 = np.random.uniform(low=-1.5, high=1.5, size=k)
    V_bd = np.random.uniform(low=-0.1, high=0.1, size=4*len(bd))
    V1 = np.append(V1, V_bd)
    V2 = np.append(V2, V_bd)

    f1 = spln(x=Z[:, 0], y=Z[:, 1], z=V1,
              bbox=[-1.1, 1.1, -1.1, 1.1], kx=5, ky=5, s=s)
    f2 = spln(x=Z[:, 0], y=Z[:, 1], z=V2,
              bbox=[-1.1, 1.1, -1.1, 1.1], kx=5, ky=5, s=s)

    f = multi_f(f1, f2)
    return f

# plotting these function


def plot_function(f, grid_size=400, dim=0, color='black'):

    # if dim=0 -> plot quiver plot, if 1/2 -> plot 3d scatter of that dimension output

    k = int(np.sqrt(grid_size))

    x, y = np.meshgrid(np.linspace(-1, 1, k), np.linspace(-1, 1, k))
    Z = np.array([[x[i, j], y[i, j]] for i in range(k) for j in range(k)])

    y1 = np.zeros(k*k)
    y2 = np.zeros(k*k)

    for j in range(len(Z)):
        y_ = f(Z[j])
        y1[j] = y_[0]
        y2[j] = y_[1]

    if dim == 0:
        fig = plt.quiver(Z[:, 0], Z[:, 1], y1, y2, color=color)

    elif dim == 1:
        import pandas as pd
        df = pd.DataFrame({"X": Z[:, 0], "Y": Z[:, 1], "Z": y1})
        import plotly.express as px
        fig = px.scatter_3d(df, x='X', y='Y', z='Z')
        fig.update_traces(marker={'size': 3})

    elif dim == 2:
        import pandas as pd
        df = pd.DataFrame({"X": Z[:, 0], "Y": Z[:, 1], "Z": y2})
        import plotly.express as px
        fig = px.scatter_3d(df, x='X', y='Y', z='Z')
        fig.update_traces(marker={'size': 3})

    return fig


###### for 1-d cases #####
def create_f1d(seed=10, plot=True):
    from scipy.interpolate import UnivariateSpline
    np.random.seed(seed)
    x_ = np.linspace(0.101, 0.899, 8)
    y_ = np.random.uniform(low=-2, high=2, size=8)
    
    x = np.append(0, np.append(x_, 1))
    y = np.append(0, np.append(y_, 0))
    spl = UnivariateSpline(x, y, bbox=(0, 1), k=3, s=1)
    z = np.linspace(0,1,200)
    f = spl(z)
    if plot:
        plt.plot(z, f)
        plt.scatter(x, y, color='blue')
        plt.show()
        
    return spl
              
    
def plot_time1d(t):
    X, Y = full_data[t]
    g = f_true[s[t]]
    z = np.linspace(0,1, 200)
    val = g(z)
    
    plt.figure(figsize=(6, 5))
    plt.plot(z, val, color="gray")
    plt.scatter(X, Y, color='blue')
    plt.show()