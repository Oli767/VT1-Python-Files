import numpy as np
import pandas as pd

def gbm(x0 = 10, mu=1, sigma=0, n_p=10, n_s=100, dt=1):
    """
    Geometric Brownian Motion (GBM)

    Parameters:
    xo (float): Value of x at t=0 (initialization value)
    mu (float): description
    sigma (float): description
    n_p (int): Number of periods to model
    n_s (int): Number of scenarios to be created
    dt: description

    Returns:
    x (float): description

    """

    x0_vect = x0 * np.ones((n_s,1))

    x = np.exp((mu - sigma ** 2 / 2) * dt \
    + sigma * np.random.normal(0, np.sqrt(dt), size=(n_s, n_p)))

    x = x0 * x.cumprod(axis=1)
        
    x = np.concatenate((x0_vect, x),axis=1)


    return x


def df_gbm(x0 = 10, mu=1, sigma=0, n_p=10, n_s=100, dt=1, year0=2023):
    """
    Geometric Brownian Motion (GBM)

    Parameters:
    xo (float): Value of x at t=0 (initialization value)
    mu (float): description
    sigma (float): description
    n_p (int): Number of periods to model
    n_s (int): Number of scenarios to be created
    dt: description

    Returns:
    df (Pandas dataframe): description

    """

    x0_vect = x0 * np.ones((1,n_s))

    x = np.exp((mu - sigma ** 2 / 2) * dt \
    + sigma * np.random.normal(0, np.sqrt(dt), size=(n_p, n_s)))

    x = x0 * x.cumprod(axis=0)    
    x = np.concatenate((x0_vect, x),axis=0)

    df = pd.DataFrame()

    df['year'] = np.arange(year0,year0+n_p+1,1)

    tmp = np.zeros(n_p)
    tmp = np.zeros(n_p+1)
    tmp[0]=x0
    for i in range(1,n_p+1):
        tmp[i] = tmp[i-1] * (1 + mu)
    df['determ']=tmp

    df_new = pd.DataFrame(x)
    df_new.columns = ['Scen'+str(i) for i in range(0,n_s)]

    df = df.join(df_new)

    return df


def bm(x0=100, mu=1, sigma=0, n_p=10, n_s=100, dt=1, year0=2023):
    """
    Standard Brownian Motion (BM) with drift = 0
    
    Arguments
    ---------
    x0 : float
        The initial condition
    n_p : int
        Number of periods/steps to simulate
    n_s : int
        Number of simulations to generate
    dt : float
        Time step
    
    Returns
    -------
    x : numpy array of floats with size (n_p, n_s)
        Simulated path followin BM

    """


    x0_vect = x0 * np.ones((1,n_s))

    x = np.random.normal(0, np.sqrt(dt), size=(n_p, n_s))
    x = np.concatenate((x0_vect, x),axis=0)
    x = np.cumsum(x,axis=0)

    return x

def bm_drift(x0=100, mu=1, sigma=1, n_p=10, n_s=100, dt=1, year0=2023):
    """
    Standard Brownian Motion (BM) with drift = mu
    
    Arguments
    ---------
    x0 : float
        The initial condition
    n_p : int
        Number of periods/steps to simulate
    n_s : int
        Number of simulations to generate
    dt : float
        Time step
    
    Returns
    -------
    x : numpy array of floats with size (n_p, n_s)
        Simulated path followin BM

    """


    x0_vect = x0 * np.ones((1,n_s))

    dx = (mu * dt) + (sigma * np.random.normal(0, np.sqrt(dt), size=(n_p, n_s)))
    x = np.concatenate((x0_vect, dx),axis=0)
    x = np.cumsum(x,axis=0)

    return x