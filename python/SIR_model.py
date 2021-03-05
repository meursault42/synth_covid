# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 16:10:45 2021

@author: u6026797
"""

#%% libraries
import pandas as pd
import numpy as np
from numpy import exp
#import matplotlib.pyplot as plt
import math 
#import seaborn as sns
import random
from statsmodels.tsa.stattools import pacf
from scipy.integrate import odeint
from scipy.optimize import minimize
from sklearn import preprocessing

#%% init params / data
pop_size = 1000000

i_0 = 1e-6
ai_0 = 1e-6
e_0 = 4 * (i_0+ai_0)
s_0 = 1 - (i_0+ai_0) - e_0
d_0 = 0
ot_0 = 0
r_0 = 0

x_0 = s_0, e_0, ai_0, i_0, ot_0, d_0, r_0

t_length = 550
grid_size = 545
t_vec = np.linspace(0, t_length, grid_size)
t_vec = np.array(range(0,t_length))
#%% functions
def _F(x, t, init_params, R0=1.6):
    '''
    SIR model used for generating disease outbreak compartments. 

    Parameters
    ----------
    x : list
        Holds components of SIR model for iteration
    t : np array
        Outbreak sequence length.
    init_params : vector of floats
        Coefficients for SIR model
    R0 : float, optional
        value of R0 at time t. Defaults to static R0 value of 1.6.

    Returns
    -------
    ds : TYPE
        DESCRIPTION.
    de : TYPE
        DESCRIPTION.
    dai : TYPE
        DESCRIPTION.
    di : TYPE
        DESCRIPTION.
    dot : TYPE
        DESCRIPTION.
    dd : TYPE
        DESCRIPTION.
    dr : TYPE
        DESCRIPTION.

    '''
    s, e, ai, i, ot, d, r = x
    
    infection_rate = init_params[0]
    recovery_rate = init_params[1]
    mortality_rate = 0.009
    asymptomatic_rate = 0.3
    a_infection_rate = init_params[2]
    a_test_rate = init_params[3]
    test_rate = init_params[4]
    a_recovery_rate = init_params[5]
    a_trans_rate = init_params[6]
    trans_rate = init_params[7]

    # New exposure of susceptibles
    transmission_rate = R0(t) if callable(R0) else R0 
    new_exposed = (transmission_rate * trans_rate * i) + (transmission_rate *a_trans_rate * ai) * s 

    # Time derivatives
    ds = - new_exposed
    de = new_exposed - (infection_rate * e) - (a_infection_rate * e)
    #asymptomatic cases
    dai = asymptomatic_rate * a_infection_rate * e - a_recovery_rate * ai
    #symptomatic cases
    di = (1- asymptomatic_rate) * infection_rate * e - recovery_rate * i    
    #observerable test
    #dai -> ai
    dot = (a_test_rate * ai) + (test_rate * i) - ot
    dd = mortality_rate * i - d
    #dai -> ai
    dr = ((1 - mortality_rate) * i) + (a_recovery_rate * ai) - r
    
    return ds, de, dai, di, dot, dd, dr

def _solve_path(R0, t_vec, init_params = [1/5.2,1/18,1/20], x_init=x_0):
    """
    Solve for i(t) and c(t) via numerical integration,
    given the time path for R0.
    """
    G = lambda x, t: _F(x, t, init_params, R0)
    s_path, e_path, ai_path, i_path, ot_path, d_path, r_path = odeint(G, x_init, t_vec).transpose()

    c_path = 1 - s_path - e_path - r_path      # cumulative cases
    return i_path, c_path, d_path, ot_path

def Rtp_mitigating(t, r0=3, intervention=0.007, r_bar=0.8, fpc=8, spc=100, fcw=.05, scw = 0.2):
    '''
    Function for generating Rt sequence with periodic noise.

    Parameters
    ----------
    t : np array
        placeholder vector for sequence length
    r0 : float, optional
        Intercept for start of Rt. The default is 3.
    intervention : float, optional
        Value for changing the slope for the exponential function. 
        The default is 0.007.
    r_bar : float, optional
        Value for tail of exponential function. The default is 0.8.
    fpc : int, optional
        Width of fast periodic component window. The default is 8.
    spc : int, optional
        Width of slow periodic component window. The default is 100.
    fcw : int, optional
        Weight applied to fast periodic component. The default is .05.
    fcw : int, optional
        Weight applied to slow periodic component. The default is .05.

    Returns
    -------
    Rt : np.array
        Vector of simulated Rt.

    '''
    Rt = (r0*exp(-intervention*t)+(1-exp(-intervention*t))*r_bar) * \
        (1+(np.sin((2*math.pi/fpc)*t))*fcw) * \
        (1+(np.sin((2*math.pi/spc)*t))*scw)
    return Rt

def Rtg_mitigating(t,r0=3,intervention=.1,r_bar=.9,n_peaks=2):
    '''
    Function for generating Rt sequence with subsequent peaks of a width 
    and height defined by a gaussian distribution.

    Parameters
    ----------
    t : np array
        place holder vector for sequencel length.
    r0 : float, optional
        Intercept for start of Rt. The default is 3.
    intervention : float, optional
        Value for changing the slope for the exponential function. 
        The default is 0.007.
    r_bar : float, optional
        Value for tail of exponential function. The default is 0.8.
    n_peaks : integer, optional
        Number of desired subsequent peaks in Rt. The default is 2.

    Returns
    -------
    Rt: np.array
        Vector of simulated Rt.
    '''
    trunc_normal_dist=np.random.normal(2,.5,size=1000)
    trunc_normal_dist[trunc_normal_dist<.7]=.7
    trunc_normal_dist[trunc_normal_dist>3]=3
    
    Rt_exp = (r0*np.exp(-intervention*t)+(1-np.exp(-intervention*t))*r_bar)
    peak_dict=dict()
    for i in range(0,n_peaks):
        key_name='center_'+str(i)
        peak_dict[key_name]=random.sample(list(range(50,(len(t)-50))),1)
    #generate N dist centered at position n
    randomNums = np.random.normal(100,12,size=100000)
    #round to discretize
    randomInts = np.round(randomNums)
    #send to pd to aggregate
    randdf = pd.DataFrame({'Vals':randomInts,'count':1})
    randdf.groupby(['Vals']).sum()
    #rescale to make feasible 
    randdf = randdf.groupby(['Vals']).sum()
    x=randdf['count'].to_numpy()
    x=x.reshape((1,-1))
    x=preprocessing.normalize(x)[0]
    
    if len(x)%2!=0:
        start_p=-math.floor(len(x)/2)
        end_p=len(x)+start_p
    else:
        start_p = -len(x)/2
        end_p = len(x)/2
    for key,val in peak_dict.items():
        Rt_exp[int(val[0]+start_p):int(val[0]+end_p)]=Rt_exp[int(val[0]+start_p):int(val[0]+end_p)]+x*random.sample(trunc_normal_dist.tolist(),1)
    return Rt_exp#, peak_dict

def L1(init_params,comp_vec_i=comp_vec_i,pcc=.1):
    R0 = lambda t: R0_mitigating(t)
    i_path, c_path, d_path, ot_path = solve_path(R0, t_vec, init_params=init_params)
    #force length
    d_path = d_path[:len(comp_vec_i)]
    #ot_path = ot_path[:len(comp_vec_i)]
    out = [path * pop_size for path in d_path]
    #resid_vec = comp_vec_i - out 
    #loss_out = sum(resid_vec**2)
    #loss_out = sum(np.round((abs(out-comp_vec_i)/comp_vec_i),4))
    #Sum Absolute Error
    loss_out = sum(abs(out-comp_vec_i))
    return loss_out



#%% generate sequences

