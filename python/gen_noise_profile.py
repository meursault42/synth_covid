# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 13:33:53 2021

@author: u6026797
"""

#%% Libraries

import pandas as pd
import numpy as np
from math import floor, ceil
from statsmodels.tsa.stattools import pacf
from scipy.stats import norm
from hmmlearn import hmm
import random 
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import STL
from copy import deepcopy

#this must go--it simply must
from r_wrapper_aa_v1 import r_ts_fit, r_arma_noise_gen

import os, contextlib

#%% Functions
def _supress_stdout(func):
    def wrapper(*a, **ka):
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                func(*a, **ka)
    return wrapper 

def _bidirround(value):
    if value<0:
        return(floor(value))
    if value>0:
        return(ceil(value))

def _shuffle_permute(input_vec,shuffle_prob=30):
    input_vec = input_vec.astype(float)
    permute_index = []
    for index, values in np.ndenumerate(input_vec):
        flip_logic = (random.randint(1,shuffle_prob)==1)
        if flip_logic == True:
            permute_index.append(index[0])
            shuffle_mag_dist = np.random.normal(.75,.1,1)
            shuffle_day_dist = _bidirround(np.random.normal(0,1,1))
            change_value=input_vec[index[0]]-(input_vec[index[0]]*shuffle_mag_dist)
            if (index[0]+shuffle_day_dist)<0 or (index[0]+shuffle_day_dist)>(len(input_vec)-1):
                continue
            elif abs(shuffle_day_dist)==1:
                input_vec[index[0]]=input_vec[index[0]]*shuffle_mag_dist
                input_vec[(index[0]+shuffle_day_dist)]+=change_value
            elif abs(shuffle_day_dist)>1:
                input_vec[index[0]]=input_vec[index[0]]*shuffle_mag_dist
                change_value = change_value/abs(shuffle_day_dist)
                start=min(shuffle_day_dist,0)
                end=max(shuffle_day_dist,0)
                if end==0:
                    for i in range(start,end):
                        input_vec[(index[0]+i)]+=change_value
                elif end>0:
                    for i in range((start+1),(end+1)):
                        input_vec[(index[0]+i)]+=change_value
    return input_vec, permute_index

def _chain_noise_gen(chain,n_vec):
    '''
    Genereates a white noise sequence. 

    '''
    delist_chain = chain[0]
    flat_list = [item for sublist in n_vec for item in sublist]
    out_seq=[]
    for i in delist_chain:
        out_seq.append(r_arma_noise_gen(i[0],i[0],random.sample(flat_list,1)[0]))
    out_seq = np.concatenate(out_seq)
    out_seq = out_seq/sum(out_seq)
    return(out_seq)

def _weight_chain_avg(input_ts,chain_noise_seq,weight=0.1):
    '''
    Calculates weighted average of two (x,) dimensional np arrays
    '''
    chain_noise_seq = chain_noise_seq[0:len(input_ts)]
    w_chain_noise=(chain_noise_seq)*weight*2
    w_raw_seq=(input_ts)*(2-(weight*2))
    w_c_avg=np.mean((w_chain_noise,w_raw_seq),axis=0)
    return w_c_avg

def _chain_smoother(input_vec,chain):
    '''
    Smoothes transitions between chains by averaging edges.
    '''
    input_vec_copy=input_vec
    indexer = -1
    for i in chain:
        indexer += i
        if indexer < (len(input_vec)-3):
            tail = np.mean(input_vec_copy[indexer-1:indexer+2])
            head = np.mean(input_vec_copy[indexer:indexer+3])
            input_vec_copy[indexer]=tail
            input_vec_copy[indexer+1]=head
        
    return input_vec_copy

def _nth_order_sort(seq,n_order):
    '''
    Parameters
    ----------
    seq : A Numpy array
        The input vector to be sorted
    n_order : Integer
        The n from top item desired. IE return the second highest = 2

    Returns
    -------
    TYPE
        Returns the nth from top item from the input sequence.

    '''
    for i in range(n_order):
        seq_max = max(seq)
        seq = seq[seq!=seq_max]
    return max(seq)

def _emission_hmm(input_vec,start_window=40,start_value=0,prior_step=0,
                 prior_magnitude=0,n_order=0, prior_dir_dict=0, cutoff = 0.05,
                 prior_noise=0,likely_index=[5,6,7,8,9,10,11,12,13]):
    '''
    Function recursively iterates through a given array and returns the 
    nth order periodic component found, along with the magnitude and
    the averaged direction vector. 

    Parameters
    ----------
    input_vec : Numpy Array
        Input array from which noise is estimated.
    start_window : Integer, optional
        The window for searching for noise. The default is 40. Function may
        fail if the window is too small.
    start_value : Integer, optional
        Used for managing recursive indexing.
    prior_step : List, optional
        Output list that holds prior results. The default is 0.
    prior_magnitude : List, optional
        Output list that holds prior noise magnitude. The default is 0.
    n_order : Integer, optional
        Desired rank of noise derived by pACF. 0 = max, 1 = 2nd highest component. 
        The default is 0.
    prior_dir_dict : dict, optional
        Holds magnitude of observed noise. The default is 0.
    cutoff : Float, optional
        Cutoff value for determining un/useful noise components by pACF coefficient.
        The default is 0.05. Values below this are presumed to be white noise.
    prior_noise : List, optional
        List used to hold noise shapes after stl decomposition. The default is 0.
    likely_index : list, optional
        List holds subset of pACF of interest. The default is 5:13

    Returns
    -------
    Numpy Array
        output_vec: Holds emmission outputs for use in subsequent modelling.
    Numpy Array
        magnitude_vec: Holds coefficients from pACF for use in later noise 
        generation.
    Numpy Array
        shape_dict: Holds magnitudes of periodic noise components for use in 
        later noise generation.
    list
        noise_list: Holds arima specifications for residual noise of sequences 

    '''
    ##check for prior_step else build array
    if isinstance(prior_step, np.ndarray):
        output_vec = prior_step
    else:
        output_vec = np.array([])
    ##check for prior_magnitude else build array
    if isinstance(prior_magnitude, np.ndarray):
        magnitude_vec = prior_magnitude
    else:
        magnitude_vec = np.array([])
    ##check for prior_dir_array else build array
    if isinstance(prior_dir_dict, dict):
        shape_dict = prior_dir_dict
    else:
        shape_dict = {'5_day':[],
                      '6_day':[],
                      '7_day':[],
                      '8_day':[],
                      '9_day':[],
                      '10_day':[],
                      '11_day':[],
                      '12_day':[],
                      '13_day':[],
                      'cutoff_null':[]}
    if isinstance(prior_noise, list):
        noise_list = prior_noise
    else:
        noise_list = []
    adj_window = min(start_window,(len(input_vec[start_value:])))
    adj_vec = input_vec[start_value:(start_value+adj_window)]
    #check if end of available seq < search window
    if len(adj_vec)>=start_window:
        pacf_out = pacf(adj_vec,nlags=(floor(start_window/2)-1))
        max_val = _nth_order_sort(pacf_out[likely_index],n_order=n_order)
        if max_val < cutoff:
            #if below cutoff then simply generate guassian noise for a week
            max_pacf_val = 7
            key = 'cutoff_null'
            mu, std = norm.fit(adj_vec[:max_pacf_val])
            s = np.random.normal(mu,std,max_pacf_val)
            qpc = (s/np.sum(s))
            shape_dict[key].append(qpc)
        else:
            max_pacf_val = np.where(pacf_out==max_val)[0][0]
            
            #extract dirction of periodic component as %change from mean
            seq_sum = np.sum(adj_vec[:max_pacf_val])
            qpc = (adj_vec[:max_pacf_val]/seq_sum)
            #add to dict
            key = str(max_pacf_val)+'_day'
            shape_dict[key].append(qpc)
            
            output_vec = np.append(output_vec, max_pacf_val)
            magnitude_vec = np.append(magnitude_vec, max_val)
            
            noise_list.append(r_ts_fit(adj_vec,max_pacf_val))
            #noise_list.append(py_stl_ar_coef(adj_vec,max_pacf_val))
        return (_emission_hmm(input_vec=input_vec,\
                      start_value=(max_pacf_val+start_value),\
                      prior_step=output_vec,\
                      prior_magnitude=magnitude_vec,\
                      n_order=n_order,\
                      prior_dir_dict = shape_dict,
                      prior_noise = noise_list))
    else:
        return output_vec, magnitude_vec, shape_dict, noise_list
    
def get_hmm_outputs(subset_index,input_data_df,n_order=0):
    '''
    Wrapper function that extracts periodic noise from a provided input df
    then generates and trains a hidden markov model on those chained periodic
    noise components.

    Parameters
    ----------
    subset_index : list
        A list of unique indexes for the input_data_df. Example: state names, 
        zip codes, etc
    input_data_df : Pandas DataFrame
        A dataframe containing at least two columns: index which lists a index
        value as described above and noise_seq, each cell of which should contain
        an np.array of the sequence of interest. 
        Example: 
            ['Index', 'noise_seq']
            'Washington', [1,2,3,...]
    n_order : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    None.

    '''
    emm_vec = np.array([])
    mag_vec = np.array([])
    a_list = []
    b_list = []
    dir_dict = {'5_day':[],
                 '6_day':[],
                 '7_day':[],
                 '8_day':[],
                 '9_day':[],
                 '10_day':[],
                 '11_day':[],
                 '12_day':[],
                 '13_day':[],
                 'cutoff_null':[]}
    n_list = []
    def _dict_key_merger(dict1,dict2):
        for key in dir_dict:
            for seq in dict2[key]:
                dict1[key].append(seq)
    
    for ind in subset_index:
        data_vec_loc = input_data_df[input_data_df['index']==ind]
        data_vec_loc = data_vec_loc['noise_seq'].to_numpy()[0]
        emm, mag, dir_dict_loc, nn = _emission_hmm(input_vec=data_vec_loc,n_order=n_order)
        emm=emm.astype(int)
        emm=emm.reshape(-1,1)
        a_list.append(emm)
        b_list.append(mag)
        n_list.append(nn)
        _dict_key_merger(dir_dict, dir_dict_loc)
    
    emm_vec = np.concatenate(a_list)
    #lengths = [len()] 
    a_lens = [len(i)for i in a_list]
    
    mag_vec = np.concatenate(b_list)
    model = hmm.MultinomialHMM(n_components=9)
    model.fit(emm_vec, a_lens)
    
    return(model,mag_vec,dir_dict, n_list)

def _gen_hmm_noise_seq(input_ts,model,mag_vec,dir_dict):
    '''
    This function takes the output of get_hmm_outputs and creates a Hidden
    Markov Model then samples from it to generate a noise sequence similar the
    observed sequence.

    Parameters
    ----------
    input_ts : Numpy Array
        An array of values that you wish to add noise to.
    model : hmm model output
        the output hmm chain from get_hmm_outputs
    mag_vec : list
        List of pACF coefficients produced by get_hmm_outputs
    dir_dict : dictionary
        Dictionary of magnitudes produced by get_hmm_outputs 

    Returns
    -------
    input_ts_c : Numpy Array
        A copy of input_ts with hmm derived noise.

    '''
    input_ts_len = len(input_ts)
    start_ind = 0
    stop_ind = 0
    input_ts_c = deepcopy(input_ts)
    local_dir_dict = dict()
    hmm_chain = model[0]
    magnitude_distribution = mag_vec[mag_vec<1] 
    
    #randomly select single shape for each length
    for shape in dir_dict:
        local_dir_dict[shape]=random.sample(dir_dict[shape],1)[0]    
    
    for i in hmm_chain.tolist():
        stop_ind += i[0]
        if stop_ind<=len(input_ts_c)-1:
            magnitude=random.sample(magnitude_distribution.tolist(),1)[0]
            shape_key=str(i[0])+'_day'
            #shape=random.sample(shape_dict[shape_key],1)[0]
            shape=local_dir_dict[shape_key]
            w_periodic_component=(np.sum(input_ts_c[start_ind:stop_ind])*shape)*magnitude*2
            w_raw_seq=(input_ts_c[start_ind:stop_ind])*(2-(magnitude*2))
            input_ts_c[start_ind:stop_ind]=np.mean((w_periodic_component,w_raw_seq),axis=0)
            start_ind += i[0]
        elif stop_ind>len(input_ts_c)-1:
            append_seq = np.ones(((abs(stop_ind-len(input_ts_c)))))
            input_ts_c=np.append(input_ts_c,append_seq,0)
            magnitude=random.sample(magnitude_distribution.tolist(),1)[0]
            shape_key=str(i[0])+'_day'
            #shape=random.sample(shape_dict[shape_key],1)[0]
            shape=local_dir_dict[shape_key]
            w_periodic_component=(np.sum(input_ts_c[start_ind:stop_ind])*shape)*magnitude*2
            w_raw_seq=(input_ts_c[start_ind:stop_ind])*(2-(magnitude*2))
            input_ts_c[start_ind:stop_ind]=np.mean((w_periodic_component,w_raw_seq),axis=0)
            start_ind += i[0]
    input_ts_c = input_ts_c[0:input_ts_len]
    return input_ts_c

def generate_noise_sequences(input_ts_list, subset_index, input_data_df,
                             n_periodic_components=1, white_noise=True, 
                             white_noise_weight = 0.05, shuffle_permute=True,
                             shuffle_prob = 30, smooth_transitions=False):
    '''
    Wrapper function that takes in a list of np arrays, then adds modelled, 
    structured noise to replicate noise seen in the covid-19 pandemic daily
    new cases reported.

    Parameters
    ----------
    input_ts_list : list
        A list of np arrays that will have noise added to them.
    subset_index : list
        A list of unique indexes for the input_data_df. Example: state names, 
        zip codes, etc
    input_data_df : Pandas DataFrame
        A dataframe containing at least two columns: index which lists a index
        value as described above and noise_seq, each cell of which should contain
        an np.array of the sequence of interest. 
        Example: 
            ['Index', 'noise_seq']
            'Washington', [1,2,3,...]
    n_periodic_components : int, optional
        The number of periodic components desired. 1= only max pACF, 2=top 2 etc.
        The default is 1.
    white_noise : Boolean, optional
        Option to include white noise derived from input sequence. The default is True.
    white_noise_weight : float, optional
        Weight applied to white noise. The default is 0.05
    shuffle_permute : Boolean, optional
        Option to apply a unidirectional shuffle as observed in the covid-19 
        pandemic. This distributes a percent of a days values backwards or 
        forwards in time a random 1-3 days. The defaul is True.
    shuffle_prob: int, optional
        Denominator of shuffle permute occurring on any given day. Default is
        approximately once a month (1/30).    
    smooth_transitions: Boolean, optional
        Option to smooth noise chains between sequences. The default is False.

    Returns
    -------
    A input_ts_list with selected, modelled noise.

    '''
    out_list=[]
    periodic_comp_dict = {}
    #generate nth order periodic noise components and white noise set
    for n in list(range(0,n_periodic_components)):
        print('Modelling {} order noise. May take a minute.'.format(n))
        model, mag_vec, dir_dict, n_vec = get_hmm_outputs(subset_index=subset_index,
                                                          input_data_df=input_data_df,
                                                          n_order=n)
        #k_list = [str(n)+'_model',str(n)+'_mag_vec',str(n)+'_dir_dict',str(n)+'_n_vec']
        periodic_comp_dict[str(n)+'_model']=model
        periodic_comp_dict[str(n)+'_mag_vec']=mag_vec
        periodic_comp_dict[str(n)+'_dir_dict']=dir_dict
        periodic_comp_dict[str(n)+'_n_vec']=n_vec
    
    for seq in input_ts_list:
        #calulate minimum chain length
        chain_len=5*len(seq)
        #add periodic noise
        seq_c = deepcopy(seq)
        for n in list(range(0,n_periodic_components)):
            hmm_noise_sample = periodic_comp_dict[str(n)+'_model'].sample(chain_len)
            seq_c = _gen_hmm_noise_seq(seq_c,
                                    hmm_noise_sample,
                                    periodic_comp_dict[str(n)+'_mag_vec'],
                                    periodic_comp_dict[str(n)+'_dir_dict'])
            if smooth_transitions==True:
                seq_c =_chain_smoother(seq_c,np.reshape(hmm_noise_sample[0],(chain_len,)))
        #add white noise
        if white_noise==True:
            chain_noise_seq = _chain_noise_gen(periodic_comp_dict['0_model'].sample(chain_len),
                                               periodic_comp_dict['0_n_vec'].sample(chain_len))
            seq_c = _weight_chain_avg(seq_c,chain_noise_seq,weight=white_noise_weight)
        #add shuffle permutations
        if shuffle_permute==True:
            seq_c = shuffle_permute(seq_c)
        ###return seq
        out_list.append(seq_c)
    return(out_list)
