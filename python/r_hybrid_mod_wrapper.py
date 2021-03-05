# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 08:11:22 2020

@author: u6026797
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 15:19:03 2020

@author: u6026797
"""

#%% Libraries
import pandas as pd
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
pandas2ri.activate()
#%% function  
rstring_hybrid="""
library(forecast)
library(WaveletArima)
r_ts_model <- function(ts, p_forecast=14){
    arm_fit<-auto.arima(y=ts, seasonal=FALSE, 
                        nmodels = 500, max.p = 10, max.q = 10, max.P = 10,
                        max.Q = 10, max.d = 10, max.order = 10, max.D = 10,
                        allowdrift = TRUE, allowmean = TRUE, parallel  = TRUE,
                        biasadj = TRUE)
    arm_out<-forecast(arm_fit,h = p_forecast)
    arm_forecast<-arm_out$mean
    arm_upper_ci<-arm_out$upper[1:14,2]
    arm_lower_ci<-arm_out$lower[1:14,2]
    
    WaveletForecast<-WaveletFittingarma(arm_fit$residuals,
                                        Waveletlevels=floor(log(length(arm_fit$residuals))),
                                        boundary='periodic',
                                        FastFlag=TRUE,
                                        MaxARParam=10,
                                        MaxMAParam=10,
                                        NForecast=14)
    
    #### Add the fitted ARIMA outouts to the fitted WBF model outputs
    hybrid_fit<-WaveletForecast$Finalforecast+arm_out$mean
    hybrid_fit<-hybrid_fit[1:14]
    output_df<-as.data.frame(do.call(cbind, list('forecast'=hybrid_fit,
                   'upper_ci'=arm_upper_ci+WaveletForecast$Finalforecast,
                   'lower_ci'=arm_lower_ci-WaveletForecast$Finalforecast)))
    return(output_df)
}
"""
rfunc=robjects.r(rstring_hybrid)

def r_hybrid_mod_wrapper(ts, p_forecast=[14]):
  '''
  Wrapper function for R wavelet+arima model via rpy2. Be sure that you have rpy2 
  and its dependencies installed. Specifically:
  
  import rpy2.robjects.packages as rpackages
  utils = rpackages.importr('utils')
  utils.chooseCRANmirror(ind=1)
  packnames = ('WaveletArima','forecast')
  from rpy2.robjects.vectors import StrVector
  names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
  if len(names_to_install) > 0:
      utils.install_packages(StrVector(names_to_install))
  
  Parameters
  ----------
  ts_data : int/float numeric vector
    Vector to be forecasted
  p_forecast : integer 
    N points to be forecasted (default is 14)
  Returns
  -------
  hybrid_fit : int/float vector
    Vector of 

  '''
  ts= robjects.IntVector(ts)
  p_forecast= robjects.IntVector(p_forecast)
  
  #output_dict = dict()
  hybrid_out = rfunc(ts=ts,
                     p_forecast=p_forecast)
  return hybrid_out

#%% thingy
'''
import random
index = list(range(1,101,1))
ts_data= list(range(1,101,1))

ts_data = [12,  2,  1, 12,  2,  3,  5,  8,  6,  4,  1,  2, 11,
        8,  5,  2,  9, 19,  1,  8,  4,  3,  2,  3,  3,  3,
        3,  3,  3,  3,  3]
index = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
for i in range(len(ts_data)):
  ts_data[i]=ts_data[i]*random.randint(1,10)

param_list_loess=np.arange(.2,.76,.05)
param_list_kz=list(range(3,26,1))



p,kz = r_smoother_wrapper(ts_data=ts_data,
                  index=index,
                  smoother='loess',
                  dparam=[.2],
                  param_jitter=[1])

#param_list_loess=np.arange(.2,.76,.05),
#param_list_kz=list(range(3, 25, 2))

#kza_out = rfunc(ts_data=ts_data,
#                index=index,
#                smoother='kz',  
#                  param_list_kz=param_list_kz)

#pandas2ri.ri2py(kza_out.rx2((i+1)).rx2('kz'))
'''