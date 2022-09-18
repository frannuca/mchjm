import numpy as np
import pandas as pd
import re
from scipy.interpolate import interp1d
from scipy.misc import derivative
from scipy.integrate import quad,quadrature
from tools import *
from typing import Dict, Tuple, Sequence


def tenornormalizer(x:str):
    #converts a tenor input string format to normalized one. For instance 1 MO converts into 1m, or 10 YR into 10y.
    x = re.sub('MO','m',x)
    x = re.sub('\s+','',x)
    return re.sub('YR','y',x)

def tenor2years(x:str):
    #given a normalized tenor string such as '1m' converts it to its numerical value in years. For instance 3m translates into 0.25

    months = re.search('(\d+.*\d*)m',x)
    years = re.search('(\d+.*\d*)y',x)
    if months:
        return x,float(months.group(1))/12.0  
    elif years:
        return x,float(years.group(1))
    else:
        raise Exception("invalid tenor format")  

def toDiscount(years:float,rate:float):
    #converts an input rate or yield quote into discounting factor or zero bond price with maturity in the number of years specified.
    #years: maturity of the underlyinh zero
    #rate: yield quote. For maturities less than 1year we use yearly compounding rates. For T larger that 1year continuous discounting is utilized.

    if years<=1:
        return 1.0/(1.0+rate*years)
    else:
        return np.exp(-rate*years)   

def tenors2yearsdict(tenors):
    #given a list of normalized tenor string returns the discution of 'normalize tenor' -> 'number of years (float)'
    return dict(map(lambda x: tenor2years(x),tenors))


def transformrates2discounts(frame:pd.DataFrame):
    #given an input pandas Dataframe with input yield rates this function converts it into a dataframe of discount factors.

    tenordict = tenors2yearsdict(frame.columns)
    nframe = frame.copy()
    for col in nframe.columns:
        nframe[col]= toDiscount(tenordict[col],frame[col])
    return nframe

def transformDiscountToInstantaneousForward(frame:pd.DataFrame):
    #given an input pandas Dataframe with input discount factors this function converts it into a dataframe of instantaneous forward rates.

    xnew = np.array(list(map(lambda x:tenor2years(x)[1], frame.columns)))
    xnew.sort()
    xnew = np.array(list(map(lambda s: np.round(s,decimals=9),xnew)))    
    data = np.zeros((frame.shape[0],len(xnew)))
    
    for i in range(frame.shape[0]):
        y = np.log(frame.iloc[i,:].values)
        f = interp1d(xnew, y,kind='cubic',assume_sorted=True,fill_value='extrapolate')
        for j in range(len(xnew)):
            data[i,j]= -derivative(f, xnew[j],dx=1.0/360)
    frame = pd.DataFrame(data=data,columns=xnew,index=frame.index)
    frame.sort_index(inplace=True)
    return frame


def integrateForward(t:float, T:float,iforward:pd.DataFrame):
    #Given a starting time t, tenor T and an instantaneous forward data frame this function computes the associted F(to,t,T)

    x = list(iforward.columns)
    y = iforward.iloc[0,:]
    f = interp1d(x, y,kind='cubic',assume_sorted=True,fill_value='extrapolate')
    return 1/T*quad(f, 0, T)[0]

def forward(frate,t:float,T:float):
    #Standard forward calculation given input yield rates, starting forward time t and tenor T
    
    return (np.exp(frate(T+t)*(T+t)-frate(t)*t)-1.0)/(T)