import numpy as np
import pandas as pd
import re
from scipy.interpolate import interp1d
from scipy.misc import derivative
from scipy.integrate import quad,quadrature
from tools import *

def tenornormalizer(x:str):
    x = re.sub('MO','m',x)
    x = re.sub('\s+','',x)
    return re.sub('YR','y',x)

def tenor2years(x:str):
    months = re.search('(\d+.*\d*)m',x)
    years = re.search('(\d+.*\d*)y',x)
    if months:
        return x,float(months.group(1))/12.0  
    elif years:
        return x,float(years.group(1))
    else:
        raise Exception("invalid tenor format")  

def toDiscount(years:float,rate:float):
    if years<=1:
        return 1.0/(1.0+rate*years)
    else:
        return np.exp(-rate*years)   

def tenors2yearsdict(tenors):
    return dict(map(lambda x: tenor2years(x),tenors))


def transformrates2discounts(frame:pd.DataFrame):
    tenordict = tenors2yearsdict(frame.columns)
    nframe = frame.copy()
    for col in nframe.columns:
        nframe[col]= toDiscount(tenordict[col],frame[col])
    return nframe

def transformDiscountToInstantaneousForward(frame:pd.DataFrame):
    xnew = np.array(list(map(lambda x:tenor2years(x)[1], frame.columns)))
    xnew.sort()
    xnew = np.array(list(map(lambda s: np.round(s,decimals=4),xnew)))    
    data = np.zeros((frame.shape[0],len(xnew)))
    
    for i in range(frame.shape[0]):
        y = np.log(frame.iloc[i,:].values)
        f = interp1d(xnew, y,kind='cubic',assume_sorted=True,fill_value='extrapolate')
        for j in range(len(xnew)):
            data[i,j]= -derivative(f, xnew[j],dx=3.0/360)
    frame = pd.DataFrame(data=data,columns=xnew,index=frame.index)
    frame.sort_index(inplace=True)
    return frame


def integrateForward(t:float, T:float,iforward:pd.DataFrame):
    x = list(iforward.columns)
    y = iforward.iloc[0,:]
    f = interp1d(x, y,kind='cubic',assume_sorted=True,fill_value='extrapolate')
    return 1/T*quad(f, 0, T)[0]