import numpy as np
import pandas as pd
import re
from  ratetransforms import *
import matplotlib.pyplot as plt

#loading data:

ratescalefactor= 100.0
ustreasury = pd.read_csv('data/USTREASURY-YIELD.csv',index_col=0)/ratescalefactor
ustreasury.columns = map(tenornormalizer,ustreasury.columns)
print(ustreasury.head())
tenor2years = tenors2yearsdict(list(ustreasury.columns))
print(tenor2years)

dframe = transformrates2discounts(ustreasury)
iforward = transformDiscountToInstantaneousForward(dframe,tyearstart=1.0/12.0,tyearend=30,N=30*6)
iforward.iloc[0:10,:].plot()
T=30
l6m = integrateForward(0.0,T,iforward)
print(f'Libor {T}y ={l6m}')
plt.show()
