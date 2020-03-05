import numpy as np
import pandas as pd
import re
from  ratetransforms import *
import matplotlib.pyplot as plt
import hjm
import tools
import seaborn as sn
#loading data:

ratescalefactor= 100.0
timewindowquotes = 250
ustreasury = pd.read_csv('data/USTREASURY-YIELD.csv',index_col=0)/ratescalefactor
ustreasury.sort_index(inplace=True)
ustreasury.columns = map(tenornormalizer,ustreasury.columns)
ustreasury =  ustreasury.iloc[-timewindowquotes::,:]
print(ustreasury.head())
tenor2years = tenors2yearsdict(list(ustreasury.columns))
print(tenor2years)

dframe = transformrates2discounts(ustreasury)
iforward = transformDiscountToInstantaneousForward(dframe)

hjmo = hjm.HJMFramework(iforward=iforward,nfactors=3)
hjmo.set_montecarlo_parameters(seed=42,timesteps=100,t_end_years=1,ntenors= 30*12)
#computing Libor 3 month distribution in t=0.5 years
liborf=[]
t=0.75
T=6.0/12.0
Nsims=5000
tools.printProgressBar(0,Nsims, prefix = 'Progress:', suffix = 'Complete', length = 50)
for n in range(Nsims):
    cube = hjmo.run_montecarlo_path()
    finterp = tools.ForwardInterpolator(mc_time=cube.index,mc_tenors=cube.columns,cube= cube.values)
    liborf.append(finterp.forward(t,T))
    tools.printProgressBar(n,Nsims, prefix = 'Progress:', suffix = 'Complete', length = 50)

sn.distplot(liborf,bins=50,kde=True,hist=True,rug=True)
plt.show()
