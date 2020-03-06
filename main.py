import numpy as np
import pandas as pd
import re
import  ratetransforms
import matplotlib.pyplot as plt
import hjm
import tools
import seaborn as sn
from scipy.stats import norm
from scipy.interpolate import interp1d
from functools import reduce

#loading data:

ratescalefactor= 100.0
timewindowquotes = 250
ustreasury = pd.read_csv('data/USTREASURY-YIELD.csv',index_col=0)/ratescalefactor
ustreasury.sort_index(inplace=True)
ustreasury.columns = map(ratetransforms.tenornormalizer,ustreasury.columns)
ustreasury =  ustreasury.iloc[-timewindowquotes::,:]
print(ustreasury.head())
tenor2years = ratetransforms.tenors2yearsdict(list(ustreasury.columns))
print(tenor2years)

dframe = ratetransforms.transformrates2discounts(ustreasury)
iforward = ratetransforms.transformDiscountToInstantaneousForward(dframe)

hjmo = hjm.HJMFramework(iforward=iforward,nfactors=3)
hjmo.set_montecarlo_parameters(seed=42,timesteps=4*12,t_end_years=1,ntenors= 30*12)
#computing Libor 3 month distribution in t=0.5 years
liborf=[]
t=[0.25,0.5,0.75,1.0]
T=[0.25,0.5,1.0]
Nsims=1000
rates={}
tools.printProgressBar(0,Nsims*len(T), prefix = 'Progress:', suffix = 'Complete', length = 50)
counter = 0
for k in T:
    rates[k]=[]
    for n in range(Nsims):
        cube = hjmo.run_montecarlo_path()
        xxx=[hjmo.integrateforward(cube=cube,t=s,TenorinYears=k) for s in t]
        rates[k].append(xxx)
        tools.printProgressBar(counter,Nsims*len(T), prefix = 'Progress:', suffix = 'Complete', length = 50)
        counter += 1

columns = np.concatenate([['T'],list(map(lambda s: str(s),t))])
datam = []
for key in rates.keys():
    datam.append(np.hstack([np.ones((len(rates[key]),1))*key,np.matrix(rates[key])]))
datam = reduce(lambda x,y: np.vstack((x,y)),datam)
rframe = pd.DataFrame(data=datam,columns=columns)
for c in T:
    ff = rframe[np.abs(rframe['T']-c)<1e-9]
    for s in rframe.columns[1:]:        
        sn.distplot(ff[s],bins=50,kde=True,hist=False,rug=False,label=f'T={c}, t={c}')

print(rframe.groupby('T').mean())
# print(rframe.mean(axis=0))
# fa = []
# tenors = [ratetransforms.tenor2years(x)[1] for x in ustreasury.columns]
# finterpan = interp1d(tenors,ustreasury.iloc[-1,:].values,kind='cubic',fill_value='extrapolate')
# for s in t:
#     fa.append(ratetransforms.forward(finterpan,s,T))
# print(fa)


plt.legend()
#plt.figure()
# plt.plot(t,rframe.mean(axis=0).values,label='forward analytical')
# plt.plot(t,fa,label='forward mc')
# plt.legend()
plt.show()
