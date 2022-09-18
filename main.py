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


ratescalefactor= 100.0 # scale factor to convert % quantities into decimals.
timewindowquotes = 250 # time window to be used to calibrate interest rate volatility. 

#loading data. We use for this example US Treasury yield curves. Our final dataframe contains as index the quote dates, 
# columns the tenors in string format such as 3m, 1y etc ...
ustreasury = pd.read_csv('data/USTREASURY-YIELD.csv',index_col=0)/ratescalefactor
ustreasury.sort_index(inplace=True)
ustreasury.columns = map(ratetransforms.tenornormalizer,ustreasury.columns)
ustreasury =  ustreasury.iloc[-timewindowquotes::,:]
print(ustreasury.head())
tenor2years = ratetransforms.tenors2yearsdict(list(ustreasury.columns))
print(tenor2years)

#given a data frame ustreasury, we apply the transformation to discounts ...
dframe = ratetransforms.transformrates2discounts(ustreasury)

# ... and from discount to instantaneous forwards
iforward = ratetransforms.transformDiscountToInstantaneousForward(dframe)

# using the calculated instantaneous forwards we  instantiate a HJM object to simulate forward interest reates.
# 3 factors: level, slope and curvature are selected to simulate the underlying SDE for instantaneous forwards.
hjmo = hjm.HJMFramework(iforward=iforward,nfactors=3)
# the montecarlo parameters are set to discretize time over one 1year with weekly samples and the given tenor including in the original 
# yield file is discretized with monthly tenor sampes. The us treasury filel includes tenors from  1 month up to 30 years, and we sample in 
#between this range 30*12 tenors (monthly)
hjmo.set_montecarlo_parameters(seed=42,timesteps=12,t_end_years=1,ntenors= 100)

#with the hjm object instantiated we can start computing forward rate distributions:
liborf=[]
#compute forwards with starting time at t and tenors at T, namely F(to,t,T), where t0 is implicitely the most recent date in the us treasury input file,
# t is the forward starting time and T is the tenor.
t= np.array([1.0/250.0,1/12.0,0.5,1.0])
T_months= np.array([1.0,2.0,3.0])/12.0
T_year = np.array([1.0,2.0,3.0,5.0,7.0,10.0,20.0,30.0])
T = np.concatenate([T_months,T_year])
#1 MO	2 MO	3 MO	6 MO	1 YR	2 YR	3 YR	5 YR	7 YR	10 YR	20 YR	30 YR
#our Monte Carlo simulation will include 1000 paths
Nsims=100
rates={}

# here we run the Monte Carlo, computing all the forwards F(t0,t,T) specificed with t and T vectors for each path.
tools.printProgressBar(0,Nsims*len(T), prefix = 'Progress:', suffix = 'Complete', length = 50)
counter = 0
sim_idx =[]
for k in T:
    rates[k]=[]
    for n in range(Nsims):
        cube = hjmo.run_montecarlo_path()
        if k==0:
            print('hi')
        xxx=[hjmo.integrateforward(cube=cube,t=s,TenorinYears=k) for s in t]
        rates[k].append(xxx)
        sim_idx.append(n)
        tools.printProgressBar(counter,Nsims*len(T), prefix = 'Progress:', suffix = 'Complete', length = 50)
        counter += 1

columns = np.concatenate([['T'],list(map(lambda s: str(s),t))])
datam = []
for key in rates.keys():
    datam.append(np.hstack([np.ones((len(rates[key]),1))*key,np.matrix(rates[key])]))
datam = reduce(lambda x,y: np.vstack((x,y)),datam)
rframe = pd.DataFrame(data=datam,columns=columns,index=sim_idx)
for c in T:
    ff = rframe[np.abs(rframe['T']-c)<1e-9]
    for s in rframe.columns[1:]:                
        sn.distplot(ff[s],bins=50,kde=True,hist=True,rug=False,label=f'T={c}, t={s}') 
    plt.legend()   
    plt.figure()

#printing the resulting MC simulations with columns T= tenor, [time ....] and index the simulation index
rframe.to_csv('./mcforward.csv',sep=',')
print(rframe.groupby('T').mean())


plt.show()
