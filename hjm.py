import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.special as special
import copy as copylib
from scipy.interpolate import griddata
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from tools import *
from typing import Dict, Tuple, Sequence


def computeVolatilities(nfactors:int,data:pd.DataFrame,volatilityscalingfactor:float) -> np.matrix:
    # Calculation of volatility risk factors using PCA decompositions.    
    # parameters:
    #  nfactors: number of factors to be extracted. Typically from 1 to 3, mapping to level, slope and curvature.
    #  data: instanteneous forward data frame, with columns as tenors sorted in ascending order and index as date in ascending rate sorting.
    # volatilityscalingfactor: typically the SQRT(T) of the time horizon to be considered.
    # Returns: a matrix with dimension "number of tenors x nfactors"

        diff_rates = pd.DataFrame(np.diff(data, axis=0),index=data.index[1:],columns=data.columns)
        sigma = np.cov(diff_rates.transpose())*volatilityscalingfactor
        eigval, eigvector = np.linalg.eig(sigma)
        index_eigval = list(reversed(np.argsort(eigval)))[0:nfactors]
        princ_comp =eigvector[:,index_eigval]
        princ_eigval = eigval[index_eigval]
        n = princ_comp.shape[0]
        aux1 = np.sqrt(np.vstack([princ_eigval for s in range(n)]))
        vols = np.multiply(aux1,princ_comp)        
        return vols

class HJMFramework:
    # Heath Jarrow and Morton instantaneous forward simulation framework. This object covers the Monte Carlo simulation on forward rates.

    volatilityscalingfactor = 250 # hardcoded volatility scaling factor set to 1year.

    def __init__(self,iforward:pd.DateOffset,nfactors:int):
        # HJM constructor
        # iforward: pandas Dataframe with columns as tenors expressed as years (float) and index as quote dates.
        # nfactors: number of factors to be used in underlying SDE process.        
        
        self.iforward = iforward
        self.iforward.bfill(axis='rows',inplace=True) # apply backward filling first
        self.iforward.ffill(axis='rows',inplace=True) # then apply forward filling to finish the gap filling process
        self.iforward.dropna(inplace=True)            # drop any na quote to avoid numerical errors
        self.nfactors = nfactors
        self.tenors = np.array(self.iforward.columns)
        self.tenors.sort()
        self.iforward = self.iforward[self.tenors]
        self.iforward.sort_index(inplace=True)
       
    
    def __computeVolInterpolation(self):
        #internal method to compute volatility factors interpolated to the number of samples set in the set_montecarlo_parameters class method.
    
        self.vols =  computeVolatilities(nfactors=self.nfactors,\
                                        data=self.iforward,\
                                        volatilityscalingfactor=HJMFramework.volatilityscalingfactor)
       
        self.vol_interpolators = []
        for n in range(self.vols.shape[1]):
            if n==0:
                level = np.mean(np.array(self.vols[:,0]).flatten())
                aux = [level for _ in range(len(self.tenors))]
                self.vol_interpolators.append(interp1d(self.tenors,aux,kind='linear',fill_value='extrapolate'))
            else:
                self.vol_interpolators.append(interp1d(self.tenors,self.vols[:,n],'cubic',fill_value='extrapolate' ))                           
        

    def __mdrift(self,T):
        #computation of the HJM arbritage free drist condition for tenor T.

        I = 0
        for f in self.vol_interpolators:
            r,_ = integrate.quad(f,0,T) * f(T)
            I += r
        return I

    def __compute_mc_vols_and_drift(self,mc_tenor_steps):
        #internal call to volatility factor interpolators and drift functional given the number of tenors to be used in the
        #interpolation.

        self.__computeVolInterpolation()
        tenors_min = np.min(list(self.iforward.columns))
        tenors_max = np.max(list(self.iforward.columns))
        mc_tenors = np.linspace(tenors_min,tenors_max,mc_tenor_steps)
        mc_vols = np.matrix([[f(t) for t in mc_tenors] for f in self.vol_interpolators])
        mc_drift = np.array([self.__mdrift(tau) for tau in  mc_tenors])

        spot = self.__get_curve_spot()
        f_interpolator = interp1d(self.tenors ,spot,'cubic')
        mc_forward_curve = np.array([f_interpolator(tau) for tau in  mc_tenors])
        return mc_tenors,mc_vols,mc_drift,mc_forward_curve
   
    def __get_curve_spot(self):
        #internal function returning the most recent instantaneous forward curve. Typically use as the first curve in the MC algorithm,
        # that is t=0

        return self.iforward.values[-1,:]


    def __run_forward_dynamics(self,proj_time,mc_tenors,mc_vols,mc_drift,mc_forward_curve):
        # internal algorithm to evolve the undelrying discretized SDE. This function returns an iterator
        # which returns on each call the evolution of whole forward curves over time t for all T.
        # Returns a tuple (t,instantaneous forwardcurve) for each reference time t for all the tenors included in the simulation.


        len_vols = len(mc_vols)
        len_tenors = len(mc_tenors)         

        yield proj_time[0],copylib.copy(mc_forward_curve)

        for it in range(1, len(proj_time)):
            t = proj_time[it]
            dt = t - proj_time[it-1]
            sqrt_dt = np.sqrt(dt)
            fprev = mc_forward_curve
            mc_forward_curve = copylib.copy(mc_forward_curve)
            dZ = np.array([np.random.normal() for _ in range(len_vols)])
            for iT in range(len_tenors):
                a = fprev[iT] + mc_drift[iT]*dt
                sum = 0.0
                for iVol, vol in enumerate(np.array(mc_vols)):
                    sum += vol[iT] * dZ[iVol]
                b= sum*sqrt_dt

                if iT+1 < len_tenors:
                    c = (fprev[iT+1]-fprev[iT])/(mc_tenors[iT+1]-mc_tenors[iT])*dt
                else:
                    c = (fprev[iT]-fprev[iT-1])/(mc_tenors[iT]-mc_tenors[iT-1])*dt

                mc_forward_curve[iT] = a+b+c

            yield t,mc_forward_curve
    
    def set_montecarlo_parameters(self,seed,timesteps,t_end_years,ntenors):
        #Simulation parameter setup:
        # seed: random seed
        # timesteps: number of simulation steps over time t.
        # t_end_years: time horizon within
        np.random.seed(seed)
        self.mc_tenors,self.mc_vols,self.mc_drift,self.mc_forward_curve =  self.__compute_mc_vols_and_drift(mc_tenor_steps=ntenors)
        self.mc_time = np.linspace(0,t_end_years,timesteps).flatten()
        
        
    def run_montecarlo_path(self) -> pd.DataFrame:
        # generation a a complete MC simulation cube. Each call to this function generates one Monte Carlo path.
        # Returns a simulation path in the form of a pandas Dataframe with columns as tenors expressed in years and index forward starting time.
        proj_rates = []
        proj_time = self.mc_time
        mc_tenors = self.mc_tenors
        mc_drift = self.mc_drift
        mc_vols = self.mc_vols
        mc_forward_curve = self.mc_forward_curve

        for i, (t, f) in enumerate(self.__run_forward_dynamics(proj_time,mc_tenors,mc_vols,mc_drift,mc_forward_curve)):
            proj_rates.append(f)
        
        columns = [tn for tn in mc_tenors]
        proj_rates = pd.DataFrame(np.matrix(proj_rates),index=proj_time,columns=columns)
        return proj_rates

     
    def integrateforward(self,cube,t,TenorinYears):
        #given a MC simulation path it computes the forward with starting time t and tenor T
        #cube: MC simulation path (instantaneous forward cube as per returned from 'run_montecarlo_path')
        #t: forward starting time
        #T: tenors
        # Returns F(t,T)
     
        iforward = ForwardInterpolator(self.mc_time,self.mc_tenors,np.matrix(cube))
        ivals = lambda T: iforward.forward(t,T)
        integralval,_ = integrate.quadpack.quad(ivals,0,TenorinYears)
        return integralval/TenorinYears