"""Seeded fixed-budget GP-EI search reused from nkwork9999/pet-xs-pair-bo v0.1.0 (BSD-3-Clause)."""
import math
import time
import warnings
import numpy as np
from scipy.stats import norm,qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel,Matern

def expected_improvement(mean,std,best):
    """Expected improvement for minimization, including deterministic limits."""
    mean,std=np.broadcast_arrays(np.asarray(mean,dtype=float),np.asarray(std,dtype=float))
    improvement=best-mean
    safe=np.maximum(std,1e-15)
    z=improvement/safe
    ei=improvement*norm.cdf(z)+std*norm.pdf(z)
    return np.where(std>1e-15,np.maximum(ei,0.),np.maximum(improvement,0.))

def search_loop(objective,method='bo',dimension=4,n_initial=8,n_trials=16,seed=42):
    """Equal evaluation budgets; both arms share the same initial Sobol points.

    EI is maximized over a seeded, fixed Sobol candidate pool. Kernel fitting
    warnings are retained as diagnostics, not silently discarded.
    """
    if method not in ('bo','sobol') or not 2<=n_initial<=n_trials or dimension<1:
        raise ValueError('invalid method, dimension, or trial budget')
    start=time.perf_counter()
    sobol=qmc.Sobol(d=dimension,scramble=True,seed=seed).random_base2(math.ceil(math.log2(n_trials)))
    pool=qmc.Sobol(d=dimension,scramble=True,seed=seed+10000).random_base2(10)
    trials=[];gp_seconds=0.;fit_seconds=0.;diagnostics=[]
    for index in range(n_trials):
        if index<n_initial or method=='sobol':
            point=sobol[index]
        else:
            gp_start=time.perf_counter()
            x=np.array([t['x'] for t in trials]); y=np.array([t['value'] for t in trials])
            kernel=ConstantKernel(1.,(1e-2,1e2))*Matern(
                length_scale=np.full(dimension,.4),length_scale_bounds=(.03,3.),nu=2.5)
            gp=GaussianProcessRegressor(kernel=kernel,alpha=1e-6,normalize_y=True,
                                        random_state=seed,n_restarts_optimizer=0)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                gp.fit(x,y)
            diagnostics.extend({'trial':index,'category':w.category.__name__,
                                'message':str(w.message)} for w in caught)
            candidates=pool[np.min(np.linalg.norm(pool[:,None,:]-x[None,:,:],axis=2),axis=1)>1e-9]
            mean,std=gp.predict(candidates,return_std=True)
            point=candidates[int(np.argmax(expected_improvement(mean,std,float(y.min()))))]
            gp_seconds+=time.perf_counter()-gp_start
        fit_start=time.perf_counter()
        value=float(objective(point.copy()))
        seconds=time.perf_counter()-fit_start
        if not np.isfinite(value):
            raise ValueError('nonfinite validation objective')
        trials.append({'index':index,'x':point.tolist(),'value':value,'fit_seconds':seconds,
                       'proposal':'sobol' if index<n_initial or method=='sobol' else 'gp_ei'})
        fit_seconds+=seconds
    best=min(trials,key=lambda t:t['value'])
    return {'method':method,'seed':seed,'n_initial':n_initial,'n_trials':n_trials,
            'best_x':best['x'],'best_value':best['value'],'best_index':best['index'],
            'trials':trials,'gp_seconds':gp_seconds,'fit_seconds':fit_seconds,
            'wall_seconds':time.perf_counter()-start,'gp_warnings':diagnostics}
