"""Training-only group cross-validation; held-out rows are rejected at the API."""
import copy
import hashlib
import numpy as np
from ase import Atoms
from .potential import energy_features
from .search import search_loop

ALPHAS=(0.,.01,.03,.1,.3,1.)


def ridge_fit(x,y,groups,ridge):
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float);groups=np.asarray(groups)
    if x.ndim!=2 or y.shape!=(len(x),) or len(groups)!=len(x) or not len(x) or not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(ridge) or ridge<=0:
        raise ValueError('finite nonempty training arrays and positive ridge required')
    scale=np.maximum(np.sqrt(np.mean(x*x,axis=0)),1e-10)
    weights=np.array([1/np.sum(groups==g) for g in groups])
    design=x/scale*np.sqrt(weights)[:,None];target=y*np.sqrt(weights)
    return np.linalg.solve(design.T@design+ridge*np.eye(x.shape[1]),design.T@target)/scale


def fold_assignments(groups):
    unique=sorted(set(groups),key=lambda g:hashlib.sha256(f'42:{g}'.encode()).hexdigest())
    if len(unique)<2:raise ValueError('at least two training formula groups required')
    k=min(5,len(unique))
    mapping={g:i%k for i,g in enumerate(unique)}
    return np.array([mapping[g] for g in groups])


def grouped_predictions(x,y,groups,ridge):
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float);groups=np.asarray(groups)
    folds=fold_assignments(groups);pred=np.empty(len(y))
    for fold in sorted(set(folds)):
        val=folds==fold;fit=~val
        coefficient=ridge_fit(x[fit],y[fit],groups[fit],ridge)
        pred[val]=x[val]@coefficient
    return pred


def macro_mae(values,groups):
    values=np.asarray(values,dtype=float);groups=np.asarray(groups)
    if values.ndim!=1 or not len(values) or len(values)!=len(groups) or not np.isfinite(values).all():
        raise ValueError('finite grouped residual vector required')
    return float(np.mean([np.mean(abs(values[groups==g])) for g in sorted(set(groups))]))


def select_alpha(y,pred,groups):
    y=np.asarray(y,dtype=float);pred=np.asarray(pred,dtype=float)
    if y.shape!=pred.shape:raise ValueError('mismatched residual and correction')
    baseline=macro_mae(y,groups);results=[]
    for alpha in ALPHAS:
        mae=macro_mae(y-alpha*pred,groups)
        results.append({'alpha':alpha,'mae':mae,'eligible':alpha==0 or mae<=.98*baseline})
    best=min((r for r in results if r['eligible']),key=lambda r:(r['mae'],r['alpha']))
    return best|{'baseline_mae':baseline,'candidates':results}


def train(rows,n_trials=16,n_initial=8,method='bo',seed=42):
    rows=copy.deepcopy(list(rows))
    if not rows or any(row.get('role')!='train' for row in rows):
        raise ValueError('fit/search accept training rows only; outer evaluation rows are rejected')
    if len({r['name'] for r in rows})!=len(rows):raise ValueError('duplicate sample identity')
    groups=[r['formula'] for r in rows]
    folds=fold_assignments(groups)
    atoms=[Atoms(numbers=r['numbers'],positions=r['positions'],cell=r['cell'],pbc=r['pbc']) for r in rows]
    y=np.array([r['baseline_atomization_ev']-r['reference_atomization_ev'] for r in rows])
    trials=[]
    def objective(unit):
        ridge=10**(-4+8*unit[0]);width=.25+1.25*unit[1]
        x=np.array([energy_features(a,width) for a in atoms])
        pred=grouped_predictions(x,y,groups,ridge)
        gate=select_alpha(y,pred,groups)
        trials.append({'ridge':float(ridge),'width':float(width),'gate':gate,'oof_correction':pred.tolist()})
        return gate['mae']
    search=search_loop(objective,method=method,dimension=2,n_initial=n_initial,n_trials=n_trials,seed=seed)
    best=trials[search['best_index']];width=best['width'];alpha=best['gate']['alpha']
    x=np.array([energy_features(a,width) for a in atoms])
    coefficients=ridge_fit(x,y,groups,best['ridge'])*alpha
    return {'coefficients':coefficients.tolist(),'width':width,'ridge':best['ridge'],'alpha':alpha,
            'training_ids':[r['name'] for r in rows],'training_formula_groups':sorted(set(groups)),
            'inner_fold_by_name':{r['name']:int(folds[i]) for i,r in enumerate(rows)},
            'inner_cv':best['gate'],'search':search,'trial_details':trials,
            'holdout_used_for_selection':False,'official_test_labels_used_for_fit_or_search':False,
            'targets':'experimental atomization-energy residual; no force or stress labels fitted'}
