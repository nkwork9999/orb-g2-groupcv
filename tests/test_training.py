import copy
import numpy as np
import pytest
from orb_g2_cv.training import ridge_fit, grouped_predictions, select_alpha, train


def test_fold_prediction_is_invariant_to_its_own_target_changes():
    # Each held-out group's own labels may change its score, never its predictions.
    x=np.array([[1.,1.],[1.,2.],[1.,3.],[1.,4.],[1.,5.],[1.,6.]])
    y=np.array([2.,4.,6.,8.,10.,12.]);groups=['A','A','B','B','C','C']
    before=grouped_predictions(x,y,groups,ridge=.01)
    after_y=y.copy();after_y[:2]=1e8
    after=grouped_predictions(x,after_y,groups,ridge=.01)
    np.testing.assert_array_equal(before[:2],after[:2])
    assert np.max(abs(before[2:]-after[2:]))>1


def test_ridge_learns_real_linear_relation_with_group_weighting():
    x=np.array([[1.,0.],[1.,1.],[1.,2.],[1.,3.]])
    coeff=ridge_fit(x,np.array([1.,3.,5.,7.]),['A','B','C','D'],ridge=1e-10)
    np.testing.assert_allclose(x@coeff,[1.,3.,5.,7.],atol=1e-8)


def test_alpha_gate_keeps_zero_for_harmful_or_too_small_gains():
    assert select_alpha([1.,-1.],[1.,-1.],['A','B'])['alpha']==1
    assert select_alpha([1.,-1.],[-1.,1.],['A','B'])['alpha']==0
    assert select_alpha([1.,-1.],[.001,-.001],['A','B'])['alpha']==0


def test_actual_training_rejects_outer_targets_and_duplicate_identity():
    base={'name':'a','formula':'H2O','numbers':[8,1,1],'positions':[[0,0,0],[1,0,0],[0,1,0]],'cell':np.eye(3).tolist(),'pbc':[False]*3,'role':'train','baseline_atomization_ev':2.,'reference_atomization_ev':3.}
    rows=[base|{'name':str(i),'formula':f'group{i}'} for i in range(5)]
    with pytest.raises(ValueError):train(rows+[base|{'role':'holdout'}],n_trials=2,n_initial=2)
    with pytest.raises(ValueError):train(rows+[rows[0]],n_trials=2,n_initial=2)


def test_training_is_deterministic_and_reports_training_only_sample_ids():
    base={'formula':'H2O','numbers':[8,1,1],'positions':[[0,0,0],[1,0,0],[0,1,0]],'cell':(np.eye(3)*50).tolist(),'pbc':[False]*3,'role':'train','baseline_atomization_ev':2.,'reference_atomization_ev':3.}
    rows=[base|{'name':str(i),'formula':f'group{i}'} for i in range(5)]
    a=train(rows,n_trials=2,n_initial=2);b=train(copy.deepcopy(rows),n_trials=2,n_initial=2)
    np.testing.assert_array_equal(a['coefficients'],b['coefficients'])
    assert a['training_ids']==['0','1','2','3','4']
    assert a['holdout_used_for_selection'] is False
