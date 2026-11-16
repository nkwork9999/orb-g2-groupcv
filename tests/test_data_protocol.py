import copy
import json
import pytest
from orb_g2_cv.data import assign_roles, freeze_manifest


def test_split_keeps_formula_groups_together_and_ignores_targets():
    rows=[{'name':f'n{i}','formula':f'g{i//2}','target':i} for i in range(12)]
    a=assign_roles(rows,seed=17)
    altered=copy.deepcopy(rows)
    for row in altered:row['target']=-1e9*row['target']
    b=assign_roles(list(reversed(altered)),seed=17)
    assert {r['name']:r['role'] for r in a}=={r['name']:r['role'] for r in b}
    roles={}
    for row in a:roles.setdefault(row['formula'],set()).add(row['role'])
    assert all(len(r)==1 for r in roles.values())
    assert sum('holdout' in r for r in roles.values())==2
    assert sum('train' in r for r in roles.values())==4


def test_freeze_refuses_resplit_and_preserves_original_bytes(tmp_path):
    path=tmp_path/'split.json';rows=[{'name':str(i),'formula':str(i)} for i in range(8)]
    freeze_manifest(path,rows,{'seed':17})
    before=path.read_bytes()
    with pytest.raises(FileExistsError):freeze_manifest(path,rows,{'seed':18})
    assert path.read_bytes()==before
    assert len(json.loads(before)['records'])==8


def test_duplicate_sample_identity_is_rejected():
    with pytest.raises(ValueError):assign_roles([{'name':'same','formula':'A'},{'name':'same','formula':'B'}],seed=17)


def test_extended_g2_target_uses_merged_source_not_g1_only_lookup():
    from orb_g2_cv.data import atomization_energy_ev
    # Independently calculated from the published G2 heat/ZPE/thermal formula.
    from ase.units import kcal, mol
    assert atomization_energy_ev('2-butyne') / (kcal/mol) == pytest.approx(1003.9475)
    assert atomization_energy_ev('H2') / (kcal/mol) == pytest.approx(109.6047)
