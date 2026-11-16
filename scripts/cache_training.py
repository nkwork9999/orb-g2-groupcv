"""Cache only training labels/ORB outputs after the immutable input split."""
import hashlib,json,time
from pathlib import Path
import numpy as np,torch
from ase import Atoms
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator

root=Path(__file__).resolve().parents[1]
rows=[json.loads(line) for line in (root/'data/train-reference.jsonl').read_text().splitlines()]
assert rows and all(r['role']=='train' for r in rows)
checkpoint=Path('/Users/nk/Desktop/diatomics-improvement-20261116/checkpoints/orb-v3-conservative-inf-mpa-ranged.ckpt')
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()=='bd1840d1488f3cd01d213970d9618a63f971590ad45e5e83bf5404d4b95d216a'
torch.set_num_threads(1);start=time.perf_counter()
model=pretrained.orb_v3_conservative_inf_mpa(weights_path=str(checkpoint),device='cpu',precision='float32-highest',compile=False)
calc=ORBCalculator(model,device='cpu')
atomic={}
for z in sorted({z for row in rows for z in row['numbers']}):
 a=Atoms(numbers=[z],positions=[[0,0,0]],cell=[50]*3,pbc=False);a.calc=calc
 atomic[z]=float(a.get_potential_energy())
result=[]
for row in rows:
 a=Atoms(numbers=row['numbers'],positions=row['positions'],cell=row['cell'],pbc=row['pbc']);a.calc=calc
 energy=float(a.get_potential_energy());forces=np.asarray(a.get_forces(),dtype=float)
 assert np.isfinite(energy) and np.isfinite(forces).all()
 result.append(row|{'baseline_energy_ev':energy,'baseline_atomization_ev':sum(atomic[z] for z in row['numbers'])-energy,'baseline_forces_ev_angstrom':forces.tolist()})
with (root/'data/train-cache.jsonl').open('x') as file:
 for row in result:file.write(json.dumps(row,allow_nan=False)+'\n')
report={'n_training_rows':len(result),'seconds':time.perf_counter()-start,'base_atomic_energies_ev':atomic,'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'train_cache_sha256':hashlib.sha256((root/'data/train-cache.jsonl').read_bytes()).hexdigest(),'holdout_rows_read':0,'holdout_labels_read':False}
(root/'evidence/train-cache.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
