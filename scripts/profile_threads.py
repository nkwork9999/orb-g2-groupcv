"""Prospective, label-free CPU thread/backend comparison on training geometries."""
import json,time
from pathlib import Path
import numpy as np,torch
from ase import Atoms
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator

path='/Users/nk/Desktop/diatomics-improvement-20261116/checkpoints/orb-v3-conservative-inf-mpa-ranged.ckpt'
rows=sorted([json.loads(l) for l in Path('data/train-cache.jsonl').read_text().splitlines()],key=lambda r:r['name'])
inputs=[Atoms(numbers=r['numbers'],positions=r['positions'],cell=r['cell'],pbc=r['pbc']) for r in rows]
torch.set_num_threads(1)
calcs={'eager':ORBCalculator(pretrained.orb_v3_conservative_inf_mpa(weights_path=path,device='cpu',precision='float32-highest',compile=False),device='cpu'),'compiled':ORBCalculator(pretrained.orb_v3_conservative_inf_mpa(weights_path=path,device='cpu',precision='float32-highest',compile=True),device='cpu')}
refs=[]
for a in inputs:
 c=calcs['eager'];refs.append((float(c.get_potential_energy(a)),c.get_forces(a).copy(),c.get_stress(a).copy()))
max_diff={};timings=[]
for threads in (1,2,4):
 torch.set_num_threads(threads)
 for backend,c in calcs.items():
  maximum=0.
  for a,reference in zip(inputs,refs,strict=True):
   c.reset();new=(c.get_potential_energy(a),c.get_forces(a),c.get_stress(a))
   for old,value in zip(reference,new,strict=True):
    old=np.asarray(old);value=np.asarray(value)
    maximum=max(maximum,float(np.max(abs(old-value)/(1e-4+1e-5*abs(old)))))
    np.testing.assert_allclose(value,old,atol=1e-4,rtol=1e-5)
  max_diff[f'{backend}-{threads}']=maximum
  print('equivalence passed',backend,threads,'normalized max',maximum,flush=True)
# Randomized paired deployment order; reset forces fresh model work.
rng=np.random.default_rng(20261116)
for repeat in range(5):
 for index,a in enumerate(inputs[:20]):
  order=[(b,t) for t in (1,2,4) for b in ('eager','compiled')];rng.shuffle(order)
  record={'repeat':repeat,'training_name':rows[index]['name'],'seconds':{}}
  for backend,threads in order:
   torch.set_num_threads(threads);c=calcs[backend];c.reset();start=time.perf_counter()
   c.get_potential_energy(a);c.get_forces(a);c.get_stress(a)
   record['seconds'][f'{backend}-{threads}']=time.perf_counter()-start
  timings.append(record)
med={key:float(np.median([r['seconds'][key] for r in timings])) for key in max_diff}
eligible=[]
for backend in ('eager','compiled'):
 for threads in (1,2,4):
  key=f'{backend}-{threads}'
  allowed=backend=='eager' and (threads==1 or med[key]<=.8*med['eager-1'])
  if backend=='compiled':allowed=med[key]<=.8*med[f'eager-{threads}']
  if allowed:eligible.append({'backend':backend,'threads':threads,'median_seconds':med[key]})
minimum=min(r['median_seconds'] for r in eligible)
best=min([r for r in eligible if r['median_seconds']<=1.05*minimum],key=lambda r:(r['threads'],r['median_seconds']))
report={'median_seconds':med,'max_normalized_difference':max_diff,'eligible':eligible,'selected':best,'official_scores_read':False,'training_geometries_only':True,'paired_timings':timings}
Path('results/thread-profile.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='paired_timings'},indent=2),flush=True)
