"""Prespecified training-only, synchronized numerical and latency gates."""
import json,time
from pathlib import Path
import numpy as np,torch,mlx.core as mx
from ase import Atoms
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator
from orb_g2_cv.mlx_calculator import MLXCalculator
from orb_g2_cv.ase import CalibratedCalculator
from orb_g2_cv.potential import RadialHead

torch.set_num_threads(4)
checkpoint='/Users/nk/Desktop/diatomics-improvement-20261116/checkpoints/orb-v3-conservative-inf-mpa-ranged.ckpt'
start=time.perf_counter()
model=pretrained.orb_v3_conservative_inf_mpa(weights_path=checkpoint,device='cpu',precision='float32-highest',compile=False)
artifact=json.loads(Path('artifacts/orb-g2-groupcv-v1.json').read_text())
def wrap(base):return CalibratedCalculator(base,RadialHead(artifact['head']['coefficients'],width=artifact['head']['width']))
calcs={'cpu':wrap(ORBCalculator(model,device='cpu')),'mlx':wrap(MLXCalculator(model))}
load=time.perf_counter()-start
rows=sorted([json.loads(l) for l in Path('data/train-cache.jsonl').read_text().splitlines()],key=lambda r:r['name'])
inputs=[Atoms(numbers=r['numbers'],positions=r['positions'],cell=r['cell'],pbc=r['pbc']) for r in rows]
synthetic=[Atoms('H',positions=[[0,0,0]],cell=[50]*3,pbc=False),Atoms('Si',positions=[[0,0,0]],cell=[3]*3,pbc=True),Atoms('Fe2',positions=[[0,0,0],[2,1,0]],cell=[[5,0,0],[1,5,0],[.5,1,5]],pbc=True)]
synthetic += [Atoms('CH',positions=[[0,0,0],[r,0,0]],cell=[50]*3,pbc=True) for r in (.1,1.3,5.99,6.01,10.)]
# Atomic-number coverage without reference labels or official geometries.
synthetic += [Atoms(numbers=[z,6],positions=[[0,0,0],[1.7,0,0]],cell=[50]*3,pbc=False) for z in range(1,93)]
def predict(key,atoms):
 calc=calcs[key];calc.reset()
 return [np.asarray(calc.get_potential_energy(atoms)),calc.get_forces(atoms),calc.get_stress(atoms)]

import cProfile,pstats
records=[]
for threads in (1,2,4):
 torch.set_num_threads(threads)
 for a in inputs[:20]:predict('mlx',a)
 profile=cProfile.Profile();profile.enable()
 for a in inputs[:20]:predict('mlx',a)
 profile.disable()
 with open(f'results/mlx-thread{threads}-profile.txt','w') as file:pstats.Stats(profile,stream=file).sort_stats('cumtime').print_stats(30)
 for repeat in range(5):
  for i,a in enumerate(inputs[:20]):
   mx.synchronize();start=time.perf_counter();predict('mlx',a);mx.synchronize()
   records.append({'threads':threads,'repeat':repeat,'name':rows[i]['name'],'seconds':time.perf_counter()-start})
 print('completed thread block',threads,flush=True)
med={t:float(np.median([r['seconds'] for r in records if r['threads']==t])) for t in (1,2,4)}
eligible=[t for t in (1,2) if med[t]<=.8*med[4]]+[4]
minimum=min(med[t] for t in eligible);selected=min(t for t in eligible if med[t]<=1.05*minimum)
# Numerical checks occur outside the fixed timing blocks.
torch.set_num_threads(4)
references=[predict('cpu',a) for a in inputs+synthetic]
torch.set_num_threads(selected)
errors=[]
for a,ref in zip(inputs+synthetic,references):
 out=predict('mlx',a)
 error=max(float(np.max(np.abs(x-y)/(1e-4+1e-5*np.abs(y)))) for x,y in zip(out,ref))
 errors.append(error)
 if error>1:raise ValueError(('final thread equivalence',error))
result={'median_seconds':med,'selected_threads':selected,'maximum_normalized_error':max(errors),'n_equivalence_cases':len(errors),'training_geometries_only':True,'all_learned_weights_unchanged':True,'paired_timings':records}
Path('results/mlx-thread-profile.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='paired_timings'},indent=2),flush=True)
