"""Fixed-deployment validation; thread pools are never switched inside timing blocks."""
import json,time
from pathlib import Path
import numpy as np,torch
from ase import Atoms
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator

checkpoint='/Users/nk/Desktop/diatomics-improvement-20261116/checkpoints/orb-v3-conservative-inf-mpa-ranged.ckpt'
rows=sorted([json.loads(l) for l in Path('data/train-cache.jsonl').read_text().splitlines()],key=lambda r:r['name'])[:20]
inputs=[Atoms(numbers=r['numbers'],positions=r['positions'],cell=r['cell'],pbc=r['pbc']) for r in rows]
torch.set_num_threads(1)
calcs={b:ORBCalculator(pretrained.orb_v3_conservative_inf_mpa(weights_path=checkpoint,device='cpu',precision='float32-highest',compile=b=='compiled'),device='cpu') for b in ['eager','compiled']}
records=[]
for count in (1,2,4):
 torch.set_num_threads(count)
 for a in inputs:
  for calc in calcs.values():calc.reset();calc.get_potential_energy(a);calc.get_forces(a);calc.get_stress(a)
 for repeat in range(5):
  for i,a in enumerate(inputs):
   record={'threads':count,'repeat':repeat,'name':rows[i]['name'],'seconds':{}}
   for backend in (['eager','compiled'] if (repeat+i)%2 else ['compiled','eager']):
    calc=calcs[backend];calc.reset();start=time.perf_counter();calc.get_potential_energy(a);calc.get_forces(a);calc.get_stress(a)
    record['seconds'][backend]=time.perf_counter()-start
   records.append(record)
 print('fixed timing block completed',count,flush=True)
med={f'{b}-{t}':float(np.median([r['seconds'][b] for r in records if r['threads']==t])) for t in (1,2,4) for b in ('eager','compiled')}
eligible=[{'backend':'eager','threads':1,'median_seconds':med['eager-1']}]
for threads in (2,4):
 if med[f'eager-{threads}']<=.8*med['eager-1']:eligible.append({'backend':'eager','threads':threads,'median_seconds':med[f'eager-{threads}']})
 if med[f'compiled-{threads}']<=.8*med[f'eager-{threads}'] and med[f'compiled-{threads}']<=.8*med['eager-1']:eligible.append({'backend':'compiled','threads':threads,'median_seconds':med[f'compiled-{threads}']})
minimum=min(r['median_seconds'] for r in eligible)
selected=min([r for r in eligible if r['median_seconds']<=1.05*minimum],key=lambda r:(r['threads'],r['median_seconds']))
result={'median_seconds':med,'eligible':eligible,'selected':selected,'compiled1_excluded_by_original_failed_gate':True,'official_scores_read':False,'training_geometries_only':True,'paired_timings':records}
Path('results/fixed-thread-profile.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='paired_timings'},indent=2),flush=True)
