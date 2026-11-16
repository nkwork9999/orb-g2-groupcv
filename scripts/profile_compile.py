"""Label-free comparison of fixed ORB eager and TorchInductor deployment."""
import json,time,traceback
from pathlib import Path
import numpy as np,torch
from ase import Atoms
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator

torch.set_num_threads(1)
path='/Users/nk/Desktop/diatomics-improvement-20261116/checkpoints/orb-v3-conservative-inf-mpa-ranged.ckpt'
rows=sorted([json.loads(l) for l in Path('data/train-cache.jsonl').read_text().splitlines()],key=lambda r:r['name'])
inputs=[Atoms(numbers=r['numbers'],positions=r['positions'],cell=r['cell'],pbc=r['pbc']) for r in rows]
synthetic=[Atoms('CH',positions=[[0,0,0],[1.3,0,0]],cell=[50]*3,pbc=True),Atoms('H',positions=[[0,0,0]],cell=[50]*3,pbc=False)]
report={'training_geometries_only':True,'official_scores_read':False,'atol':1e-4,'rtol':1e-5,'compile_threads':1}
try:
 start=time.perf_counter()
 eager=ORBCalculator(pretrained.orb_v3_conservative_inf_mpa(weights_path=path,device='cpu',precision='float32-highest',compile=False),device='cpu')
 compiled=ORBCalculator(pretrained.orb_v3_conservative_inf_mpa(weights_path=path,device='cpu',precision='float32-highest',compile=True),device='cpu')
 maximum=0.
 for n,a in enumerate(synthetic+inputs):
  e1=eager.get_potential_energy(a);f1=eager.get_forces(a);s1=eager.get_stress(a)
  e2=compiled.get_potential_energy(a);f2=compiled.get_forces(a);s2=compiled.get_stress(a)
  for old,new in [(np.asarray(e1),np.asarray(e2)),(f1,f2),(s1,s2)]:
   ratio=float(np.max(abs(old-new)/(1e-4+1e-5*abs(old))))
   maximum=max(maximum,ratio)
   np.testing.assert_allclose(new,old,atol=1e-4,rtol=1e-5)
  print('equivalence',n+1,'/',len(inputs)+len(synthetic),'max normalized',maximum,flush=True)
 report['cold_and_equivalence_seconds']=time.perf_counter()-start
 records=[]
 for repeat in range(3):
  for i,a in enumerate(inputs[:20]):
   order=['eager','compiled'] if (i+repeat)%2==0 else ['compiled','eager']
   record={}
   for label in order:
    c={'eager':eager,'compiled':compiled}[label]
    c.reset();t=time.perf_counter();c.get_potential_energy(a);c.get_forces(a);c.get_stress(a);record[label]=time.perf_counter()-t
   records.append(record)
 median={label:float(np.median([r[label] for r in records])) for label in ['eager','compiled']}
 report|={'status':'passed','n_equivalence_geometries':len(inputs)+len(synthetic),'maximum_normalized_difference':maximum,'median_seconds':median,'paired_timings':records,'speedup':median['eager']/median['compiled'],'adopt_compilation':median['compiled']<=.8*median['eager']}
except Exception as error:
 report|={'status':'failed','error':repr(error),'traceback':traceback.format_exc(),'adopt_compilation':False}
 print(report['traceback'],flush=True)
Path('results/compile-profile.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
