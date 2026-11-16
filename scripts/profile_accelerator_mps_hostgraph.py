"""Label-free, synchronized backend/precision profiling of frozen ORB."""
import json,time,cProfile,pstats
from pathlib import Path
import numpy as np,torch
from ase import Atoms
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator
checkpoint='/Users/nk/Desktop/diatomics-improvement-20261116/checkpoints/orb-v3-conservative-inf-mpa-ranged.ckpt'
rows=sorted([json.loads(l) for l in Path('data/train-cache.jsonl').read_text().splitlines()],key=lambda r:r['name'])
inputs=[Atoms(numbers=r['numbers'],positions=r['positions'],cell=r['cell'],pbc=r['pbc']) for r in rows]
inputs += [Atoms('H',positions=[[0,0,0]],cell=[50]*3,pbc=False)]+[Atoms('CH',positions=[[0,0,0],[r,0,0]],cell=[50]*3,pbc=True) for r in (.1,1.3,10.)]
torch.set_num_threads(4)
from ase.calculators.calculator import Calculator,all_changes
from orb_models.forcefield.atomic_system import ase_atoms_to_atom_graphs
from orb_models.utils import to_numpy
class HostGraphCalculator(Calculator):
 implemented_properties=['energy','forces','stress']
 def __init__(self,model):super().__init__();self.model=model
 def calculate(self,atoms=None,properties=None,system_changes=all_changes):
  super().calculate(atoms,properties,system_changes)
  batch=ase_atoms_to_atom_graphs(self.atoms,system_config=self.model.system_config,device='cpu',output_dtype=torch.float32,graph_construction_dtype=torch.float32).to(device='mps',dtype=torch.float32)
  out=self.model.predict(batch)
  self.results={'energy':float(to_numpy(out[self.model.energy_name]).reshape(-1)[0]),'forces':to_numpy(out[self.model.grad_forces_name]),'stress':to_numpy(out[self.model.grad_stress_name]).reshape(6)}
configs=[('cpu-highest','cpu','float32-highest'),('cpu-high','cpu','float32-high'),('mps-highest','mps','float32-highest')]
calcs={}; errors={};precision={}
for key,device,p in configs:
 try:
  if device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable')
  model=pretrained.orb_v3_conservative_inf_mpa(weights_path=checkpoint,device='cpu',precision=p,compile=False)
  model=model.to(device)
  assert all(t.device.type==device for t in model.parameters())
  calcs[key]=HostGraphCalculator(model) if device=='mps' else ORBCalculator(model,device=device);precision[key]=p
 except Exception as e:errors[key]=repr(e)
def predict(key,atoms):
 torch.set_float32_matmul_precision('high' if precision[key]=='float32-high' else 'highest')
 calc=calcs[key];calc.reset()
 return [np.asarray(calc.get_potential_energy(atoms)),calc.get_forces(atoms),calc.get_stress(atoms)]
maxerr={k:0. for k in calcs}
for a in inputs:
 ref=predict('cpu-highest',a)
 for key in list(calcs):
  if key=='cpu-highest':continue
  try:
   out=predict(key,a)
   error=max(float(np.max(np.abs(x-y)/(1e-4+1e-5*np.abs(y)))) for x,y in zip(out,ref))
   maxerr[key]=max(maxerr[key],error)
   if error>1:raise ValueError(f'equivalence normalized error {error}')
  except Exception as e:errors[key]=repr(e);del calcs[key]
print('equivalence',maxerr,'errors',errors,flush=True)
# Profile the scalar reference on training geometries to identify real bottlenecks.
p=cProfile.Profile();p.enable()
for a in inputs[:20]:predict('cpu-highest',a)
p.disable();pstats.Stats(p).sort_stats('cumtime').dump_stats('results/cpu-reference-hostgraph-profile.pstats')
with open('results/cpu-reference-hostgraph-profile.txt','w') as f:pstats.Stats(p,stream=f).sort_stats('cumtime').print_stats(35)
for key in calcs:
 for a in inputs[:20]:predict(key,a)
records=[]
keys=list(calcs)
for repeat in range(5):
 for i,a in enumerate(inputs[:20]):
  for key in (keys if (repeat+i)%2 else keys[::-1]):
   if key.startswith('mps'):torch.mps.synchronize()
   start=time.perf_counter();predict(key,a)
   if key.startswith('mps'):torch.mps.synchronize()
   records.append({'key':key,'repeat':repeat,'name':rows[i]['name'],'seconds':time.perf_counter()-start})
med={key:float(np.median([r['seconds'] for r in records if r['key']==key])) for key in keys}
eligible=[key for key in keys if key=='cpu-highest' or med[key]<=.8*med['cpu-highest']]
result={'median_seconds':med,'selected':min(eligible,key=lambda k:med[k]),'errors':errors,'maximum_normalized_errors':maxerr,'training_geometries_only':True,'timings':records}
Path('results/accelerator-hostgraph-profile.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='timings'},indent=2),flush=True)
