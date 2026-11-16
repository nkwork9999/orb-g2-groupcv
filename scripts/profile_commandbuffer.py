"""Prospective fresh-process Metal scheduling profile on training geometries."""
import json,os,sys,time,subprocess
from pathlib import Path

OUT=Path('results/commandbuffer')
OUT.mkdir(exist_ok=True)
if '--child' not in sys.argv:
    records=[]
    candidates=[40,128,256]
    for block in range(5):
        order=candidates[block%3:]+candidates[:block%3]
        if block%2:order=order[::-1]
        for ops in order:
            env=os.environ.copy();env['MLX_MAX_OPS_PER_BUFFER']=str(ops)
            env.pop('MLX_MAX_MB_PER_BUFFER',None);env.pop('MLX_METAL_FAST_SYNCH',None)
            subprocess.run([sys.executable,__file__,'--child',str(ops),str(block)],env=env,check=True)
            records.append(json.loads((OUT/f'{ops}-{block}.json').read_text()))
    import statistics
    med={ops:statistics.median(t for r in records if r['ops']==ops for t in r['seconds']) for ops in candidates}
    selected=min(candidates,key=lambda k:(med[k],k))
    result={'adopt':selected!=40 and med[selected]<=.8*med[40],'selected_ops':selected,'median_seconds':med,'speedup':med[40]/med[selected],'official_labels_used':False,'learned_weights_changed':False,'records':records}
    Path('results/commandbuffer-profile.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2),flush=True)
else:
    ops,block=map(int,sys.argv[-2:])
    assert os.environ.get('MLX_MAX_OPS_PER_BUFFER')==str(ops)
    assert 'MLX_METAL_FAST_SYNCH' not in os.environ
    import numpy as np,mlx.core as mx
    from ase import Atoms
    from orb_g2_cv.ase import load_calculator
    calc=load_calculator('artifacts/orb-g2-groupcv-v1.json',device='mps',base_checkpoint='/Users/nk/Desktop/diatomics-improvement-20261116/checkpoints/orb-v3-conservative-inf-mpa-ranged.ckpt')
    rows=sorted([json.loads(l) for l in Path('data/train-cache.jsonl').read_text().splitlines()],key=lambda r:r['name'])[:20]
    atoms=[Atoms(numbers=r['numbers'],positions=r['positions'],cell=r['cell'],pbc=r['pbc']) for r in rows]
    def predict(a):
        calc.reset();calc.get_potential_energy(a);calc.get_forces(a);calc.get_stress(a)
    for a in atoms:predict(a)
    seconds=[]
    for a in atoms:
        mx.synchronize();start=time.perf_counter();predict(a);mx.synchronize();seconds.append(time.perf_counter()-start)
    record={'ops':ops,'block':block,'pid':os.getpid(),'device':str(mx.default_device()),'mb_limit':'default40','fast_synch':'unset','names':[r['name'] for r in rows],'seconds':seconds}
    (OUT/f'{ops}-{block}.json').write_text(json.dumps(record,indent=2)+'\n')
    print(f'block {block} ops {ops}: {float(np.median(seconds)):.6f}s',flush=True)
