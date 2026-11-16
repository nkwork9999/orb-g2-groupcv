"""Run the immutable training-only plan and write its single candidate."""
import hashlib,json,time
from pathlib import Path
from orb_g2_cv.training import train

root=Path(__file__).resolve().parents[1]
freeze=json.loads((root/'evidence/pre-fit-freeze.json').read_text())
for name,sha in freeze['sha256'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha,name
assert not (root/'data/holdout-cache.jsonl').exists(),'Do not train after holdout evaluation'
output=root/'artifacts/orb-g2-groupcv-v1.json'
assert not output.exists(),'Preserve the existing frozen candidate'
rows=[json.loads(line) for line in (root/'data/train-cache.jsonl').read_text().splitlines()]
manifest=json.loads((root/'data/split.json').read_text())
expected={r['name']:r for r in manifest['records'] if r['role']=='train'}
assert set(expected)=={r['name'] for r in rows}
for row in rows:
    for key in ['formula','numbers','positions','cell','pbc','role']:
        assert row[key]==expected[row['name']][key],(row['name'],key)
start=time.perf_counter();result=train(rows);seconds=time.perf_counter()-start
plan=json.loads((root/'plan.json').read_text())
artifact={'schema_version':1,'model_name':'ORB-G2 GroupCV v1','base':plan['base']|{'checkpoint_url':'https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orb-v3/orb-v3-conservative-inf-mpa-20250404.ckpt'},'head':{'coefficients':result['coefficients'],'width':result['width'],'alpha_already_applied':result['alpha'],'cutoff':5.,'centers':[.6,1.075,1.55,2.025,2.5,2.975,3.45,3.925,4.4]},'selection':{'ridge':result['ridge'],'alpha':result['alpha'],'inner_cv':result['inner_cv'],'training_ids':result['training_ids']},'provenance':{'plan_sha256':freeze['sha256']['plan.json'],'split_sha256':freeze['sha256']['data/split.json'],'train_cache_sha256':freeze['sha256']['data/train-cache.jsonl'],'source_sha256':{n:s for n,s in freeze['sha256'].items() if n.startswith('orb_g2_cv/')},'holdout_used_for_selection':False,'official_test_labels_used_for_fit_or_search':False,'test_set_informed':['diatomics']},'scope':'Experimental molecular atomization-energy calibration, no force/stress labels fitted; no demonstrated bulk or MD improvement'}
with output.open('x') as f:json.dump(artifact,f,indent=2,allow_nan=False);f.write('\n')
result['training_seconds']=seconds
(root/'results/training.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'seconds':seconds,'alpha':result['alpha'],'ridge':result['ridge'],'width':result['width'],'inner_cv':result['inner_cv'],'candidate_sha256':hashlib.sha256(output.read_bytes()).hexdigest()},indent=2))
