import hashlib
import json
import pytest
from orb_g2_cv.reproduce import load_training_rows


def fixture(root,role='train'):
    (root/'data').mkdir();(root/'artifacts').mkdir()
    rows=[{'name':'a','formula':'H2O','role':role}]
    payload=(json.dumps(rows[0])+'\n').encode()
    (root/'data/train-cache.jsonl').write_bytes(payload)
    artifact={'provenance':{'train_cache_sha256':hashlib.sha256(payload).hexdigest()},'selection':{'training_ids':['a']}}
    (root/'artifacts/orb-g2-groupcv-v1.json').write_text(json.dumps(artifact))
    return rows


def test_reproducer_loads_only_original_training_cache(tmp_path):
    rows=fixture(tmp_path)
    assert load_training_rows(tmp_path)==rows
    (tmp_path/'data/holdout-cache.jsonl').write_text('malformed changed outer labels')
    assert load_training_rows(tmp_path)==rows


def test_reproducer_rejects_changed_cache_and_outer_roles(tmp_path):
    fixture(tmp_path)
    (tmp_path/'data/train-cache.jsonl').write_bytes(b'changed')
    with pytest.raises(ValueError):load_training_rows(tmp_path)


def test_reproducer_rejects_holdout_even_if_hash_is_updated(tmp_path):
    fixture(tmp_path,role='holdout')
    with pytest.raises(ValueError):load_training_rows(tmp_path)
