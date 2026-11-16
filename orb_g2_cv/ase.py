"""Native Python3.14 ORB ASE adapter with a general conservative correction."""
import hashlib
import json
import os
import tempfile
import urllib.request
from pathlib import Path
import numpy as np
from ase.calculators.calculator import Calculator,all_changes
from .potential import RadialHead


def verify_file(path,expected):
    path=Path(path)
    with path.open('rb') as file:actual=hashlib.file_digest(file,'sha256').hexdigest()
    if actual!=expected:raise ValueError('checkpoint SHA256 mismatch')
    return path


class CalibratedCalculator(Calculator):
    implemented_properties=['energy','forces','stress']

    def __init__(self,base,head):
        super().__init__();self.base=base;self.head=head

    def calculate(self,atoms=None,properties=None,system_changes=all_changes):
        super().calculate(atoms,properties,system_changes)
        if not len(self.atoms) or np.any(self.atoms.numbers<1) or not np.isfinite(self.atoms.positions).all():
            raise ValueError('nonempty physical atoms with finite coordinates required')
        base_atoms=self.atoms.copy()
        if not np.any(base_atoms.pbc) and abs(np.linalg.det(base_atoms.cell.array))<1e-10:
            extent=float(np.max(np.ptp(base_atoms.positions,axis=0)))
            base_atoms.set_cell([max(50.,2*extent+20.)]*3)
        e=float(self.base.get_potential_energy(base_atoms));f=self.base.get_forces(base_atoms);s=self.base.get_stress(base_atoms)
        de,df,ds=self.head.evaluate(self.atoms)
        self.results={'energy':e+de,'forces':np.asarray(f,dtype=float)+df,'stress':np.asarray(s,dtype=float)+ds}
        if any(not np.isfinite(value).all() for value in map(np.asarray,self.results.values())):
            raise ValueError('nonfinite calibrated prediction')


def load_calculator(checkpoint=None,device='cpu',dtype='float32',base_checkpoint=None):
    if device!='cpu' or dtype!='float32':
        raise ValueError('this frozen runtime requires CPU float32')
    path=Path(checkpoint) if checkpoint else Path(__file__).with_name('model.json')
    artifact=json.loads(path.read_text())
    if artifact.get('schema_version')!=1 or artifact['base']['kind']!='orb-v3-conservative-inf-mpa':
        raise ValueError('unsupported calibrated artifact schema or backbone')
    threads=artifact.get('runtime',{}).get('torch_threads',1)
    compiled=artifact['base'].get('compile',False)
    if type(threads) is not int or threads not in (1,2,4):
        raise ValueError('deployment torch_threads must be one of1,2,4')
    if type(compiled) is not bool:
        raise ValueError('deployment compile flag must be boolean')
    head=RadialHead(artifact['head']['coefficients'],width=artifact['head']['width'])
    pinned=artifact['base']['checkpoint_sha256']
    supplied=base_checkpoint or os.environ.get('ORB_G2_BASE_CHECKPOINT')
    if supplied:
        base_path=verify_file(supplied,pinned)
    else:
        directory=Path.home()/'.cache/orb-g2-cv';directory.mkdir(parents=True,exist_ok=True)
        base_path=directory/f'{pinned}.ckpt'
        if not base_path.exists():
            with tempfile.NamedTemporaryFile(dir=directory,delete=False) as stream:
                temporary=Path(stream.name)
                try:
                    with urllib.request.urlopen(artifact['base']['checkpoint_url'],timeout=60) as remote:
                        while data:=remote.read(1024*1024):stream.write(data)
                    stream.flush();verify_file(temporary,pinned)
                    os.replace(temporary,base_path)
                finally:
                    if temporary.exists():temporary.unlink()
        verify_file(base_path,pinned)
    import torch
    from orb_models.forcefield import pretrained
    from orb_models.forcefield.calculator import ORBCalculator
    torch.set_num_threads(threads)
    model=pretrained.orb_v3_conservative_inf_mpa(weights_path=str(base_path),device=device,precision='float32-highest',compile=compiled)
    base=ORBCalculator(model,device=device)
    if compiled:
        from ase import Atoms
        # Charge tracing/compilation to model load; retain no benchmark predictions.
        for atoms in [
            Atoms('CH',positions=[[0,0,0],[1.3,0,0]],cell=[50]*3,pbc=True),
            Atoms('H',positions=[[0,0,0]],cell=[50]*3,pbc=True),
            Atoms('CH',positions=[[0,0,0],[10,0,0]],cell=[50]*3,pbc=True),
        ]:
            base.get_potential_energy(atoms);base.get_forces(atoms);base.get_stress(atoms)
        base.reset()
    calculator=CalibratedCalculator(base,head)
    if compiled:calculator.mbd_acceleration=('torch_compile',)
    return calculator
