"""Smooth unordered-pair energy with analytic conservative force and stress."""
from dataclasses import dataclass
import numpy as np
from ase.neighborlist import neighbor_list

CENTERS=np.linspace(.6,4.4,9)
CUTOFF=5.


def edge_design(atoms,width):
    if not np.isfinite(width) or width<=0 or not np.isfinite(atoms.positions).all():
        raise ValueError('finite coordinates and positive width required')
    i,j,r,v=neighbor_list('ijdD',atoms,CUTOFF,self_interaction=False)
    keep=(atoms.numbers[i]<=17)&(atoms.numbers[j]<=17)
    i,j,r,v=i[keep],j[keep],r[keep],v[keep]
    if np.any(r<=0):raise ValueError('coincident atoms are outside the radial domain')
    z1,z2=atoms.numbers[i],atoms.numbers[j]
    chem=np.column_stack([np.ones(len(i)),(z1+z2)/20,abs(z1-z2)/10,z1*z2/100])
    x=r/CUTOFF;envelope=(1-x)**4*(1+4*x);derivative=-20*x*(1-x)**3/CUTOFF
    offset=r[:,None]-CENTERS
    gaussian=np.exp(-.5*(offset/width)**2)
    e=np.einsum('ni,nj->nij',chem,envelope[:,None]*gaussian).reshape(len(i),36)
    negative_derivative=gaussian*(envelope[:,None]*offset/width**2-derivative[:,None])
    f=np.einsum('ni,nj->nij',chem,negative_derivative).reshape(len(i),36)
    return i,j,r,v,e,f


def energy_features(atoms,width):
    return .5*edge_design(atoms,width)[4].sum(axis=0)


@dataclass
class RadialHead:
    coefficients: np.ndarray
    width: float=.8

    def __post_init__(self):
        self.coefficients=np.asarray(self.coefficients,dtype=float).copy()
        if self.coefficients.shape!=(36,) or not np.isfinite(self.coefficients).all() or not np.isfinite(self.width) or self.width<=0:
            raise ValueError('36 finite coefficients and a positive width required')

    def evaluate(self,atoms):
        i,j,r,v,e,f=edge_design(atoms,self.width)
        energy=float(.5*e.sum(axis=0)@self.coefficients)
        radial=f@self.coefficients
        forces=np.zeros((len(atoms),3));stress=np.zeros((3,3))
        if len(i):
            np.add.at(forces,i,-radial[:,None]*v/r[:,None])
            if np.any(atoms.pbc):
                volume=atoms.get_volume()
                if volume<=0:raise ValueError('positive periodic volume required')
                stress=np.einsum('n,ni,nj->ij',-.5*radial/r,v,v)/volume
        return energy,forces,np.array([stress[0,0],stress[1,1],stress[2,2],stress[1,2],stress[0,2],stress[0,1]])
