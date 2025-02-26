
# calc.py
from Qchem_2D import Molecule

geom = '''
Li  0.0 0.0

 
units bohr
'''

mol = Molecule(geom, "Li: 36s", charge=0, mult=2)
result = mol.scf(method='uhf',verbose=2)

