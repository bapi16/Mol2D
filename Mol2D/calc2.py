from Qchem_2D import Molecule


geom = '''
H 0.0 0.0
H 0.0 0.1

units bohr
'''
basis_str = "H: 32s"
mol = Molecule(geom, basis_str, charge=0, mult=1)
result = mol.scf(method='rhf', verbose=3)
    
#print("\nFinal Results:")
#print(f"Total energy: {result['energy']:.6f} Hartree")
#print(f"Alpha electrons: {result['n_alpha']}")
#print(f"Beta electrons: {result['n_beta']}")
