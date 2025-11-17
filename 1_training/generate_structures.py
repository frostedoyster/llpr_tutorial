import ase.build
import ase.io
from ase.calculators.emt import EMT

calculator = EMT()
structure = ase.build.bulk("Al", "fcc", cubic=True)

structures = []
for i in range(1000):
    s = structure.copy()
    s.rattle(0.3, seed=i)
    s.calc = calculator
    s.info["energy"] = s.get_potential_energy()
    s.arrays["forces"] = s.get_forces()
    s.info["stress"] = s.get_stress(voigt=False)
    s.calc = None
    structures.append(s)

ase.io.write("dataset.xyz", structures[:100])
ase.io.write("evaluation.xyz", structures[100:])
