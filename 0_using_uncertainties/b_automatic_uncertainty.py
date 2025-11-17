import numpy as np
import ase.build
import ase.units
from ase.md import VelocityVerlet
from metatomic.torch.ase_calculator import MetatomicCalculator

np.random.seed(42)


# Build a fcc copper structure
copper = ase.build.bulk("Cu", "fcc", cubic=True)

# Initialize an ASE calculator with a pre-trained potential
calculator = MetatomicCalculator("pet-mad.pt")

# Set the calculator to the copper structure
copper.calc = calculator

# Initial small rattle to avoid zero forces
copper.rattle(0.1)

# Run some molecular dynamics
dyn = VelocityVerlet(copper, timestep=1.0 * ase.units.fs)
dyn.run(100)
print("First MD run completed")

# Run again, this time with a crazy structure
copper.rattle(5.0)
dyn.run(100)
print("Second MD run completed (hopefully with a warning!)")
