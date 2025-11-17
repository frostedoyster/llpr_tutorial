import ase.build
from metatomic.torch.ase_calculator import MetatomicCalculator
from metatomic.torch import ModelOutput


# Build a fcc copper structure
copper = ase.build.bulk("Cu", "fcc", cubic=True)

# Initialize an ASE calculator with a pre-trained potential
calculator = MetatomicCalculator("pet-mad.pt")

# Get global uncertainty
outputs = calculator.run_model(
    copper,
    outputs={"energy_uncertainty": ModelOutput(unit="eV", per_atom=False)},
)
global_uncertainty = outputs["energy_uncertainty"].block().values.item()
print("Global energy uncertainty (eV):", global_uncertainty)

# Get local uncertainties (local prediction rigidity)
outputs = calculator.run_model(
    copper,
    outputs={"energy_uncertainty": ModelOutput(unit="eV", per_atom=True)},
)
local_uncertainty = outputs["energy_uncertainty"].block().values.squeeze(-1).cpu().numpy()
print("Local energy uncertainties (eV):", local_uncertainty)

########## Rattle the atoms to see how uncertainties change #############
print("Rattling the atoms!")
copper.rattle(1.0)
#########################################################################

# Get global uncertainty after rattling
outputs = calculator.run_model(
    copper,
    outputs={"energy_uncertainty": ModelOutput(unit="eV", per_atom=False)},
)
global_uncertainty = outputs["energy_uncertainty"].block().values.item()
print("Global energy uncertainty after rattling (eV):", global_uncertainty)

# Get local uncertainties after rattling
outputs = calculator.run_model(
    copper,
    outputs={"energy_uncertainty": ModelOutput(unit="eV", per_atom=True)},
)
local_uncertainty = outputs["energy_uncertainty"].block().values.squeeze(-1).cpu().numpy()
print("Local energy uncertainties after rattling (eV):", local_uncertainty)
