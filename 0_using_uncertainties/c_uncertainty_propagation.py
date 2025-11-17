import torch
import ase.build
from metatomic.torch.ase_calculator import MetatomicCalculator
from metatomic.torch import ModelOutput


# Build a fcc copper structure and replicate it to have more atoms
copper = ase.build.bulk("Cu", "fcc", cubic=True)
copper *= (3, 3, 3)  # 108 atoms

# Create a derived structure where 1 atom has been removed
copper_defective = copper.copy()
del copper_defective[50]  # remove atom number 50

# Set the calculator to both structures
calculator = MetatomicCalculator("pet-mad.pt")
copper.calc = calculator
copper_defective.calc = calculator

# Print the defect formation energy
E_perfect = copper.get_potential_energy()
E_defective = copper_defective.get_potential_energy()
formation_energy = E_defective - (107/108) * E_perfect  # adjust for number of atoms
print("Defect formation energy (eV):", formation_energy)

# Find the uncertainty from analytical propagation of the LLPR model
inverse_covariance = calculator._model.module.inv_covariance_energy_uncertainty
multiplier = calculator._model.module.multiplier_energy_uncertainty ** 2
last_layer_features_perfect = calculator.run_model(
    copper,
    outputs={"mtt::aux::energy_last_layer_features": ModelOutput(per_atom=False)},
)["mtt::aux::energy_last_layer_features"].block().values
last_layer_features_defective = calculator.run_model(
    copper_defective,
    outputs={"mtt::aux::energy_last_layer_features": ModelOutput(per_atom=False)},
)["mtt::aux::energy_last_layer_features"].block().values
feature_difference = last_layer_features_defective - (107/108) * last_layer_features_perfect
variance = multiplier * torch.einsum("...i,ij,...j->...", feature_difference, inverse_covariance, feature_difference)
uncertainty = variance.item()**0.5
print("Defect formation energy uncertainty (eV):", uncertainty, "[analytical propagation]")

# Find the uncertainty from ensemble propagation
ensemble_perfect = calculator.run_model(
    copper,
    outputs={"energy": ModelOutput(), "energy_ensemble": ModelOutput(per_atom=False)},
)["energy_ensemble"].block().values
ensemble_defective = calculator.run_model(
    copper_defective,
    outputs={"energy": ModelOutput(), "energy_ensemble": ModelOutput(per_atom=False)},
)["energy_ensemble"].block().values
formation_energies_ensemble = ensemble_defective - (107/108) * ensemble_perfect
uncertainty_ensemble = formation_energies_ensemble.std().item()
print("Defect formation energy uncertainty (eV):", uncertainty_ensemble, "[ensemble propagation]")
