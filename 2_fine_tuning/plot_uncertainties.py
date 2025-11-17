import numpy as np
import ase.io


structures = ase.io.read("evaluation.xyz", ":")
evaluated_structures = ase.io.read("output.xyz", ":")

uncertainties = [s.info["energy_uncertainty"] for s in evaluated_structures]
predicted_energies = [s.get_potential_energy() for s in evaluated_structures]
true_energies = [s.get_potential_energy() for s in structures]

# Hard-code the zoomed in region of the plot and iso-lines.
quantile_lines = [0.00916, 0.10256, 0.4309805, 1.71796, 2.5348, 3.44388]
min_val, max_val = 3e-3, 3e-2

import matplotlib.pyplot as plt
plt.figure(figsize=(6, 6))
plt.plot([min_val, max_val], [min_val, max_val], "k--", lw=0.75)
for factor in quantile_lines:
    plt.plot([min_val, max_val], [factor * min_val, factor * max_val], "k:", lw=0.75)
plt.scatter(uncertainties, np.abs(np.array(predicted_energies) - np.array(true_energies)))
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Predicted energy uncertainty (eV)")
plt.ylabel("Absolute error in predicted energy (eV)")
plt.title("Predicted uncertainty vs actual error")
plt.grid()
plt.savefig("uncertainty_vs_error.png", dpi=300)


