python generate_structures.py
mtt train options.yaml
mtt train options-llpr.yaml -o model-llpr.pt
mtt eval model-llpr.pt eval.yaml -b 20
python plot_uncertainties.py
