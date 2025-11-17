wget https://huggingface.co/lab-cosmo/pet-mad/resolve/v1.0.2/models/pet-mad-v1.0.2.ckpt
python generate_structures.py
mtt train options.yaml
mtt train options-llpr.yaml -o model-llpr.pt
mtt eval model-llpr.pt eval.yaml -b 20
python plot_uncertainties.py
