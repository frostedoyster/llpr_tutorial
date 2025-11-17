wget https://huggingface.co/lab-cosmo/pet-mad/resolve/v1.0.2/models/pet-mad-v1.0.2.ckpt
mtt export pet-mad-v1.0.2.ckpt -o pet-mad.pt
python a_get_uncertainty.py
python b_automatic_uncertainty.py
python c_uncertainty_propagation.py
