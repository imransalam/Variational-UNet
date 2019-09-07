# Variational-UNet
The code is written in TensorFlow 2.0 and trained over CamVid dataset.

Adding a probability distribution layer after every skip connection. The idea is that each layer contributes mask formation, hence to sample from probability distribution from each layer makes sense rather than sampling from the last encoder layer.
