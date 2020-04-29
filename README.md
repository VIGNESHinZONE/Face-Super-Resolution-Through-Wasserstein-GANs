# Face-Super-Resolution-Through-Wasserstein-GANs
Paper implementation in Pytorch

Paper Link - https://arxiv.org/pdf/1705.02438.pdf

Author's Code - https://github.com/MandyZChen/srez {TensorFlow }

# Docker Build:
You can build the docker image using the following command
`cd Dockerfiles
docker build -t WGAN_resolution:latest .` 

# Training
for the generating new celeb images add the following command in main.py
`from genrator import WGAN_GP`

for the face resolution part add the following command in main.py
`from Face_Resolution_model import WGAN_GP`

and then run this command on
`python main.py`

# Results:

Resolution images:

<img src="https://github.com/VIGNESHinZONE/Face-Super-Resolution-Through-Wasserstein-GANs/blob/master/runs/individualImage.png" height="500" width="500">

Image Generation:

<img src="https://github.com/VIGNESHinZONE/Face-Super-Resolution-Through-Wasserstein-GANs/blob/master/runs/individualImage%20(2).png" height="500" width="500">
