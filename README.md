# GenModelsQuantumWalksThesis
In this repository, there are python files that where used for my thesis, "Modeling, Analyzing and Exploring Correlated Quantum Walks Using Deep Generative Models".

<ins>QWproject folder</ins> <br/>
This folder contains the code that is required for generation of data:
- "TwoParticlesQW.py" contains the code that recieves the initial state and parameters and calculates the correlation or U.
- "example_batch_creator.py" contains the code that creates a dataset with variable physical parameters and saves it as an image.
- "KL_for_batch_diffusion.py" calculates the KL distribution for a batch of generated samples in the diffusion model.
- "KLscore_for_batch.py" calculates the KL distribution for a batch of generated samples in the StyleGAN2 model.

<ins>direction_finder folder</ins> <br/>
This folder contains the code for the parts of the thesis regarding the physical direction research with StyleGAN2 model.
-  "pca_analyzer.py" contains the code that finds the directions with various methods and analyzing the results.
- "model.py" and "train.py" inside "inferPhysicalParams" subfolder contain the model and training code of the hidden parameter estimation that was used in the physical direction research.

<ins>diffusion folder</ins> <br/>
This folder contains the code for the diffusion model parts of the thesis. 
- "Unet.py" contains the diffusion denoising model.
- "Diffusion.py" contains the train and sample generation codes.
- "interpolation.py" contains the code performing the imeage interpolation.
<br/>Some of the code in the diffusion part took inspiration from the code in: "https://github.com/dome272/Diffusion-Models-pytorch"

<ins>StyleGAN2 model</ins> <br/>
The StyleGAN2 model that was used is the official pytorch implementation, that can be found in "https://github.com/NVlabs/stylegan3" <br/>
The model was trained using the following settings: <br/>
python stylegan3/train.py --cfg=stylegan2 --batch=16  --gamma=10
