To easy use this work and its codes, fellow the following

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Description of Each Script

EnhancedGAN.py
Implements an enhanced Generative Adversarial Network (GAN) for improving image quality.
Features include image preprocessing, noise addition, generator and discriminator models, and GAN training pipeline.
Outputs an enhanced image saved as enhanced_image.jpg.

SRCNN.py
Implements a Super-Resolution Convolutional Neural Network (SRCNN) for image super-resolution.
Simulates low-resolution inputs, trains the SRCNN model, and generates high-resolution outputs.
Outputs a super-resolved image saved as output_image.jpg.

DVDNet.py
Implements a Deep Video Denoising Network (DVDNet) for video frame denoising.
Processes video frames with Gaussian noise and applies the DVDNet model to restore them.
Outputs a denoised frame saved as denoised_frame.jpg.

DANet.py
Implements a Dual Attention Network (DANet) for image denoising and enhancement.
Utilizes position and channel attention modules to enhance image features.
Outputs an enhanced image saved as danet_output.jpg.

GBIF.py
Graph-based Image Filtering (GBIF) is implemented for image restoration using graph signal processing techniques.
Outputs a filtered image saved as filtered_image.jpg.

PSO_GCRA_Framework.py
Combines Particle Swarm Optimization (PSO) with Graph-based approaches for image and signal analysis.
Includes adaptive filters, wavelet transforms, and optimization algorithms.



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
General Requirements
Install the required Python libraries using:
pip install tensorflow scikit-image numpy network


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Running Each Script

EnhancedGAN.py
Preprocessing: Place input images in the project directory.
Training: Run the script to train the GAN model and generate enhanced images.
Command Example:
python EnhancedGAN.py
Replace example_image.jpg with your image path in the code.
SRCNN.py

Preprocessing: Simulates low-resolution input and upscale processes.
Training: Executes training with synthetic data and saves a high-resolution image.
Command Example:
python SRCNN.py
DVDNet.py

Preprocessing: Add Gaussian noise to the input video frame.
Training: Executes the denoising process with DVDNet and saves the restored frame.
Command Example:
python DVDNet.py
DANet.py

Preprocessing: Prepares and resizes images.
Training: Runs the model with position and channel attention for denoising.
Command Example:
python DANet.py
GBIF.py

Preprocessing: Builds an image graph and applies graph-based smoothing.
Training: Constructs and reconstructs the filtered image graph.
Command Example:
python GBIF.py
PSO_GCRA_Framework.py

Preprocessing: Applies filters, transforms, and optimizations for analysis.
Command Example:
python PSO_GCRA_Framework.py
Dataset Details


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%55
This framework supports the following datasets, which can be accessed at the provided links:

VISION Dataset
Source Identification in Images and Videos.
Download https://lesc.dinfo.unifi.it/VISION/

HDR Dataset
Assesses HDR and SDR Imaging Algorithms.
Download https://drive.google.com/drive/folders/1ygjPeVAX5-v-xlzODj6FgXzoQhn_BCit

BOREAS Dataset
Seasonal Self-Driving Vehicle Dataset.
Download https://arxiv.org/abs/2203.10168

Bosch Small Traffic Lights Dataset
Contains Traffic Light Bounding Boxes for Evaluation.
Download https://hci.iwr.uni-heidelberg.de/content/bosch-small-traffic-lights-dataset
Ensure datasets are downloaded and correctly linked to the scripts as per the preprocessing instructions.



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Example Usages
Training EnhancedGAN on the VISION Dataset:
python EnhancedGAN.py --input_data_dir path_to_VISION_dataset --epochs 1000 --batch_size 32

Denoising a frame using DVDNet:
python DVDNet.py --input_frame example_frame.jpg

Filtering images with GBIF:
python GBIF.py --input_image example_image.jpg


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Cite this work:

@article{abualigah2025multi_resolution,
  title={Multi-Resolution Deep Learning Framework for Efficient Real-Time Image and Video Processing},
  author={Abualigah, Laith and Alomari, Saleh Ali and Almomani, Mohammad H. and Abu Zitar, Raed and Saleem, Kashif and Migdady, Hazem and Snasel, Vaclav and Smerat, Aseel and Zhang, Peiying},
  journal={The Visual Computer},
  publisher={Springer},
  year={2025}
}


Abualigah et al. (2025). Multi-Resolution Deep Learning Framework for Efficient Real-Time Image and Video Processing. The Visual Computer. Springer







