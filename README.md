# ClearVision: High-Fidelity Image Restoration with ESRGAN

## Overview

ClearVision is a deep learning project dedicated to restoring degraded images using the Enhanced Super-Resolution Generative Adversarial Network (ESRGAN). Our goal is to recover intricate details and textures lost to noise, blur, or compression, delivering sharp images for photography, medical imaging, and surveillance. ESRGAN’s focus on perceptual quality produces visually compelling results, distinguishing this work from traditional methods.

A Streamlit-based web tool allows users to upload 256×256 RGB images and view restored outputs instantly. The app is deployed at [ClearVision Streamlit App](https://clearvision2.streamlit.app/).

You can view a demo video [here](https://drive.google.com/file/d/1zQLSS2aV6gn9Oxebdq0FiGp5XC8PDap4/view?usp=sharing).

## Dataset

We used the [CelebA Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), a collection of over 200,000 celebrity face images, each annotated with 40 attributes and five facial landmarks. Its variety in expressions, textures, and lighting makes it perfect for training ESRGAN to handle real-world challenges. Images were resized to 256×256 for computational efficiency.

### Image Degradation Pipeline

To simulate real-world degradation, we built a custom pipeline in `data_prep.py`:

* **Gaussian Noise**: Adds noise $\varepsilon \sim \mathcal{N}(0, 25^2)$ to pixels $x$, yielding $x + \varepsilon$, mimicking sensor noise.
* **Gaussian Blur**: Applies a $5 \times 5$ kernel with $\sigma = 1$, computed as $x * G_{\sigma}$, simulating motion or defocus blur.
* **JPEG Compression**: Encodes/decodes images at quality 30, introducing web-like artifacts.

This pipeline ensures robust handling of diverse degradations.

## Model Architecture

ESRGAN, implemented in model.py, is an improved version of SRGAN designed to produce high-quality images that are perceptually convincing. Here’s how it works in detail:

1. **Generator Network**:

   * Based on **Residual-in-Residual Dense Blocks (RRDBs)**, which stack residual dense blocks in a residual manner. This allows deeper networks without exploding or vanishing gradients.
   * Each RRDB includes multiple densely connected layers, which encourage feature reuse and help in learning complex patterns.
   * The generator does **not use batch normalization**, which can introduce artifacts in super-resolution tasks.
   * After feature extraction, a **PixelShuffle layer** is used for upsampling. This rearranges the channel dimensions into spatial dimensions to scale the image by 4× efficiently.

2. **Relativistic Discriminator**:

   * Instead of a binary real/fake judgment, the discriminator estimates the probability that a real image looks more realistic than a generated one. This "relativistic" perspective improves the adversarial dynamics.
   * This leads to better convergence and makes the model focus on learning textures that are more natural and plausible.

3. **Loss Functions**:

   * **Perceptual Loss (weight = 1.0)**: Uses feature maps from the VGG19 network (specifically the relu5\_4 layer) to ensure the generated image looks perceptually close to the target.
   * **Adversarial Loss (weight = 0.05)**: Encourages the generator to produce images that fool the discriminator by making them look more photo-realistic.
   * **Pixel Loss (weight = 0.005)**: A basic MSE loss between generated and ground truth pixels. While not always sufficient alone, it helps stabilize early training.


## Training

We trained ESRGAN for 25 epochs on CelebA with our degradation pipeline. Settings:

* **Optimizers**: Adam, generator $\text{lr} = 2 \times 10^{-4}$, discriminator $\text{lr} = 1 \times 10^{-4}$.
* **Loss Weights**: Perceptual = 1, Adversarial = 0.05, Pixel = 0.005.

### Results

Below are examples from the 10th and 25th epochs, showing the improvement in detail restoration.

**10 Epochs**:

PSNR=19.7 dB, SSIM=0.52



![10 Epochs](https://github.com/teja123git/Clear_Vison/blob/main/10_epochs.jpg)


**25 Epochs**:

 PSNR=24.3 dB, SSIM=0.72
 
![25 Epochs](https://github.com/teja123git/Clear_Vison/blob/main/25_epochs.jpg)

* **Test Average**:

  * PSNR: 23.4 dB
  * SSIM: 0.67

Hardware constraints limited training to 25 epochs, but performance improved steadily.

### Challenges:

* Discriminator dominance required noise injection for training stability.
* Optimal learning rates and loss weights demanded extensive trials.
* Realistic degradation pipeline was computationally intensive.
* GPU limitations capped training duration.
* PSNR/SSIM didn’t fully reflect visual quality, relying on perceptual loss.


### Quantitative Results:

* **PSNR**: 23.4 dB (test average).
* **SSIM**: 0.67, indicating strong structural similarity.

### Qualitative Results:

Restored images show sharper details compared to degraded inputs.


## Repository Structure

| **File/Folder**               | **Description**                                    |
| ----------------------------- | -------------------------------------------------- |
| `data_prep.py`                | Loads, splits, and degrades CelebA images.         |
| `model.py`                    | Defines ESRGAN (RRDB, discriminator, losses).      |
| `ESRGAN_train.py`             | Runs training with data pipeline and optimization. |
| `Evaluation.py`               | Computes PSNR, SSIM, and visualizes results.       |
| `ESRGAN_Streamlit_app/`       | Hosts web app for image restoration.               |
| `ESRGAN_Streamlit_app/app.py` | Script for uploading and processing images.        |
| `vgg_pre_training.py`         | Sets up VGG19 for perceptual loss.                 |
| `.gitattributes`              | Manages Git LFS for `generator_epoch_25.pth`.      |

## Streamlit Application

The Streamlit app in `ESRGAN_Streamlit_app/app.py` lets users upload 256×256 RGB images and view restored results. Run locally:

```bash
cd ESRGAN_Streamlit_app
streamlit run app.py
```

## Setup and Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/teja123git/Clear_Vision.git
   cd Clear_Vision
   ```

2. **Set Up Virtual Environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:

   ```bash
   pip install -r ESRGAN_Streamlit_app/requirements.txt
   ```


To train from scratch:

```bash
python ESRGAN_train.py
```

## Evaluation

Run evaluation:

```bash
python Evaluation.py
```


## Future Work

* Train on diverse datasets like FFHQ or DIV2K for broader robustness.
* Optimize for real-time inference.
* Benchmark against RCAN or SRFlow.
* Extend training to 50+ epochs with better hardware.
* Explore SwinIR or Real-ESRGAN for enhanced performance.

## References

* [ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks](https://arxiv.org/abs/1809.00219)
* [CelebA Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
* [Streamlit Documentation](https://streamlit.io/)

