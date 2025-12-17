# aircraft-recognition-clip-resnet
A multi-modal aircraft search engine allowing users to perform Text-to-Image and Image-to-Image retrieval using OpenAI CLIP and ResNet-50 features.

# Hybrid Aircraft Image Retrieval System

This project implements a Content-Based Image Retrieval (CBIR) system specifically designed for aircraft classification and search. By integrating **OpenAI's CLIP (Contrastive Language-Image Pre-training)** model with **ResNet-50** architecture, the system enables users to perform semantic searches across an aircraft image dataset.

Unlike traditional classification models, this system functions as a search engine, allowing for both visual and textual queries to identify aircraft types based on high-dimensional feature embeddings.

## Project Overview

The primary objective of this project is to bridge the gap between visual data and natural language in the context of aviation. The system leverages the power of multi-modal learning to understand the semantic relationship between aircraft images and their textual descriptions.

Key capabilities include:
* **Text-to-Image Retrieval:** Users can input natural language queries (e.g., "F-16 fighter jet taking off") to find relevant images.
* **Image-to-Image Retrieval:** Users can upload an image to find visually and semantically similar aircraft in the dataset.

## Methodology

The architecture combines two powerful deep learning models to extract and compare features:

1.  **Feature Extraction:**
    * **Visual Encoder:** A pre-trained **ResNet-50** and CLIP's visual transformer are used to extract high-level visual features from aircraft images.
    * **Text Encoder:** CLIP's text transformer converts user text queries into high-dimensional vectors.

2.  **Similarity Measurement:**
    * The system utilizes **Cosine Similarity** to measure the distance between the query vector (text or image) and the database vectors.
    * Results are ranked by similarity score, ensuring the most relevant images are retrieved first.

## Technical Stack

* **Programming Language:** Python 3.x
* **Deep Learning Frameworks:** PyTorch, TensorFlow/Keras
* **Models:** OpenAI CLIP, ResNet-50 (Transfer Learning)
* **Data Processing:** NumPy, Pandas, Pillow (PIL)
* **Visualization:** Matplotlib

## Installation

To set up the project locally, follow these steps:

1.  Clone the repository:
    ```bash
    git clone [https://github.com/hilallertunc/aircraft-recognition-clip-resnet.git](https://github.com/hilallertunc/aircraft-recognition-clip-resnet.git)
    cd aircraft-recognition-clip-resnet
    ```

2.  Install the required dependencies:
    ```bash
    pip install torch torchvision
    pip install git+[https://github.com/openai/CLIP.git](https://github.com/openai/CLIP.git)
    pip install tensorflow numpy pandas matplotlib
    ```

##  Model Access & Usage

Due to GitHub's file size limitations, the pre-trained model weights and the aircraft dataset are not included in this repository.

The code provided here serves as a reference for the architecture and logic of the system. If you wish to run the project or need access to the trained model files, please **contact me**.

* **Email:** hilallertunc@gmail.com

## Acknowledgments

This project utilizes the pre-trained models provided by OpenAI (CLIP) and the ResNet-50 architecture available in Keras/PyTorch applications.

---
**Developer:** Hilal Beyza Ertunc










