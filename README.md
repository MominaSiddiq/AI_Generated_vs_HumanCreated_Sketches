# AI_Generated_vs_HumanCreated_Sketches

## Overview

This repository contains the implementation of a project aimed at distinguishing between AI-generated and human-created sketches using deep learning models. Three approaches were explored: a baseline Convolutional Neural Network (CNN) developed from scratch, ResNet-50, and VGG-16, leveraging transfer learning for enhanced feature extraction and generalization. The study evaluates the effectiveness, challenges, and limitations of each model on a nuanced and abstract dataset.

## Features

Baseline CNN Model: Custom-built architecture to establish foundational benchmarks.
ResNet-50: Pre-trained model with residual connections, fine-tuned for nuanced classification.
VGG-16: Minimal augmentation and strategic fine-tuning for robust generalization.
Data Augmentation: Techniques such as flipping, rotation, and brightness adjustments for improved training diversity.
Evaluation Metrics: Validation accuracy, loss curves, confusion matrices, and ROC-AUC scores.
Dataset

The dataset includes AI-generated and human-created sketches. Training and testing subsets were balanced for optimal model evaluation:

Subset Dataset: 1,000 images for rapid prototyping and architectural adjustments.
Full Dataset: Comprehensive dataset for final model training and evaluation.

## Key Results

Model	Training Accuracy	Validation Accuracy	Validation Loss	
CNN	       67.75%	            63.30%	             0.6786	
ResNet-50	 99.01%	            66.60%	             0.788	
VGG-16	   81.23%	            75.02%	             0.621	


## Prerequisites

Python 3.7 or higher
TensorFlow 2.x
Keras
NumPy
Matplotlib
Scikit-learn

## Installation

Clone the repository
git clone https://github.com/MominaSiddiq/AI_Generated_vs_HumanCreated_Sketches.git

Nevigate to the Project 
cd ai-vs-human-sketch-classification

install dependencies 
pip install -r requirements.txt

## Usage

### Prepare Dataset
This project uses a dataset split into train, valid, and test directories for sketch classification. To prepare the dataset, follow these steps:

### Organize the Dataset:
Create three directories: train, valid, and test.
Inside each directory, organize the data into subdirectories corresponding to the classes (e.g., AI-generated and Human-created).
The structure should look like this:

sketch_classification_project/
├── train/
│   ├── class1/
│   ├── class2/
├── valid/
│   ├── class1/
│   ├── class2/
├── test/
│   ├── class1/
│   ├── class2/

Upload to Google Drive:
Upload the sketch_classification_project folder to your Google Drive.
Ensure the directory paths in the code match your Drive structure.

Link Dataset to the Code:
Update the paths in the script to point to the dataset in your Google Drive:

train_dir = '/content/drive/MyDrive/sketch_classification_project/train'
valid_dir = '/content/drive/MyDrive/sketch_classification_project/valid'
test_dir = '/content/drive/MyDrive/sketch_classification_project/test'

Verify the Dataset:
Ensure all images are correctly labeled and organized in the respective directories.
Double-check the directory paths and permissions to avoid runtime errors.

## Train Models
This project provides scripts to train three deep learning models: CNN (from scratch), ResNet-50, and VGG-16, for classifying AI-generated and human-created sketches. Follow the steps below to train the models:

### Set Up the Environment:
Ensure all dependencies are installed as described in the Installation section.
Verify that the dataset is correctly prepared and accessible. See Prepare Dataset for details.

Choose the Model:
The code includes scripts to train:
A baseline CNN model from scratch.
ResNet-50 with transfer learning.
VGG-16 with transfer learning.

Open the corresponding script in the repository:
train_cnn.py for the CNN model.
train_resnet.py for ResNet-50.
train_vgg.py for VGG-16.

Modify Training Parameters (Optional):
Update hyperparameters such as learning rate, batch size, and the number of epochs in the script to customize the training process:

batch_size = 32
epochs = 50
learning_rate = 0.0003

Adjust the model structure if needed (e.g., fine-tuning layers for transfer learning).

### Start Training:
Run the script corresponding to the desired model. For example:
python train_cnn.py

This will start training the CNN model using the specified dataset and configurations.
Logs and outputs (e.g., training and validation accuracy/loss) will be displayed during the training process.

### Monitor Training:
Training metrics, such as accuracy and loss, will be saved to a file or displayed as plots.
EarlyStopping and learning rate adjustments (e.g., ReduceLROnPlateau) are integrated to optimize the training process.

### Evaluate the Model:
Once training is complete, the trained model is evaluated on the test dataset:
python evaluate_model.py --model_path <path_to_saved_model>

The evaluation script generates metrics such as confusion matrices and ROC curves to assess model performance.

### Directory Structure

The following directory structure is recommended to organize the project code and associated files:

project_root/ │ ├── datasets/ # Directory for dataset storage │ ├── train/ # Training dataset │ ├── valid/ # Validation dataset │ └── test/ # Test dataset │ ├── subsets/ # Directory for subsets of datasets (optional) │ ├── small_dataset/ # Subset for quick experimentation │ ├── models/ # Model training and evaluation scripts │ ├── cnn_training.py # Script for training CNN from scratch │ ├── resnet_training.py # Script for training ResNet-50 │ ├── vgg_training.py # Script for training VGG-16 │ └── evaluate_models.py # Script for evaluating and comparing models │ ├── visualizations/ # Output visualizations of training and evaluation │ ├── cnn_visualization/ # Visualizations for CNN training and results │ ├── resnet_visualization/ # Visualizations for ResNet-50 │ ├── vgg_visualization/ # Visualizations for VGG-16 │ └── comparison/ # Comparative visualizations across all models │ ├── outputs/ # Training outputs (saved models, logs, etc.) │ ├── saved_models/ # Directory for saved model files │ ├── training_logs/ # Logs generated during training │ └── plots/ # Generated plots for training and results │ ├── requirements.txt # Dependencies for the project ├── README.md # Project documentation └── main.py # Main script to run the project workflow

## Limitations

Dataset quality and labeling ambiguities influenced model accuracy.
ResNet-50 exhibited sensitivity to stylistic overlaps and overfitting.
Further optimization is required for handling highly abstract sketches.

## Future Work

Incorporate adaptive attention mechanisms for improved feature emphasis.
Explore additional datasets for broader generalization.
Apply explainable AI techniques for model interpretability.
Contributors

Momina Hammad (momnamehar753@gmail.com)

## License
This project is licensed under the MIT License. See the LICENSE file for details.


