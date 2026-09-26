# Pistachio Image Classification with Transfer Learning

A computer vision project for classifying **Kirmizi Pistachio** and **Siirt Pistachio** varieties using **MobileNetV2 transfer learning with TensorFlow/Keras**.

The project demonstrates an end-to-end image classification workflow including dataset preparation, image preprocessing, transfer learning, model training, validation, performance evaluation, and model saving.

## Project Objective

The objective of this project is to classify pistachio images into two categories:

- **Kirmizi Pistachio**
- **Siirt Pistachio**

A pre-trained **MobileNetV2** model is used as the feature extractor, allowing visual features learned from ImageNet to be transferred to the pistachio classification task.

## Dataset

The dataset contains **2,148 pistachio images** belonging to two classes.

Dataset split:

```text
Total Images:        2,148
Training Images:     1,719
Validation Images:     429
```

Classes:

```text
Kirmizi_Pistachio
Siirt_Pistachio
```

The dataset used in the notebook is based on the pistachio image dataset available from the Murat Koklu dataset collection.

## Project Workflow

```text
Pistachio Image Dataset
        ↓
Dataset Inspection
        ↓
Image Visualization
        ↓
Train / Validation Split
        ↓
Image Preprocessing
        ↓
MobileNetV2
        ↓
Transfer Learning
        ↓
Classification Layer
        ↓
Training & Validation
        ↓
Performance Evaluation
        ↓
Model Saving
```

## Transfer Learning Approach

Instead of training a convolutional network entirely from scratch, this project uses **MobileNetV2 pre-trained on ImageNet**.

The convolutional base is initially frozen:

```python
base_model.trainable = False
```

This allows the model to reuse previously learned visual features while training a new classification head specifically for the two pistachio varieties.

## Model Architecture

The classification pipeline consists of:

```text
Input Image (128 × 128 × 3)
        ↓
Image Rescaling
        ↓
Pre-trained MobileNetV2
        ↓
Global Average Pooling
        ↓
Dropout (0.3)
        ↓
Dense Layer
        ↓
2-Class Softmax Output
```

The model is compiled using:

```text
Optimizer: Adam
Loss: Categorical Crossentropy
Metric: Accuracy
```

## Training

The model was trained for:

```text
Epochs: 10
```

Training and validation datasets were created using an **80/20 split** with a fixed random seed for reproducibility.

## Results

The final model achieved:

| Metric | Result |
| --- | ---: |
| Training Accuracy | **95.99%** |
| Validation Accuracy | **96.74%** |
| Training Loss | **0.0943** |
| Validation Loss | **0.0939** |

The close training and validation results indicate that the model generalized well to the validation dataset.

## Technologies

- Python
- TensorFlow
- Keras
- MobileNetV2
- NumPy
- Matplotlib
- Google Colab
- Jupyter Notebook
- Computer Vision
- Transfer Learning

## Project Structure

```text
pistachio-image-classification-transfer-learning/
│
├── pistachio_image_classification_transfer_learning.ipynb
├── pistachio_transfer_learning_model.keras
└── README.md
```

## Model Saving

The trained model can be saved using the native Keras format:

```python
model.save("pistachio_transfer_learning_model.keras")
```

and loaded again with:

```python
loaded_model = tf.keras.models.load_model(
    "pistachio_transfer_learning_model.keras"
)
```

## Key Learning Points

This project demonstrates practical experience with:

- Image classification
- Computer vision
- Transfer learning
- Pre-trained neural networks
- MobileNetV2
- TensorFlow / Keras
- Dataset preparation
- Train / validation splitting
- Model training
- Performance visualization
- Model persistence

## Project Purpose

This project was developed for **educational, demonstration, and portfolio purposes**.

It shows how transfer learning can be applied to a real-world two-class image classification problem using a pre-trained convolutional neural network.
