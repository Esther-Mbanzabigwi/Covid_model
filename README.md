# **Lung Segmentation of Normal, Viral Pneumonia, and COVID-affected Lungs**

This project leverages Convolutional Neural Networks (CNNs) to perform lung segmentation from chest X-ray images, focusing on three categories:

Normal Lungs
Viral Pneumonia-affected Lungs
COVID-19-affected Lungs

By utilizing the COVID-19 Radiography Dataset, the CNN model is designed to accurately identify and segment lung regions as a preliminary step for the diagnosis of these lung conditions. This project aims to support healthcare professionals by automating the detection process, potentially enhancing early intervention.

---


### Workflow:

1. Acquire a publicly available dataset suitable for classification tasks.  
   - Load the dataset.
   - Split the data into training and testing sets.

2. Implement a 2 Models :
    -A simple machine learning model based on neural networks on the chosen dataset without any optimization techniques and
    -A model applying at least 3 optimization techniques
   

3. Discuss the results of optimizations and parameter setting
   - Apply L2 regularization using the Adam optimizer.
   - Monitor the model's performance using early stopping.
 

4. Make predictions using test data
   -Make sure you have training,
   -validation, 
   -test datasets

---

## Dataset

- The dataset used for this project contains several water quality parameters and a binary target label (`Potability`).
- The dataset is split into an 80/20 ratio for training and testing purposes.


Lung Segmentation of Normal, Viral Pneumonia, and COVID-affected Lungs
Project Overview
This project leverages Convolutional Neural Networks (CNNs) to perform lung segmentation from chest X-ray images, focusing on three categories:

Normal Lungs
Viral Pneumonia-affected Lungs
COVID-19-affected Lungs
By utilizing the COVID-19 Radiography Dataset, the CNN model is designed to accurately identify and segment lung regions as a preliminary step for the diagnosis of these lung conditions. This project aims to support healthcare professionals by automating the detection process, potentially enhancing early intervention.

**Dataset**
The project uses the COVID-19 Radiography Dataset, which contains chest X-ray images categorized into:


Normal
Viral Pneumonia
COVID-19

link to the dataset :[(https://www.kaggle.com/datasets/preetviradiya/covid19-radiography-dataset?select=COVID-19_Radiography_Dataset)]

Each image undergoes preprocessing before being fed into the CNN model for segmentation and classification.

**Project Structure**



bash
Copy code
├── data/                   # Folder containing the dataset
├── notebooks/              # Jupyter Notebooks for model training and analysis
├── models/                 # Trained models and checkpoints
├── scripts/                # Python scripts for data processing and model deployment
├── results/                # Results and visualizations
└── README.md               # Project overview and instructions
**Requirements**


To run this project, you will need the following libraries and frameworks installed:

Python 3.x
TensorFlow/Keras
NumPy
Matplotlib
OpenCV
scikit-learn

You can install the required libraries using the following command:

bash
Copy code
pip install -r requirements.txt
Model Training

**The CNN model is trained using the chest X-ray images from the dataset**

Key steps include:

Data Preprocessing: Images are resized, normalized, and augmented to enhance model performance.

Model Architecture: The CNN model consists of several convolutional layers, followed by pooling, dropout, and fully connected layers.

Training: The model is trained with training and validation data splits, using optimization techniques to reduce overfitting.

---


##**Regularization and Optimization Techniques**


We implemented three optimization techniques to improve the model's accuracy and prevent overfitting:

Dropout Regularization
Batch Normalization
Adam Optimizer

---

##**Future Work**


Model Optimization: Further improvements to the model using advanced techniques such as fine-tuning with transfer learning.
Deployment: Development of a mobile or web app interface to make the model accessible for healthcare practitioners.

---

