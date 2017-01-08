# Diabetic Retinopathy Detection Challenge (Kaggle)

### This repository contains tensorflow code for the kaggle challenge: Diabetic Retinopathy Detection.

**@Author: Aman Rana**
</br>





</br>

The datset details are as follows:

* **data:** This folder contains the following folders
 * **train:** This folder contains the training images. There are 300 RGB images of the original dataset.
 * **test:** This folder contains the testing images. There are 180 RGB images of the original dataset.
 * **trainLabels.csv:** This file cotains labels for all the training images.

The code for this project is contined within the two files:

1. **run.py:** This python file is the starting point of the code. It processes the images and uses a **custom generator** to create batches of any desired size and calls the model.py file for training the model
2. **model.py:** This file contains code for creation of the **Convolutional Neural Network (CNN)** model and training based on the training batch passed by run.py file.

**Note:** Since my laptop is not very powerful and I have installed tensorflow usinf conda (no GPU support via anaconda), I have used a small subset of the original dataset for training. I am uploading an even smaller subset of the original kaggle dataset. The original dataset can be downloaded [here](https://www.kaggle.com/c/diabetic-retinopathy-detection)
