# Diabetic_Retinopathy_Detection

### This repository contains tensorflow code for the kaggle challenge: Diabetic Retinopathy Detection.

**@Author: Aman Rana**
</br>





</br>

The datset details are as follows:

* **data:** This folder contains the following folders
 * **train:** This folder contains the training images. There are a little over 35K RGB images in total.
 * **test:** This folder contains the testing images. There are around 12K RGB images in total.
 * **trainLabels.csv:** This file cotains labels for all the training images.

The code for this project is contined within the two files:

1. **run.py:** This python file is the starting point of the code. It processes the images and uses a _**custon generator**_ to create batches of any desired size and calls the model.py file for training the model
2. **model.py:** This file contains code for creation of the **Convolutional Neural Network (CNN)** model and training based on the training batch passed by run.py file.

