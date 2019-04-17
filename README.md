# Udacity's Data Scientist Nanodegree Project 02: Developing an AI application

--------------------------------------
1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## 1. Installation <a name="installation"></a>

- This code was created by using Python versions 3.*.
- Following libraries have to be imported:

* pandas
* numpy
* matplotlib
* torch
* torchvision
* PIL
* json
* collections

- copy repository: git clone https://github.com/oliverkroening/Udacity_DSND_Project02


## 2. Project Motivation <a name="motivation"></a>
For this Udacity project, I created a code to train an image classifier for recognizing different species of flowers. This could by applied to smart phones to classify flowers in the environment. I used a dataset of 102 flower categories, which can be found [here](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html).

The project is devided into three parts:
- Load and preprocess the image dataset
- Train the image classifier on the dataset
- predict image content by using image classifier

Afterwards,  terminal application was coded to train a model and to predict the flower within a photo

## 3. File Descriptions <a name="files"></a>  
* Image Classifier Project.ipynb: Jupyter Notebook including the whole process to fulfill the tasks of this project.
* Image Classifier Project.html: the corresponding HTML-file
* LICENSE: MIT-License-file for Udacity
* cat_to_name.json: JSON-file containing the names for each category to assign the flowers
* image_classifier_utils.py: supporting function for image classifier terminal application
* predict.py: terminal application for making predictions
* train.py: terminal application for training the model
* workspace-utils.py: supporting function from Udacity

## 4. Results <a name="results"></a>
I created a dense neural network based on three hidden layers to predict the category of flowers by reading a picture of that flower. The model achieved an accuracy score of 81.5% on the test dataset.

Furthermore, a terminal application was created to form and train a new neural network based on pretrained VGG networks.

## 5. Licensing, Authors, Acknowledgements<a name="licensing"></a>
All data was obtained from [www.robots.ox.ac.uk](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) as well as Udacity, thus, I must give credit to them. Other references are cited in the Jupyter notebook.






