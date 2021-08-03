# HAR-in-Bangle.js

## Abstract
Human Activity Recognition(HAR) on edge devices such as wearables is proven to have lower latency and better performance when compared to the activity recognition performed on a cloud server. Usually, HAR using deep learning algorithms is claimed to have a high performance and accuracy. However, the edge devices are resource-constrained and cannot support high computation, which makes it impossible to deploy the state of art deep learning models on to them. In this project we used the open-source deep learning framework developed by Google, TensorFlow lite to convert deep learning models into edge device compatible models and perform on device inference. This project mainly focuses on achieving Human activity recognition in the Bangle.JS smartwatches. Our approach uses the accelerometer data from the Bangle.js to achieve HAR. This report briefly discusses about organizing the sensor data, developing the deep learning model, deployement of the model on to bangle.js and on-device inference.

## Introduction
One of most challenging and dynamic research in recent times is Human Activity Recognition(HAR) on edge devices. Human Activity Recognition is the study of recognizing human activities based on series of data or observations captured by sensors which tend to vary based on the activity performed by the subject. Generally, deep learning models are used to tackle HAR problems. The deep learning approach needs a very little data feature engineering when compared to traditional machine learning methods and yet providing better results. The main focus of this project is to be able to infer human activities on Bangle.js smartwatch. Bangle.js is a completely hackable smartwatch powered by espruino. The Bangle.js is equipped with variety of sensors. The set of sensors include a 3 Axis Accelerometer (with Pedometer and Tap detect). The accelerometer produce a continuous data stream which can be interpreted as a time series data. This data is used as an input for the human activity classifier. 

The main challenge is to successfully deploy the deep learning model on to Bangle.js. Bangle.js comes with 64KB of ram and 4MB of flash storage which does not leave enough room for a state of art deep learning model. Bangle.js comes with tensorflow lite built in which allows us to run memory optimized deep learning models. The optimized deep learning models are memory efficient, have lower latency and above all, doesn't lose accuracy. The further sections explain all the steps necessary from capturing accelerometer data to infering Human activities on bangle.js in detail.

## Methodology

### Accelerometer data
Now a days, every mobile device have accelerometer embedded in it. The accelerometer determines the orientation and the movement of the wearer's arm. The data aquisition is majorly effected by two factors, sampling and sensitivity.
Sampling is how often does the accelerometer gathers data and sensitivity is how effectively does the watch recognizes the tiniest vibration that goes through the wearer's arm.
Bangle.js operating system limits the sampling of the accelerometer to a maximum of 100 hz. We can choose between four different sampling rates for the watch to work on. This can be chosen by setting appropriate bits in the accelerometer write register.
The maximum sensitivity that can be achieved by Bangle.js is +-8g. We can chose between three different sensitivity rates for the atch to work on. All the data recorded using the watch for this project is recorded with 50 hz sampling rate and +-8g sensitivity.
* organizing the data
* window sampling method.
* startified splitting.


### Deep Learning model
* CNN
* why light weight model?
* How CNN works?
* Explain the model used briefly?
* what loss function and optimizer are used in the model?
* 
### Deployment
* Tf lite for microcontrollers.
* quantization?
* conversion?
* 
### Inference
* How to access model?
* How to set input
* how to infer
* how to match the output to class names?
