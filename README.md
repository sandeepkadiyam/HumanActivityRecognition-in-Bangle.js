# HAR-in-Bangle.js

## Abstract:
Recent studies in Human Activity Recognition(HAR) have shown that  HAR on edge devices such as wearables has lower latency and better performance when inferred locally than on an external server. One of the conventional methods to achieve HAR on edge devices is by using deep learning algorithms. However, the edge devices are memory-constrained and have limited processing power, making it challenging to deploy state-of-the-art deep learning models. In this project, we develop and deploy a deep learning model for HAR on the bangle.js smartwatch. To transfer the model to the smartwatch, we optimize it using TensorFlow lite, reducing its size without losing much accuracy. As a result, we reduced the model's size by 96.87\%, i.e., from 210.547Kb to 6.578Kb, by losing only 0.2\% accuracy on a custom dataset and 0.9\% accuracy on a HAPT dataset.

## Steps involved in the project:
The project is structured in five steps. The first step of the project is to acquire and pre-process the accelerometer data. The next step is to develop and train a Convolutional Neural Network. Then, the third step is to optimize and deploy the model on bangle.js. After deployment, inference will be performed in real-time using the deployed model.

## Scripts and files:
1) A step-by-step process of model development and model deployment was implemented in [this](HumanActivityRecognition.ipnyb) python notebook.
2) The script to perfrom inference on bangle.js can be found [here](Utils/prediction.js).
