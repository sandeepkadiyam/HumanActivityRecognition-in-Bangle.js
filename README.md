# HAR-in-Bangle.js

## Abstract:
Recent studies in Human Activity Recognition(HAR) have shown that  HAR on edge devices such as wearables has lower latency and better performance when inferred locally than on an external server. One of the conventional methods to achieve HAR on edge devices is by using deep learning algorithms. However, the edge devices are memory-constrained and have limited processing power, making it challenging to deploy state-of-the-art deep learning models. In this project, we develop and deploy a deep learning model for HAR on the bangle.js smartwatch. To transfer the model to the smartwatch, we optimize it using TensorFlow lite, reducing its size without losing much accuracy. As a result, we reduced the model's size by 96.87\%, i.e., from 210.547Kb to 6.578Kb, by losing only 0.2\% accuracy on a custom dataset and 0.9\% accuracy on a HAPT dataset.

## Structure of the project:
The project is structured in five steps. The first step of the project is to acquire and pre-process the accelerometer data\ref{datapre-processing}. The next step is to develop\ref{definemodel} and train\ref{modeltraining} a Convolutional Neural Network. Then, the third step is to optimize\ref{optimization} and deploy\ref{deployment} the model on bangle.js. After deployment, inference\ref{inference} will be performed in real-time using the deployed model. The final stage of the project is to evaluate\ref{evaluation} the model's performance using standard evaluation metrics such as precision, recall, and f1-score\cite{Rahul2019}.

## Files in the Repository:
[HumanActivityRecognition.ipynb](HumanActivityRecognition.ipynb) : A step-by-step implementation of model development and model deployment.  
[prediction.js](prediction.js) : The script to perform inference on bangle.js.  
[Evaluation.ipynb](Evaluation.ipynb) : The jupyter notebook to evaluate the model using K-fold cross-validation.  
[LoadDataset.py](Utils/LoadDataset.py) : The script to load and split the dataset into train, test and validation datasets.  
[Optimization.py](Utils/Optimization.py) : The script to convert the keras model to the TensorFlow Lite model.

## Results
The model's performance was evaluated before and after optimization using a custom dataset and a public HAPT dataset from the UCI machine learning repository. All the results were obtained using the k-fold cross-validation method.  

**Model's Performance on the custom dataset:** The custom dataset was collected from three subjects performing four activities, running, walking, sitting, and laying, wearing a bangle.js on the left-hand wrist. The data was sampled at 50Hz with a sensitivity of +/-8g.


We developed and deployed a deep learning model for Human activity recognition on bangle.js. For the model development and model deployment on bangle.js, a detailed python jupyter notebook was created and can be found [here](HumanActivityRecognition.ipynb). We used the deployed model to perform on-device inference, and the script to perform inference on bangle.js can be found [here](prediction.js).
