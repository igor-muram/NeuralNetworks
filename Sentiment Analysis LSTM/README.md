# Sentiment Analysis LSTM

Analysis of the emotional tone of texts from the social network Twitter using a recurrent neural network LSTM.

## Parameters

* Binary cross-entropy is used as the loss function.
* Accuracy, i.e. the proportion of correctly recognized sentences, is used as a metric for the quality of neural network performance.
* The database of automatically collected texts from the English segment of the social network Twitter was used for training and testing the neural network.
* The optimizer Adam is used to improve the quality of training.
* To achieve the best quality of neural network performance the hyperparameters of the model are selected on cross-validation.

## Techonogies

Libraries and tools used to make up the network:

* Python
* TensorFlow
* Keras
* Keras Tuner (select the hyperparameters of the model on cross validation)

## Structure

Structure of a neural network:

* Embedding layer with 96 outputs.
* LSTM layer of 96 neurons.
* LSTM layer of 48 neurons.
* LSTM layer with 48 neurons.
* Layer of full-connected network with 2 neurons and softmax activation function, the output of which will be a vector of two elements, the values of which will be in the range from 0 to 1. The value of the first element will show the probability that the phrase has a positive emotional tone, and the value of the second element will show the probability that the phrase has a negative emotional tone.

## Plots

![image](https://user-images.githubusercontent.com/54866075/126515045-ce6d0518-e32e-43fa-bb5a-4d2579f086cf.png)      ![image](https://user-images.githubusercontent.com/54866075/126515061-83fcf78a-1297-458c-a73c-5f13ad285cc2.png)

## Loss and accuracy

![image](https://user-images.githubusercontent.com/54866075/126515029-2fb2696d-430e-43c4-bff2-976c80d3bb8b.png)

## An example of the result of a neural network

![image](https://user-images.githubusercontent.com/54866075/126514154-f9629967-3066-44a2-a4b2-e37b5a2d9811.png)      ![image](https://user-images.githubusercontent.com/54866075/126515183-44411f52-5304-4c3d-be56-387a501ec431.png)
