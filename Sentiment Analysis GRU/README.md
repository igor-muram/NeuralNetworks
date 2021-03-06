# Sentiment Analysis GRU

Analysis of the emotional tone of texts from the social network Twitter using a recurrent neural network GRU.

## Parameters

* Binary cross-entropy is used as the loss function.
* Accuracy, i.e. the proportion of correctly recognized sentences, is used as a metric for the quality of neural network performance.
* The database of automatically collected texts from the English segment of the social network Twitter was used for training and testing the neural network.
* The optimizer Adam is used to improve the quality of training.
* To achieve the best quality of neural network performance the hyperparameters of the model are selected on cross-validation.

## Technologies

Libraries and tools used to make up the network:

* Python
* TensorFlow
* Keras
* Keras Tuner (select the hyperparameters of the model on cross validation)

## Structure

Structure of a neural network:

* Embedding layer with 128 outputs.
* GRU layer of 128 neurons.
* GRU layer of 32 neurons.
* GRU layer with 32 neurons.
* Layer of full-connected network with 2 neurons and softmax activation function, the output of which will be a vector of two elements whose values will be in the range from 0 to 1. The value of the first element will show the probability that the phrase has a positive emotional tone, and the value of the second element will show the probability that the phrase has a negative emotional tone.

## Plots

![image](https://user-images.githubusercontent.com/54866075/126515539-92a00b40-508d-405a-bd11-1c884d051284.png)      ![image](https://user-images.githubusercontent.com/54866075/126515553-412b7528-d30c-46bc-ae4f-1933664ced4f.png)

## Loss and accuracy

![image](https://user-images.githubusercontent.com/54866075/126515507-d1f8f7ae-5529-41a8-8bdb-475ab3bfafb1.png)

## An example of the result of a neural network

![image](https://user-images.githubusercontent.com/54866075/126514154-f9629967-3066-44a2-a4b2-e37b5a2d9811.png)      ![image](https://user-images.githubusercontent.com/54866075/126515581-4c39e297-2312-4f57-a1e7-5d646682714e.png)
