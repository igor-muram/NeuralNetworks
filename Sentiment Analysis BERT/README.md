# Sentiment Analysis BERT

Analysis of the emotional tone of texts from the social network Twitter using a bidirectional transformer BERT.

## Parameters

<ul>
	<li>Binary cross-entropy is used as the loss function.</li>
	<li>Accuracy, i.e. the proportion of correctly recognized sentences, is used as a metric for the quality of neural network performance.</li>
	<li>The database of automatically collected texts from the English segment of the social network Twitter was used for training and testing the neural network.</li>
	<li>The optimizer Adam is used to improve the quality of training.</li>
	<li>To achieve the best quality of neural network performance the hyperparameters of the model are selected on cross-validation.</li>
</ul>

## Technologies

<p>Libraries and tools used to make up the network:</p>

<ul>
	<li>Python</li>
    	<li>TensorFlow</li>
  	<li>Keras</li>
  	<li>Keras Tuner (select the hyperparameters of the model on cross validation)</li>
</ul>

## Structure

Structure of a neural network:

<ul>
	<li>Input layer.</li>
    	<li>The Keras layer for preparing data from the input layer.</li>
    	<li>The encoder input layer, obtained by passing data from the first input layer through the Keras layer.</li>
    	<li>Encoder layer.</li>
    	<li>Output layer, resulting from passing data from the input layer for encoder through the encoder layer.</li>
    	<li>A "merged output" layer to improve classification results.</li>
    	<li>Dropout layer with 0.1 neuron dropout probability.</li>
    	<li>Layer of full-connected network with 1 neuron, the output of which will be a value in the range from 0 to 1. This value will show how high the positive tone of the text is. If the obtained value is less than 0.5, we will consider the text negative, otherwise it will be positive.</li>
</ul>


## Plots

![image](https://user-images.githubusercontent.com/54866075/126515847-d09486e7-9234-4210-a442-790f12d94d27.png)

## Loss and accuracy

![image](https://user-images.githubusercontent.com/54866075/126515923-5e9c8788-ee6f-46b1-b7d2-1b697839d4ae.png)

## An example of the result of a neural network

![image](https://user-images.githubusercontent.com/54866075/126515872-fbee9a65-b44d-466c-a6dd-aa2c141cc266.png)      ![image](https://user-images.githubusercontent.com/54866075/126515884-0f62b101-9557-4f93-a967-f8049530ff39.png)
