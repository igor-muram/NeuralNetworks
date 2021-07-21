# Sentiment Analysis LSTM

Analysis of the emotional tone of texts from the social network Twitter using a recurrent neural network LSTM.

Binary cross-entropy is used as the loss function.<br>
Accuracy, i.e. the proportion of correctly recognized sentences, is used as a metric for the quality of neural network performance.<br>
The database of automatically collected texts from the English segment of the social network Twitter was used for training and testing the neural network.<br>
The optimizer Adam is used to improve the quality of training.<br> 
To achieve the best quality of neural network performance the hyperparameters of the model are selected on cross-validation.

<p>Libraries and tools used to make up the network:</p>

<ul>
	  <li>Python</li>
    <li>TensorFlow</li>
  	<li>Keras</li>
  	<li>Keras Tuner (select the hyperparameters of the model on cross validation)</li>
</ul>

Structure of a neural network:<br>
<ul>
	  <li>Embedding layer with 96 outputs.</li>
    <li>LSTM layer of 96 neurons.</li>
    <li>LSTM layer of 48 neurons.</li>
    <li>LSTM layer with 48 neurons.</li>
    <li>Layer of full-connected network with 2 neurons and softmax activation function, the output of which will be a vector of two elements, the values of which will be in the range from 0 to 1. The value of the first element will show the probability that the phrase has a positive emotional tone, and the value of the second element will show the probability that the phrase has a negative emotional tone.</li>
</ul>
