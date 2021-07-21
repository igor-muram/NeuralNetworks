# Sentiment Analysis GRU

Analysis of the emotional tone of texts from the social network Twitter using a recurrent neural network GRU.

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
	  <li>Embedding layer with 128 outputs.</li>
    <li>GRU layer of 128 neurons.</li>
    <li>GRU layer of 32 neurons.</li>
    <li>GRU layer with 32 neurons.</li>
    <li>Layer of full-connected network with 2 neurons and softmax activation function, the output of which will be a vector of two elements whose values will be in the range from 0 to 1. The value of the first element will show the probability that the phrase has a positive emotional tone, and the value of the second element will show the probability that the phrase has a negative emotional tone.</li>
</li>
</ul>
