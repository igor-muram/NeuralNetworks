# Sentiment Analysis AvgPooling

Analysis of the emotional tone of texts from the social network Twitter using a neural network with the calculation of the average value of the Embedding layer.

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
	<li>Embedding layer.</li>
  	<li>GlobalAveragePooling1D operation for weights obtained from the Embedding layer.</li>
  	<li>Layer of full-connected network with 1 neuron and sigmoidal activation function, the output of which will be a value in the range from 0 to 1. This value will show how high the positive tone of the text is. If the obtained value is less than 0.5, we will consider the text negative, otherwise we will consider it positive.</li>
</ul>

Plots:

![image](https://user-images.githubusercontent.com/54866075/126513310-c13e72a8-cc18-447a-821c-9151cb6569f0.png)      ![image](https://user-images.githubusercontent.com/54866075/126513376-ecbd2660-c5f6-4125-a3eb-8b1a3f02986d.png)

Loss and accuracy:<br>

![image](https://user-images.githubusercontent.com/54866075/126513807-001a5d0f-2032-4174-926b-b199dfc7090e.png)

An example of the result of a neural network:

![image](https://user-images.githubusercontent.com/54866075/126514154-f9629967-3066-44a2-a4b2-e37b5a2d9811.png)      ![image](https://user-images.githubusercontent.com/54866075/126514186-d424cba3-2350-42c1-aaec-76cc59a72bb4.png)
