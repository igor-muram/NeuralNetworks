# Transfer Learning

Image recognition using Transfer Learning technology and Facebook's PyTorch framework for machine learning in Python.<br>
The images for the datasets were downloaded automatically from ImageNet using a Python script.

## Parameters

<ul>
<li>Cross-entropy is used as a loss function.</li>
<li>Accuracy, i.e. the proportion of correctly recognized instances, is used as a metric of neural network performance.</li>
<li>The resnet18 neural network, which is a convolutional neural network with 18 layers, is used as the pre-trained part of the neural network.</li>
<li>To increase the size of the dataset we use data augmentation during training, in particular, random horizontal rotation and random cropping of images.</li>
<li>For testing the neural network images in their original form are used.</li>
<li>The maximum number of epochs in training is 25.</li>
<li>The stochastic gradient descent with rate coefficient of learning equal to 0.001 and torque equal to 0.9 has been chosen for training.</li>
</ul>
	
## Some theory

<ul>
	<li>The presence of torque allows not to stay in the local minimum during gradient descent.</li>
	<li>The stochastic gradient descent differs from the usual one as the gradient of the optimized function at each step is calculated not as the sum of gradients from each element of the sample but as the gradient from one randomly selected element.</li>
	<li>In case of Fine Tuning all weights of the neural network are changed. Instead of random initialization, the network is initialized with a pre-trained network, such as the one trained on the Imagenet dataset.</li>
	<li>With Fixed Feature Extractor, only the weights of the last layer we added are changed, while the remaining weights of the pre-trained part of the neural network remain unchanged.</li>
</ul>	
	
## Technologies

<p>Libraries and tools used to make up the network:</p>

<ul>
	<li>Python</li>
    	<li>PyTorch</li>
</ul>

## Approaches

<p>Approaches used in development:</p>

<ul>
	<li>Fixed Feature Extractor</li>
    	<li>Fine Tuning</li>
</ul>
