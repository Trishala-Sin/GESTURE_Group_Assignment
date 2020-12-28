# Assignment : Gesture Recognition Case Study:

#### Gesture Recognition Case Study :
Objective: 
To develop a cool feature in the smart-TV for Home electronics that can recognize five different gestures performed by the user which will help users control the TV without using a remote.

Each gesture corresponds to a specific command. Following are the possible gesture classes: 
1.	Thumbs up   : Increase the volume (index - 4)
2.	Thumbs down : Decrease the volume (index - 3)
3.	Left swipe : 'Jump' backwards 10 seconds ( index - 0)
4.	Right swipe : 'Jump' forward 10 seconds ( index - 1)
5.	Stop	    : Pause the movie (index - 2)
Input :  Training  data includes short videos 30 frames (images) for each gestures and labeled data as mentioned above.
Image specification: 
There are 2 types of images available in the training dataset 
•	(360X360) pixels in height, width
•	(120X120) height, width 
Possible step: Make the images homogenous in size.  (Image resizing and Cropping h)

Resize and cropping to: 120X120 (image in standard format - 28X28, 32X32, 64X64, 128X128) 
Suggestion - Having good GPU can result in better image processing hence we can choose accordingly.

Approach to case study: 
•	Two types of architectures are used commonly. One is the standard CNN + RNN architecture in which we can pass the images of a video through a CNN architecture which extracts a feature vector for each image, and then pass the sequence of these feature vectors through an RNN.
RNN use lots of parameters (due to their recurrent nature) as compared to CNNs,
So we have to avoid creating lots of unnecessary parameters.

•	Other one is natural extension of CNNs - a 3D convolution network. In this case, the input to a 3D convolution is a video (which is a sequence of 30 RGB images) .Assuming that the shape of each image is 100x100x3, for example, the video becomes a 4-D tensor of shape 100x100x3x30 which can be written as (100x100x30)x3 where 3 is the number of channels.

•	Using generators we create batches of images , here we are taking size of 10 initially, Others are
        Frame = 20, rows = 120, cols = 120 , Channel_color = 3, num_classes = 5. For Image processing 
We also apply Normalization on the image using Open-CV Library methods.  

Experiment Number 1 : Model-1 (vanilla) Conv3D

Created Model using vanilla CNN architecture with input layer(16Neurons) , 2 hidden layer (1st layer : 64 Neurons 2nd layer : 128 )
Fully connected and output (dense layer).
Filter used (3X3X3).
Result : Throws Generator error.
Model not trainable as a lot of parameters.

On the second run, however 
It is running extremely slow for 10 epochs.

Accuracy : low ( 0.20)

Decision + Explanation: Try to hyper tune the network and apply layers and other possible ways to get model performance and train and validation accuracy.


Experiment Number 2 : Model-2 Reducing Filter (2,2,2) & Added Hyper Tuning Parameters (Conv3D)

Reducing filter size to (2X2X2) and optimizer to ‘adam’. Using Batch Normalization after each layer. Added dropout after each hidden layer.

Result: 
Total parameters are : 3,301,941
Trainable parameters: 3,301,461
Accuracy: 0.2612
Valid accuracy: 0.2300 

Decision + Explanation : Reduce the size of the image and filter size did not produce and good accuracy hence getting back to same filter. We have also tried to hyper tuning of parameters through dropout and Batch normalization.



Experiment Number 3 : Model -3 changing optimizer to "SGD" & Added L2 Regularization(Conv3D)

Resuming filter size to 3X3X3 and optimizer to ‘SGD’ and applying L2 regularization.

Result : Total parameters: 3,507,141
Trainable parameters: 3,506,661
Accuracy: 0.6672
Valid accuracy : 0.6100

Decision + Explanation : Increase the amount of trainable data/ reduce the filter size. We assume that using more hyper tuning will increase the accuracy hence we are trying different combinations of neural layers and trying to reduce over fitting while increasing the accuracy decreasing loss.


Experiment Number 4 : Model -4 Adding More Layers (Conv3D)

Adding more hidden layers  to increase the accuracy 
Result : Total parameters: 2,481,701
Trainable parameters: 2,480,773
Accuracy : 0.7687
Valid accuracy : 0.5700

Decision + Explanation : We Added more hidden layers but it didn’t improved accuracy any further. We just doubled each of the hidden layer and performed max pooling on the output to produce better result and using dropout we are managing the over fitting of training data. 

Experiment Number 5 : Model – 5 (CNN + SIMPLE RNN NETWORK)

Using 2nd method of (CNN+RNN) architecture 
Result : Total parameters: 1,836,901
Trainable parameters: 1,836,421
Accuracy : 85.07
Valid accuracy : 72.00

Decision + Explanation : We are using first approach of using a combination of (CNN) for image processing and feature extraction and passing it to RNN for sequential feature analysis and producing output in one of the 5 category of gestures on the basis of RNN output.

Experiment Number 6 : Model – 6 (CNN+RNN)
Added more neurons and dropout as hyper tuning in (CNN+RNN) for better accuracy 

Result : Total parameters: 3,870,469
Trainable parameters: 3,869,893
Accuracy : 84.93
Valid Accuracy: 68.00 

Decision + Explanation : We observed that vanilla architecture for CNN+RNN requires more hyper tuning due to over fitting hence we tried to add dropout and increase neurons to gather as much as input from image processing.

Experiment Number 7 : Model – 7 (CNN + LSTM)
Added 3 hidden layers with batch normalization. (CNN + LSTM 

Result : Total parameters: 3,392,869
Trainable parameters: 3,392,389
Accuracy : 84.33
Valid Accuracy: 67.00 

Decision + Explanation : We assume that LSTM would provide better results than RNN as LSTM can handle over fitting which can be seen in the previous results and it helps in better learning of model as it stores the relevant information plus the current state. 

Experiment Number 8 : Model – 8 (CNN + LSTM)
CNN + LSTM Network(Added more cells, Removed Hidden Layer, Added Dropout)

Result : Total parameters: 60,392,389
Trainable parameters: 60,391,941
Accuracy : 78.33
Valid Accuracy: 61.00 

Decision + Explanation : We tried to manage the output using LSTM and reduce the layer and added dropout for managing over fitting. But this resulted in increase in the number of parameters.

Experiment Number 9 : Model – 9 (CNN +GRU)
CNN + GRU Network for better speed in computation.


Result : Total parameters: 5,179,749
Trainable parameters: 5,179,269
Accuracy : 84.33
Valid Accuracy: 67.00

Decision + Explanation : We also experimented with CNN+GRU architecture in order to check the accuracy over traditional RNN and LSTM models as GRU provides better and fast computation as compared to others. But we observed major gap between accuracy and validation accuracy. 

Experiment Number 10 : MODEL-10 CNN +GRU
 
Result : Total parameters: 3,121,541
Trainable parameters: 3,120,645
Accuracy : 77.33
Valid Accuracy: 48.00


Decision + Explanation : We tried a few variations in previous model to increase validation accuracy and tried to increase parameters for better learning while controlling the output with dropout. we saw this is not achieved by changes we made in the model. 

Final Model: Model -3 changing optimizer to "SGD" & Added L2 Regularization(Conv3D)

Explaination : We Found that over all  every other model other than model-3 have increased number of parameters and while other provides good train accuracy but when compared to validation accuracy they all seem to over fit training data. Hence Model -3 shows closest relationship between train and validation accuracy with less parameters we will choose model -3 for our case study.

