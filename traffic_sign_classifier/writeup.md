**Build a Traffic Sign Recognition Project**

An important part of self driving cars is the detection of traffic signs. Cars need to automatically detect
traffic signs and appropriately take actions. However, traffic sign detection is a difficult problem because of the absence of any standard whatsoever. Different countries have different traffic signs and they mean different
things. Weather plays an important role in the presence traffic signs. A very good example of this would be a place which gets heavy snow vs a place which is hot and has deserts around it. Traffic rules and signs around such places are different and hence need to be identified differently.

This was the 2nd project for the Udacity's Self Driving Car Nanodegree. The problem set provided training and testing data for traffic signs.

For the purpose of this project, Udacity made it a little easier and provided a zip file with test, validation & training data. The zip file contained 3 different pickle files for each.
* train.p - Training Data with 34799 images of each 32x32x3
* valid.p - Validation Data with 4410 images of each 32x32x3
* test.p - Testing Data with 12630 images of each 32x32x3

Each of the **Labels** are denoted by a number between 1 & 42. These labels correspond to sign names which are

| ClassId | SignName |
|---------|:---------|
|0        |Speed limit (20km/h) |
|1        |                             Speed limit (30km/h)
|2        |                             Speed limit (50km/h)
|3        |                             Speed limit (60km/h)
|4        |                             Speed limit (70km/h)
|5        |                             Speed limit (80km/h)
|6        |                      End of speed limit (80km/h)
|7        |                            Speed limit (100km/h)
|8        |                            Speed limit (120km/h)
|9        |                                       No passing
|10       |     No passing for vehicles over 3.5 metric tons
|11       |            Right-of-way at the next intersection
|12       |                                    Priority road
|13       |                                            Yield
|14       |                                             Stop
|15       |                                      No vehicles
|16       |         Vehicles over 3.5 metric tons prohibited
|17       |                                         No entry
|18       |                                  General caution
|19       |                      Dangerous curve to the left
|20       |                     Dangerous curve to the right
|21       |                                     Double curve
|22       |                                       Bumpy road
|23       |                                    Slippery road
|24       |                        Road narrows on the right
|25       |                                        Road work
|26       |                                  Traffic signals
|27       |                                      Pedestrians
|28       |                                Children crossing
|29       |                                Bicycles crossing
|30       |                               Beware of ice/snow
|31       |                            Wild animals crossing
|32       |              End of all speed and passing limits
|33       |                                 Turn right ahead
|34       |                                  Turn left ahead
|35       |                                       Ahead only
|36       |                             Go straight or right
|37       |                              Go straight or left
|38       |                                       Keep right
|39       |                                        Keep left
|40       |                             Roundabout mandatory
|41       |                                End of no passing
|42       | End of no passing by vehicles over 3.5 metric ...

[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"
[labeldistribution]: ./images/label_distribution.png "Label Distribution"
[originalsampleimg]: ./images/original_sample_img.png "Original Sample Image from Training Data"
[edgedetectedimg]: ./images/edge_detected.png "Edge Detected Image"
[flippedimg]: ./images/flip_image.png "Flipped Image"
[rotateimg]: ./images/rotate_image.png "Rotated Image"
[translateimg]: ./images/translate_image.png "Translated Image"
[0]: ./images/0.png "20 km/hr speed Limit"
[simplenn1]: ./images/simple_nn1.svg "Simple NN1 SVG"
[simplenn2]: ./images/simple_nn2.svg "Simple NN2 SVG"
[final_test_cross]: ./images/german_traffic_sign/cross.jpg "Image Cross"
[final_test_speedlimit50]: ./images/german_traffic_sign/speed_limit_50.jpg "Image Cross"
[final_test_stop]: ./images/german_traffic_sign/stop.jpg "Image Cross"
[final_test_walk]: ./images/german_traffic_sign/walk.jpg "Image Cross"
[final_test_yield]: ./images/german_traffic_sign/yield.jpg "Image Cross"
[150_epochs]: ./images/150_epochs.png "150 Epochs training on all 4 networks"

# Input Data
![alt text][labeldistribution]

The input data is not balanced across the labels which could affect the accuracy of the neural net. As can be
seen from the graph above, some labels have a larger amount of data than others.

So for this reason I augmented the input data so that it could provide a uniform set of training data.

There are multiple techniques which can be used to augment the data and there is even a
library which can be used to augment the training data. [ImgAug Link](https://github.com/aleju/imgaug)

However, I wrote my own code using a mix of opencv and skimage libraries to have a better
control over the whole process. Also there were restrictions with what kinds of image
augmentation could be performed which I wasn't sure could be handled with ImgAug. Generating such a huge dataset is computationally expensive and hence I used python's multiprocessing to speed up that process.
8 threads were launched to parallelize the operation across the 8 cores I had available on the cpu to near utilize 100% of cpu time. This brought down the time from about 20 hrs on a Macbook Pro 15" to about 7 mins on a desktop with core i7 & 32 gb memory.

### Original Image
![alt text][originalsampleimg]

## Techniques
* Random rotations between -10 and 10 degrees.
![alt text][rotateimg]
* Random translation between -10 and 10 pixels in any direction.
![alt text][translateimg]
* Random flipping horizontally or vertically or both depending on sign. There are restrictions on this since flipping a traffic sign could change it's meaning. Hence, labels have been classified on whether they can be flipped or not.
![alt text][flippedimg]
* Canny Edge detection
![alt text][edgedetectedimg]

## Example Dataset after Proprocessing
![alt text][0]

## Code
To handle this preprocessing, there are 2 classes in the code. The Data class and the Image class. The data class is responsible for handling the data loading as well as for augmenting the dataset.
```python
class Data:
    """
    Encode the different data so its easier to pass them around
    """
    def __init__(self, X_train, y_train, X_validation, y_validation, X_test,
                 y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_validation = X_validation
        self.y_validation = y_validation
        self.X_test = X_test
        self.y_test = y_test

        self.frame = pd.read_csv('signnames.csv')

    def augment_data(self, augmentation_factor):
        """
        Augment the input data with more data so that we can make all the labels
        uniform
        """
        # Find the class label which has the highest images. We will decide the
        # augmentation size based on that multipled by the augmentation factor
        pool = multiprocessing.Pool(multiprocessing.cpu_count())

        training_labels = np.concatenate((self.y_train, self.y_validation))
        training_data   = np.concatenate((self.X_train, self.X_validation))

        bincounts = np.bincount(training_labels)
        label_counts = bincounts.shape[0]

        max_label_count = np.max(bincounts)
        augmentation_data_size = max_label_count * augmentation_factor

        print_header("Summary for Training Data for Augmentation")
        print("Max Label Count: %s" % max_label_count)
        print("Augmented Data Size: %s" % augmentation_data_size)

        args = []
        for i in range(0, label_counts):
            if i in training_labels:
                args.append((i, augmentation_data_size, training_labels, training_data))

        results = pool.starmap(self._augment_data_for_class, args)
        pool.close()
        pool.join()

        features, labels = zip(*results)

        features = np.array(features)
        labels = np.array(labels)

        augmented_features = np.concatenate(features, axis=0)
        augmented_labels = np.concatenate(labels, axis=0)
        all_features = np.concatenate(np.array([training_data, augmented_features]), axis=0)
        all_labels = np.concatenate(np.array([training_labels, augmented_labels]), axis=0)

        all_features, all_labels = shuffle(all_features, all_labels)

        train = {}
        train['features'] = all_features
        train['labels'] = all_labels

        f = open('augmented/augmented.p', 'wb')
        pickle.dump(train, f, protocol=4)
```

# Network architecture

In this project I considered 4 different architectures. The reason for using 4 architectures was to see how different architectures behave under such a huge training set.  I used a GTX1080 FTI edition to do the training on my local desktop.
* Simple NN1
* Simple NN2
* LeNet
* MultiLayer NN

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

## Input Data Preprocessing
Apart from using image augmentation, I didn't use any kind of preprocessing like converting to grayscale because it didn't help with the training.

## Results with & without Training Data Augmentation

I used scikit-learn's training testing split function to generate the training as well as the validation set data.



####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the seventh cell of the ipython notebook.

## Simple NN1
![alt text][simplenn1]
| Layer         		    |     Description	        					          |
|:---------------------:|:-------------------------------------------:|
| Input         		    | 32x32x3 RGB image   							          |
| Convolution 7x7     	| 1x1 stride, VALID padding, outputs 26x26x12, 12 output Filters	|
| RELU					        |												                      |
| Flatten	          	  |                               				      |
| Fully connected       | 96 Outputs   									              |
| Batch Normalization   |                                             |
| Logits Softmax        | 43 Outputs   									              |

## Simple NN2
![alt text][simplenn2]

| Layer         		    |     Description	        					          |
|:---------------------:|:-------------------------------------------:|
| Input         		    | 32x32x3 RGB image   							          |
| Convolution 5x5     	| 1x1 stride, SAME padding, outputs 26x26x12, 12 output Filters	|
| RELU					        |												                      |
| Maxpool & Dropout     |	2x2 maxpool, 2x2 stride, dropout=0.2, VALID |
| Convolution 7x7     	| 1x1 stride, SAME padding, 24 output Filters	|
| RELU					        |												                      |
| Maxpool & Dropout     |	2x2 maxpool, 2x2 stride, dropout=0.4, VALID |
| Flatten	          	  |                               				      |
| Fully connected       | 96 Outputs   									              |
| Batch Normalization   |                                             |
| Logits Softmax        | 43 Outputs   									              |

## Lenet

## AlexNet
![alt text][alexnet]

My final model consisted of the following layers:

| Layer         		    |     Description	        					          |
|:---------------------:|:-------------------------------------------:|
| Input         		    | 32x32x3 RGB image   							          |
| Convolution 3x3     	| 1x1 stride, SAME padding, 12 output Filters	|
| RELU					        |												                      |
| Convolution 5x5     	| 1x1 stride, VALID padding, Outputs 28x28x24, 24 output Filters	|
| RELU					        |												                      |
| Convolution 5x5     	| 1x1 stride, VALID padding, Outputs 24x24x48, 48 output Filters	|
| RELU					        |												                      |
| Convolution 9x9     	| 1x1 stride, SAME padding, Outputs 16x16x96, 96 output Filters	|
| RELU					        |												                      |
| Convolution 3x3     	| 1x1 stride, SAME padding, Outputs 16x16x192, 192 output Filters	|
| RELU					        |												                      |
| Maxpool               |	2x2 maxpool, 2x2 stride, SAME, Outputs 16x16x384               |
| Convolution 11x11   	| 1x1 stride, SAME padding, Outputs 8x8x384, 384 output Filters	|
| RELU					        |												                      |
| Maxpool               |	2x2 maxpool, 2x2 stride, SAME, Outputs 4x4x384               |
| Flatten	          	  | 6144 Outputs                             				      |
| Fully connected       | 3072 Outputs   									              |
| Batch Normalization   |                                             |
| Fully connected       | 1536 Outputs   									              |
| Batch Normalization   |                                             |
| Fully connected       | 768 Outputs   									              |
| Batch Normalization   |                                             |
| Fully connected       | 384 Outputs   									              |
| Batch Normalization   |                                             |
| Fully connected       | 192 Outputs   									              |
| Batch Normalization   |                                             |
| Fully connected       | 96 Outputs   									              |
| Batch Normalization   |                                             |
| Logits Softmax        | 43 Outputs   									              |



#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

For all the networks, AdamOptimizer was used since that is the most popular optimizer and gave decent results.
Along with this, I also added L2 Loss to the operation. This improved the validation accuracy by about 2%

#### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ?
* test set accuracy of ?

The graphs below show the performance of 4 networks as well as their loss functions.
![alt text][150_epochs]

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?


### Real Dataset

Here are five German traffic signs that I found on the web:

![alt text][final_test_walk] ![alt text][final_test_stop] ![alt text][final_test_yield]![alt text][final_test_cross] ![alt text][final_test_speedlimit50]

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Stop Sign      		| Stop sign   									|
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .60         			| Stop sign   									|
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ...

# Links & References
- Block diagrams built with http://interactive.blockdiag.com/

# Appendix

- Simple NN1
blockdiag {
  Conv1-7x7x12 -> BatchNormalization -> Activation -> Flatten -> FullyConnected-96 -> Logits;
}

- Simple NN2
blockdiag {
  Conv1-5x5x12 -> BatchNormalization1 -> Activation1 -> Maxpool1 -> Dropout-0.2 -> Conv2-7x7x24 -> BatchNormalization2 -> Activation2 -> Maxpool2 -> Dropout-0.4 -> Flatten -> FullyConnected-96 -> Logits;
}

- Lenet:
