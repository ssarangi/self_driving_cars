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

| idx | network name | epochs | validation accuracy| loss | test accuracy |
|--------------|------------|---------------------|------|-------------------|-------|
| 0            | simple_nn1 | 1                   | 91   | 2.52185201644897  | 0.886 |
| 1            | simple_nn1 | 2                   | 94.8 | 1.55742692947388  | 0.886 |
| 2            | simple_nn1 | 3                   | 94.9 | 1.12162351608276  | 0.886 |
| 3            | simple_nn1 | 4                   | 95.9 | 0.860485732555389 | 0.886 |
| 4            | simple_nn1 | 5                   | 95.7 | 0.700304865837097 | 0.886 |
| 5            | simple_nn1 | 6                   | 96.5 | 0.592534780502319 | 0.886 |
| 6            | simple_nn1 | 7                   | 96.2 | 0.567364394664764 | 0.886 |
| 7            | simple_nn1 | 8                   | 96.3 | 0.564719319343567 | 0.886 |
| 8            | simple_nn1 | 9                   | 97.2 | 0.529635608196258 | 0.886 |
| 9            | simple_nn1 | 10                  | 96.5 | 0.45400869846344  | 0.886 |
| 10           | simple_nn1 | 11                  | 96.9 | 0.427338153123856 | 0.886 |
| 11           | simple_nn1 | 12                  | 96.7 | 0.453817665576935 | 0.886 |
| 12           | simple_nn1 | 13                  | 96.7 | 0.437843978404999 | 0.886 |
| 13           | simple_nn1 | 14                  | 97   | 0.440367996692658 | 0.886 |
| 14           | simple_nn1 | 15                  | 96.9 | 0.543636620044708 | 0.886 |
| 15           | simple_nn1 | 16                  | 96.7 | 0.443966567516327 | 0.886 |
| 16           | simple_nn1 | 17                  | 96.2 | 0.39118891954422  | 0.886 |
| 17           | simple_nn1 | 18                  | 96.8 | 0.388409405946732 | 0.886 |
| 18           | simple_nn1 | 19                  | 96.7 | 0.372857511043549 | 0.886 |
| 19           | simple_nn1 | 20                  | 96.6 | 0.403676927089691 | 0.886 |
| 20           | simple_nn1 | 21                  | 96.2 | 0.4980109333992   | 0.886 |
| 21           | simple_nn1 | 22                  | 96.9 | 0.436672955751419 | 0.886 |
| 22           | simple_nn1 | 23                  | 96.6 | 0.413485020399094 | 0.886 |
| 23           | simple_nn1 | 24                  | 96.3 | 0.388778507709503 | 0.886 |
| 24           | simple_nn1 | 25                  | 96.7 | 0.33022665977478  | 0.886 |
| 25           | simple_nn1 | 26                  | 96.6 | 0.37101536989212  | 0.886 |
| 26           | simple_nn1 | 27                  | 96.9 | 0.457292646169663 | 0.886 |
| 27           | simple_nn1 | 28                  | 96.5 | 0.361608326435089 | 0.886 |
| 28           | simple_nn1 | 29                  | 96.7 | 0.440926492214203 | 0.886 |
| 29           | simple_nn1 | 30                  | 96.1 | 0.397851824760437 | 0.886 |
| 30           | simple_nn1 | 31                  | 97   | 0.359396874904633 | 0.886 |
| 31           | simple_nn1 | 32                  | 96.6 | 0.388941466808319 | 0.886 |
| 32           | simple_nn1 | 33                  | 96.7 | 0.608758091926575 | 0.886 |
| 33           | simple_nn1 | 34                  | 96.8 | 0.370195418596268 | 0.886 |
| 34           | simple_nn1 | 35                  | 96.5 | 0.42000424861908  | 0.886 |
| 35           | simple_nn1 | 36                  | 96.9 | 0.420544624328613 | 0.886 |
| 36           | simple_nn1 | 37                  | 96.5 | 0.434133231639862 | 0.886 |
| 37           | simple_nn1 | 38                  | 95.8 | 0.440094113349915 | 0.886 |
| 38           | simple_nn1 | 39                  | 97.4 | 0.359337568283081 | 0.886 |
| 39           | simple_nn1 | 40                  | 96.7 | 0.439890146255493 | 0.886 |
| 40           | simple_nn1 | 41                  | 96.9 | 0.462374836206436 | 0.886 |
| 41           | simple_nn1 | 42                  | 96.9 | 0.367310523986816 | 0.886 |
| 42           | simple_nn1 | 43                  | 96.9 | 0.370069921016693 | 0.886 |
| 43           | simple_nn1 | 44                  | 97.1 | 0.418380856513977 | 0.886 |
| 44           | simple_nn1 | 45                  | 96.8 | 0.395580410957336 | 0.886 |
| 45           | simple_nn1 | 46                  | 96.8 | 0.379706233739853 | 0.886 |
| 46           | simple_nn1 | 47                  | 97.2 | 0.362332224845886 | 0.886 |
| 47           | simple_nn1 | 48                  | 97.2 | 0.328829437494278 | 0.886 |
| 48           | simple_nn1 | 49                  | 96.2 | 0.492945283651352 | 0.886 |
| 49           | simple_nn1 | 50                  | 97.4 | 0.41621944308281  | 0.886 |
| 50           | simple_nn1 | 51                  | 96.6 | 0.458861321210861 | 0.886 |
| 51           | simple_nn1 | 52                  | 97.1 | 0.436484277248383 | 0.886 |
| 52           | simple_nn1 | 53                  | 96.7 | 0.405433148145676 | 0.886 |
| 53           | simple_nn1 | 54                  | 96.8 | 0.382160127162933 | 0.886 |
| 54           | simple_nn1 | 55                  | 97.4 | 0.376876294612884 | 0.886 |
| 55           | simple_nn1 | 56                  | 96.5 | 0.439840376377106 | 0.886 |
| 56           | simple_nn1 | 57                  | 96.9 | 0.37757596373558  | 0.886 |
| 57           | simple_nn1 | 58                  | 96.2 | 0.372366070747375 | 0.886 |
| 58           | simple_nn1 | 59                  | 97   | 0.347762048244476 | 0.886 |
| 59           | simple_nn1 | 60                  | 97   | 0.366534560918808 | 0.886 |
| 60           | simple_nn1 | 61                  | 96.7 | 0.371229946613312 | 0.886 |
| 61           | simple_nn1 | 62                  | 96.8 | 0.34061986207962  | 0.886 |
| 62           | simple_nn1 | 63                  | 97.1 | 0.363103032112122 | 0.886 |
| 63           | simple_nn1 | 64                  | 96.9 | 0.438998341560364 | 0.886 |
| 64           | simple_nn1 | 65                  | 96.9 | 0.532387137413025 | 0.886 |
| 65           | simple_nn1 | 66                  | 97   | 0.358926028013229 | 0.886 |
| 66           | simple_nn1 | 67                  | 96.6 | 0.524419128894806 | 0.886 |
| 67           | simple_nn1 | 68                  | 97.1 | 0.376309484243393 | 0.886 |
| 68           | simple_nn1 | 69                  | 96.9 | 0.377277612686157 | 0.886 |
| 69           | simple_nn1 | 70                  | 97.1 | 0.366865396499634 | 0.886 |
| 70           | simple_nn1 | 71                  | 96.6 | 0.354009419679642 | 0.886 |
| 71           | simple_nn1 | 72                  | 97   | 0.512196660041809 | 0.886 |
| 72           | simple_nn1 | 73                  | 97   | 0.403117001056671 | 0.886 |
| 73           | simple_nn1 | 74                  | 97.4 | 0.329790472984314 | 0.886 |
| 74           | simple_nn1 | 75                  | 97.1 | 0.320823669433594 | 0.886 |
| 75           | simple_nn1 | 76                  | 96.7 | 0.505114555358887 | 0.886 |
| 76           | simple_nn1 | 77                  | 96.5 | 0.381211340427399 | 0.886 |
| 77           | simple_nn1 | 78                  | 97.4 | 0.450467377901077 | 0.886 |
| 78           | simple_nn1 | 79                  | 96.9 | 0.354236871004105 | 0.886 |
| 79           | simple_nn1 | 80                  | 97.2 | 0.374177753925323 | 0.886 |
| 80           | simple_nn1 | 81                  | 97.2 | 0.390896677970886 | 0.886 |
| 81           | simple_nn1 | 82                  | 96.2 | 0.39518803358078  | 0.886 |
| 82           | simple_nn1 | 83                  | 96.9 | 0.301871299743652 | 0.886 |
| 83           | simple_nn1 | 84                  | 96.8 | 0.417699784040451 | 0.886 |
| 84           | simple_nn1 | 85                  | 96.9 | 0.382896333932877 | 0.886 |
| 85           | simple_nn1 | 86                  | 97.4 | 0.413862228393555 | 0.886 |
| 86           | simple_nn1 | 87                  | 97.3 | 0.385150015354156 | 0.886 |
| 87           | simple_nn1 | 88                  | 96.6 | 0.451551795005798 | 0.886 |
| 88           | simple_nn1 | 89                  | 97.1 | 0.412499099969864 | 0.886 |
| 89           | simple_nn1 | 90                  | 97.3 | 0.359417021274567 | 0.886 |
| 90           | simple_nn1 | 91                  | 97   | 0.472228646278381 | 0.886 |
| 91           | simple_nn1 | 92                  | 97.4 | 0.440199494361877 | 0.886 |
| 92           | simple_nn1 | 93                  | 97.4 | 0.444103956222534 | 0.886 |
| 93           | simple_nn1 | 94                  | 96.9 | 0.473282933235168 | 0.886 |
| 94           | simple_nn1 | 95                  | 96.9 | 0.369516611099243 | 0.886 |
| 95           | simple_nn1 | 96                  | 96.7 | 0.543198823928833 | 0.886 |
| 96           | simple_nn1 | 97                  | 96.9 | 0.452869117259979 | 0.886 |
| 97           | simple_nn1 | 98                  | 97   | 0.410798013210297 | 0.886 |
| 98           | simple_nn1 | 99                  | 97.1 | 0.329835772514343 | 0.886 |
| 99           | simple_nn1 | 100                 | 96.7 | 0.497502148151398 | 0.886 |
| 100          | simple_nn1 | 101                 | 97.3 | 0.421614944934845 | 0.886 |
| 101          | simple_nn1 | 102                 | 97.2 | 0.356275141239166 | 0.886 |
| 102          | simple_nn1 | 103                 | 96.9 | 0.565607130527496 | 0.886 |
| 103          | simple_nn1 | 104                 | 96.9 | 0.610546469688416 | 0.886 |
| 104          | simple_nn1 | 105                 | 96.5 | 0.429919183254242 | 0.886 |
| 105          | simple_nn1 | 106                 | 96.8 | 0.446462363004684 | 0.886 |
| 106          | simple_nn1 | 107                 | 96.9 | 0.46742445230484  | 0.886 |
| 107          | simple_nn1 | 108                 | 96.6 | 0.559587240219116 | 0.886 |
| 108          | simple_nn1 | 109                 | 97.3 | 0.358132898807526 | 0.886 |
| 109          | simple_nn1 | 110                 | 96.5 | 0.503685235977173 | 0.886 |
| 110          | simple_nn1 | 111                 | 96.1 | 0.520423352718353 | 0.886 |
| 111          | simple_nn1 | 112                 | 97.3 | 0.354635953903198 | 0.886 |
| 112          | simple_nn1 | 113                 | 97.2 | 0.372787475585937 | 0.886 |
| 113          | simple_nn1 | 114                 | 96.4 | 0.429350554943085 | 0.886 |
| 114          | simple_nn1 | 115                 | 97.1 | 0.384742200374603 | 0.886 |
| 115          | simple_nn1 | 116                 | 96.6 | 0.370227813720703 | 0.886 |
| 116          | simple_nn1 | 117                 | 97.1 | 0.447450369596481 | 0.886 |
| 117          | simple_nn1 | 118                 | 96.9 | 0.359192550182343 | 0.886 |
| 118          | simple_nn1 | 119                 | 97.1 | 0.442915797233582 | 0.886 |
| 119          | simple_nn1 | 120                 | 96.6 | 0.405739903450012 | 0.886 |
| 120          | simple_nn1 | 121                 | 96.7 | 0.35691225528717  | 0.886 |
| 121          | simple_nn1 | 122                 | 97.2 | 0.351175338029861 | 0.886 |
| 122          | simple_nn1 | 123                 | 96.7 | 0.570326626300812 | 0.886 |
| 123          | simple_nn1 | 124                 | 97.3 | 0.485489428043366 | 0.886 |
| 124          | simple_nn1 | 125                 | 97.2 | 0.391121923923492 | 0.886 |
| 125          | simple_nn1 | 126                 | 96.8 | 0.415542185306549 | 0.886 |
| 126          | simple_nn1 | 127                 | 96.6 | 0.410668253898621 | 0.886 |
| 127          | simple_nn1 | 128                 | 97   | 0.32479265332222  | 0.886 |
| 128          | simple_nn1 | 129                 | 96.6 | 0.306562632322311 | 0.886 |
| 129          | simple_nn1 | 130                 | 97   | 0.42772513628006  | 0.886 |
| 130          | simple_nn1 | 131                 | 96.8 | 0.465156733989716 | 0.886 |
| 131          | simple_nn1 | 132                 | 96.9 | 0.3470259308815   | 0.886 |
| 132          | simple_nn1 | 133                 | 96.9 | 0.449479520320892 | 0.886 |
| 133          | simple_nn1 | 134                 | 97.2 | 0.410668224096298 | 0.886 |
| 134          | simple_nn1 | 135                 | 96.5 | 0.438186407089233 | 0.886 |
| 135          | simple_nn1 | 136                 | 97   | 0.394161224365234 | 0.886 |
| 136          | simple_nn1 | 137                 | 96.7 | 0.450700640678406 | 0.886 |
| 137          | simple_nn1 | 138                 | 96.6 | 0.390126347541809 | 0.886 |
| 138          | simple_nn1 | 139                 | 96.6 | 0.402502208948135 | 0.886 |
| 139          | simple_nn1 | 140                 | 97.6 | 0.290476590394974 | 0.886 |
| 140          | simple_nn1 | 141                 | 96.7 | 0.493057638406754 | 0.886 |
| 141          | simple_nn1 | 142                 | 97.4 | 0.315250694751739 | 0.886 |
| 142          | simple_nn1 | 143                 | 96.8 | 0.410568475723267 | 0.886 |
| 143          | simple_nn1 | 144                 | 97.2 | 0.488884091377258 | 0.886 |
| 144          | simple_nn1 | 145                 | 96.7 | 0.344393134117126 | 0.886 |
| 145          | simple_nn1 | 146                 | 97   | 0.412308573722839 | 0.886 |
| 146          | simple_nn1 | 147                 | 96.7 | 0.402419924736023 | 0.886 |
| 147          | simple_nn1 | 148                 | 97.1 | 0.400189638137817 | 0.886 |
| 148          | simple_nn1 | 149                 | 97.3 | 0.350034952163696 | 0.886 |
| 149          | simple_nn1 | 150                 | 97.2 | 0.341144472360611 | 0.886 |
| 0            | simple_nn2 | 1                   | 67.4 | 2.66741013526916  | 0.833 |
| 1            | simple_nn2 | 2                   | 79.1 | 2.06428122520447  | 0.833 |
| 2            | simple_nn2 | 3                   | 83.6 | 1.66706335544586  | 0.833 |
| 3            | simple_nn2 | 4                   | 85.8 | 1.45307374000549  | 0.833 |
| 4            | simple_nn2 | 5                   | 86.5 | 1.17383193969727  | 0.833 |
| 5            | simple_nn2 | 6                   | 88.4 | 1.17119479179382  | 0.833 |
| 6            | simple_nn2 | 7                   | 89.4 | 1.08104002475739  | 0.833 |
| 7            | simple_nn2 | 8                   | 89.6 | 1.09526884555817  | 0.833 |
| 8            | simple_nn2 | 9                   | 90.2 | 0.920234441757202 | 0.833 |
| 9            | simple_nn2 | 10                  | 90.4 | 0.985201239585876 | 0.833 |
| 10           | simple_nn2 | 11                  | 90.4 | 0.918547749519348 | 0.833 |
| 11           | simple_nn2 | 12                  | 91.1 | 0.859308660030365 | 0.833 |
| 12           | simple_nn2 | 13                  | 90.8 | 0.920692920684814 | 0.833 |
| 13           | simple_nn2 | 14                  | 91.2 | 0.798810482025147 | 0.833 |
| 14           | simple_nn2 | 15                  | 91   | 0.928196310997009 | 0.833 |
| 15           | simple_nn2 | 16                  | 91.4 | 0.843706309795379 | 0.833 |
| 16           | simple_nn2 | 17                  | 90.8 | 0.871180951595306 | 0.833 |
| 17           | simple_nn2 | 18                  | 90.7 | 0.69849157333374  | 0.833 |
| 18           | simple_nn2 | 19                  | 91.4 | 0.744959831237793 | 0.833 |
| 19           | simple_nn2 | 20                  | 91.4 | 0.758549332618713 | 0.833 |
| 20           | simple_nn2 | 21                  | 91.4 | 0.785315155982971 | 0.833 |
| 21           | simple_nn2 | 22                  | 91.8 | 0.709188580513001 | 0.833 |
| 22           | simple_nn2 | 23                  | 91.4 | 0.702919363975525 | 0.833 |
| 23           | simple_nn2 | 24                  | 91.4 | 0.831278324127197 | 0.833 |
| 24           | simple_nn2 | 25                  | 92.2 | 0.824142038822174 | 0.833 |
| 25           | simple_nn2 | 26                  | 92   | 0.785410523414612 | 0.833 |
| 26           | simple_nn2 | 27                  | 91.7 | 0.956349492073059 | 0.833 |
| 27           | simple_nn2 | 28                  | 92.2 | 0.884123682975769 | 0.833 |
| 28           | simple_nn2 | 29                  | 91.9 | 0.654408872127533 | 0.833 |
| 29           | simple_nn2 | 30                  | 91.8 | 0.762592196464539 | 0.833 |
| 30           | simple_nn2 | 31                  | 92.2 | 0.69425642490387  | 0.833 |
| 31           | simple_nn2 | 32                  | 92.4 | 0.80879819393158  | 0.833 |
| 32           | simple_nn2 | 33                  | 92.3 | 0.602601587772369 | 0.833 |
| 33           | simple_nn2 | 34                  | 92.1 | 0.581052780151367 | 0.833 |
| 34           | simple_nn2 | 35                  | 92.6 | 0.757736682891846 | 0.833 |
| 35           | simple_nn2 | 36                  | 91.7 | 0.673137545585632 | 0.833 |
| 36           | simple_nn2 | 37                  | 91.4 | 0.855386972427368 | 0.833 |
| 37           | simple_nn2 | 38                  | 91.5 | 0.704023957252502 | 0.833 |
| 38           | simple_nn2 | 39                  | 91.4 | 1.01098251342773  | 0.833 |
| 39           | simple_nn2 | 40                  | 91.9 | 0.77825665473938  | 0.833 |
| 40           | simple_nn2 | 41                  | 92.1 | 0.740197420120239 | 0.833 |
| 41           | simple_nn2 | 42                  | 92.8 | 0.803945779800415 | 0.833 |
| 42           | simple_nn2 | 43                  | 92.1 | 0.774712979793549 | 0.833 |
| 43           | simple_nn2 | 44                  | 92.6 | 0.664218246936798 | 0.833 |
| 44           | simple_nn2 | 45                  | 91.7 | 0.667913198471069 | 0.833 |
| 45           | simple_nn2 | 46                  | 91.4 | 0.822562873363495 | 0.833 |
| 46           | simple_nn2 | 47                  | 92.2 | 0.750758290290833 | 0.833 |
| 47           | simple_nn2 | 48                  | 92.5 | 0.864637732505798 | 0.833 |
| 48           | simple_nn2 | 49                  | 92.6 | 0.792967319488525 | 0.833 |
| 49           | simple_nn2 | 50                  | 92.2 | 0.843515753746033 | 0.833 |
| 50           | simple_nn2 | 51                  | 92.3 | 0.840755343437195 | 0.833 |
| 51           | simple_nn2 | 52                  | 92.2 | 0.702629625797272 | 0.833 |
| 52           | simple_nn2 | 53                  | 93.1 | 0.842038035392761 | 0.833 |
| 53           | simple_nn2 | 54                  | 91.7 | 0.625619351863861 | 0.833 |
| 54           | simple_nn2 | 55                  | 92   | 0.831954181194305 | 0.833 |
| 55           | simple_nn2 | 56                  | 91.9 | 0.918867588043213 | 0.833 |
| 56           | simple_nn2 | 57                  | 92.6 | 0.828541040420532 | 0.833 |
| 57           | simple_nn2 | 58                  | 92.3 | 0.790839433670044 | 0.833 |
| 58           | simple_nn2 | 59                  | 93.2 | 0.686766147613525 | 0.833 |
| 59           | simple_nn2 | 60                  | 92.6 | 0.644819974899292 | 0.833 |
| 60           | simple_nn2 | 61                  | 92.9 | 0.84963059425354  | 0.833 |
| 61           | simple_nn2 | 62                  | 93.3 | 0.618531405925751 | 0.833 |
| 62           | simple_nn2 | 63                  | 92.9 | 0.853845000267029 | 0.833 |
| 63           | simple_nn2 | 64                  | 93   | 0.774389505386353 | 0.833 |
| 64           | simple_nn2 | 65                  | 92.8 | 0.820277631282806 | 0.833 |
| 65           | simple_nn2 | 66                  | 92.3 | 0.779565691947937 | 0.833 |
| 66           | simple_nn2 | 67                  | 92.3 | 0.718371391296387 | 0.833 |
| 67           | simple_nn2 | 68                  | 92.4 | 0.728549957275391 | 0.833 |
| 68           | simple_nn2 | 69                  | 92.5 | 0.829271137714386 | 0.833 |
| 69           | simple_nn2 | 70                  | 92.4 | 0.796635866165161 | 0.833 |
| 70           | simple_nn2 | 71                  | 92.6 | 0.704606473445892 | 0.833 |
| 71           | simple_nn2 | 72                  | 92.4 | 0.739652037620544 | 0.833 |
| 72           | simple_nn2 | 73                  | 92.7 | 0.610159158706665 | 0.833 |
| 73           | simple_nn2 | 74                  | 92.2 | 0.739535450935364 | 0.833 |
| 74           | simple_nn2 | 75                  | 92.6 | 0.677067041397095 | 0.833 |
| 75           | simple_nn2 | 76                  | 92.6 | 1.02190315723419  | 0.833 |
| 76           | simple_nn2 | 77                  | 92.8 | 0.772656559944153 | 0.833 |
| 77           | simple_nn2 | 78                  | 92.1 | 0.68167769908905  | 0.833 |
| 78           | simple_nn2 | 79                  | 93.3 | 0.665133953094482 | 0.833 |
| 79           | simple_nn2 | 80                  | 92.7 | 0.684036433696747 | 0.833 |
| 80           | simple_nn2 | 81                  | 92.8 | 0.708353877067566 | 0.833 |
| 81           | simple_nn2 | 82                  | 92.2 | 0.645722270011902 | 0.833 |
| 82           | simple_nn2 | 83                  | 93.1 | 0.635635852813721 | 0.833 |
| 83           | simple_nn2 | 84                  | 92.1 | 0.651131987571716 | 0.833 |
| 84           | simple_nn2 | 85                  | 92.4 | 0.795397520065308 | 0.833 |
| 85           | simple_nn2 | 86                  | 92   | 0.61284601688385  | 0.833 |
| 86           | simple_nn2 | 87                  | 92.6 | 0.85322117805481  | 0.833 |
| 87           | simple_nn2 | 88                  | 93.2 | 0.678975164890289 | 0.833 |
| 88           | simple_nn2 | 89                  | 92.1 | 0.745078086853027 | 0.833 |
| 89           | simple_nn2 | 90                  | 92.4 | 0.824579119682312 | 0.833 |
| 90           | simple_nn2 | 91                  | 92.4 | 0.654780387878418 | 0.833 |
| 91           | simple_nn2 | 92                  | 92.9 | 0.709741055965424 | 0.833 |
| 92           | simple_nn2 | 93                  | 92.1 | 0.920865654945374 | 0.833 |
| 93           | simple_nn2 | 94                  | 93.1 | 0.706007599830627 | 0.833 |
| 94           | simple_nn2 | 95                  | 92.8 | 0.635524153709412 | 0.833 |
| 95           | simple_nn2 | 96                  | 92.7 | 0.723670184612274 | 0.833 |
| 96           | simple_nn2 | 97                  | 92.5 | 0.624796271324158 | 0.833 |
| 97           | simple_nn2 | 98                  | 92.9 | 0.689633965492249 | 0.833 |
| 98           | simple_nn2 | 99                  | 92   | 0.763447523117065 | 0.833 |
| 99           | simple_nn2 | 100                 | 93   | 0.701947450637817 | 0.833 |
| 100          | simple_nn2 | 101                 | 92.9 | 0.830548644065857 | 0.833 |
| 101          | simple_nn2 | 102                 | 93.1 | 0.758425116539001 | 0.833 |
| 102          | simple_nn2 | 103                 | 92.9 | 0.936367154121399 | 0.833 |
| 103          | simple_nn2 | 104                 | 93   | 0.638247430324554 | 0.833 |
| 104          | simple_nn2 | 105                 | 92.8 | 0.676411032676697 | 0.833 |
| 105          | simple_nn2 | 106                 | 93   | 0.705335378646851 | 0.833 |
| 106          | simple_nn2 | 107                 | 93.1 | 0.809989213943481 | 0.833 |
| 107          | simple_nn2 | 108                 | 92.8 | 0.725915193557739 | 0.833 |
| 108          | simple_nn2 | 109                 | 92.6 | 0.839659094810486 | 0.833 |
| 109          | simple_nn2 | 110                 | 92.5 | 0.868207216262817 | 0.833 |
| 110          | simple_nn2 | 111                 | 92.6 | 0.667255222797394 | 0.833 |
| 111          | simple_nn2 | 112                 | 92.8 | 0.804173946380615 | 0.833 |
| 112          | simple_nn2 | 113                 | 93.4 | 0.668572902679443 | 0.833 |
| 113          | simple_nn2 | 114                 | 93   | 0.688282012939453 | 0.833 |
| 114          | simple_nn2 | 115                 | 92.8 | 0.774609804153442 | 0.833 |
| 115          | simple_nn2 | 116                 | 92.8 | 0.831467628479004 | 0.833 |
| 116          | simple_nn2 | 117                 | 93.1 | 0.653700947761535 | 0.833 |
| 117          | simple_nn2 | 118                 | 92.4 | 0.692154943943024 | 0.833 |
| 118          | simple_nn2 | 119                 | 92.7 | 0.646853685379028 | 0.833 |
| 119          | simple_nn2 | 120                 | 93.3 | 0.721762418746948 | 0.833 |
| 120          | simple_nn2 | 121                 | 92.9 | 0.786952614784241 | 0.833 |
| 121          | simple_nn2 | 122                 | 92.8 | 0.554685711860657 | 0.833 |
| 122          | simple_nn2 | 123                 | 93.1 | 0.80525529384613  | 0.833 |
| 123          | simple_nn2 | 124                 | 93.2 | 1.00832855701447  | 0.833 |
| 124          | simple_nn2 | 125                 | 93.2 | 0.656032085418701 | 0.833 |
| 125          | simple_nn2 | 126                 | 92.9 | 0.691248297691345 | 0.833 |
| 126          | simple_nn2 | 127                 | 93.4 | 0.738853454589844 | 0.833 |
| 127          | simple_nn2 | 128                 | 93   | 0.599465012550354 | 0.833 |
| 128          | simple_nn2 | 129                 | 93   | 0.859844923019409 | 0.833 |
| 129          | simple_nn2 | 130                 | 92.7 | 0.639019370079041 | 0.833 |
| 130          | simple_nn2 | 131                 | 92   | 0.620681703090668 | 0.833 |
| 131          | simple_nn2 | 132                 | 92.4 | 0.720185160636902 | 0.833 |
| 132          | simple_nn2 | 133                 | 93.3 | 0.666834890842438 | 0.833 |
| 133          | simple_nn2 | 134                 | 93.2 | 0.804269969463348 | 0.833 |
| 134          | simple_nn2 | 135                 | 92.8 | 0.671154975891113 | 0.833 |
| 135          | simple_nn2 | 136                 | 92.6 | 0.68506246805191  | 0.833 |
| 136          | simple_nn2 | 137                 | 92.6 | 0.715125560760498 | 0.833 |
| 137          | simple_nn2 | 138                 | 92.5 | 0.636201143264771 | 0.833 |
| 138          | simple_nn2 | 139                 | 92.9 | 0.69077205657959  | 0.833 |
| 139          | simple_nn2 | 140                 | 93.2 | 0.830615043640137 | 0.833 |
| 140          | simple_nn2 | 141                 | 92.3 | 0.771365404129028 | 0.833 |
| 141          | simple_nn2 | 142                 | 93.2 | 0.751102447509766 | 0.833 |
| 142          | simple_nn2 | 143                 | 93.2 | 0.636639475822449 | 0.833 |
| 143          | simple_nn2 | 144                 | 93.1 | 0.661173045635223 | 0.833 |
| 144          | simple_nn2 | 145                 | 92.9 | 0.778049230575562 | 0.833 |
| 145          | simple_nn2 | 146                 | 92.8 | 0.538061857223511 | 0.833 |
| 146          | simple_nn2 | 147                 | 92.6 | 0.57715231180191  | 0.833 |
| 147          | simple_nn2 | 148                 | 93.4 | 0.724302768707275 | 0.833 |
| 148          | simple_nn2 | 149                 | 93.1 | 0.585075736045837 | 0.833 |
| 149          | simple_nn2 | 150                 | 92.1 | 0.72547721862793  | 0.833 |
| 0            | alexnet    | 1                   | 96.3 | 26.2749404907227  | 0.954 |
| 1            | alexnet    | 2                   | 96.7 | 7.74424123764038  | 0.954 |
| 2            | alexnet    | 3                   | 95.8 | 3.71041321754456  | 0.954 |
| 3            | alexnet    | 4                   | 97   | 2.54075765609741  | 0.954 |
| 4            | alexnet    | 5                   | 95.1 | 1.75660753250122  | 0.954 |
| 5            | alexnet    | 6                   | 97   | 1.45307183265686  | 0.954 |
| 6            | alexnet    | 7                   | 98   | 0.961805701255798 | 0.954 |
| 7            | alexnet    | 8                   | 98.2 | 1.08632457256317  | 0.954 |
| 8            | alexnet    | 9                   | 98.5 | 0.701003432273865 | 0.954 |
| 9            | alexnet    | 10                  | 98.9 | 0.747330367565155 | 0.954 |
| 10           | alexnet    | 11                  | 97.3 | 0.969053030014038 | 0.954 |
| 11           | alexnet    | 12                  | 98.8 | 0.697041690349579 | 0.954 |
| 12           | alexnet    | 13                  | 98.3 | 0.665219306945801 | 0.954 |
| 13           | alexnet    | 14                  | 97.8 | 0.504586637020111 | 0.954 |
| 14           | alexnet    | 15                  | 99.2 | 0.58502072095871  | 0.954 |
| 15           | alexnet    | 16                  | 98.7 | 0.789038777351379 | 0.954 |
| 16           | alexnet    | 17                  | 98.8 | 0.534981727600098 | 0.954 |
| 17           | alexnet    | 18                  | 98.7 | 0.690969347953796 | 0.954 |
| 18           | alexnet    | 19                  | 98.4 | 0.598352432250977 | 0.954 |
| 19           | alexnet    | 20                  | 99.4 | 0.385382771492004 | 0.954 |
| 20           | alexnet    | 21                  | 99.4 | 0.365357279777527 | 0.954 |
| 21           | alexnet    | 22                  | 98.4 | 0.764978945255279 | 0.954 |
| 22           | alexnet    | 23                  | 99   | 0.443120777606964 | 0.954 |
| 23           | alexnet    | 24                  | 98.8 | 0.380575120449066 | 0.954 |
| 24           | alexnet    | 25                  | 98.7 | 0.592876553535461 | 0.954 |
| 25           | alexnet    | 26                  | 98.7 | 0.392586529254913 | 0.954 |
| 26           | alexnet    | 27                  | 99.1 | 0.378941267728806 | 0.954 |
| 27           | alexnet    | 28                  | 98.9 | 0.365493774414062 | 0.954 |
| 28           | alexnet    | 29                  | 98.9 | 0.394469290971756 | 0.954 |
| 29           | alexnet    | 30                  | 99.5 | 0.351047217845917 | 0.954 |
| 30           | alexnet    | 31                  | 99.2 | 0.349187791347504 | 0.954 |
| 31           | alexnet    | 32                  | 99.4 | 0.439707964658737 | 0.954 |
| 32           | alexnet    | 33                  | 99.2 | 0.479944109916687 | 0.954 |
| 33           | alexnet    | 34                  | 98.6 | 0.383744418621063 | 0.954 |
| 34           | alexnet    | 35                  | 99.6 | 0.304406851530075 | 0.954 |
| 35           | alexnet    | 36                  | 97.8 | 0.453451126813889 | 0.954 |
| 36           | alexnet    | 37                  | 99.2 | 0.362347424030304 | 0.954 |
| 37           | alexnet    | 38                  | 99.1 | 0.389520525932312 | 0.954 |
| 38           | alexnet    | 39                  | 99.2 | 0.405095040798187 | 0.954 |
| 39           | alexnet    | 40                  | 99.6 | 0.281712234020233 | 0.954 |
| 40           | alexnet    | 41                  | 99.2 | 0.454672515392303 | 0.954 |
| 41           | alexnet    | 42                  | 99.2 | 0.35501104593277  | 0.954 |
| 42           | alexnet    | 43                  | 99.3 | 0.461896300315857 | 0.954 |
| 43           | alexnet    | 44                  | 99.1 | 0.360627174377441 | 0.954 |
| 44           | alexnet    | 45                  | 99.1 | 0.313998609781265 | 0.954 |
| 45           | alexnet    | 46                  | 99.2 | 0.353889256715775 | 0.954 |
| 46           | alexnet    | 47                  | 99.2 | 0.329591065645218 | 0.954 |
| 47           | alexnet    | 48                  | 99.1 | 0.32823172211647  | 0.954 |
| 48           | alexnet    | 49                  | 98.6 | 0.367546081542969 | 0.954 |
| 49           | alexnet    | 50                  | 99.3 | 0.379570513963699 | 0.954 |
| 50           | alexnet    | 51                  | 97.7 | 0.703997611999512 | 0.954 |
| 51           | alexnet    | 52                  | 99   | 0.359448432922363 | 0.954 |
| 52           | alexnet    | 53                  | 98.8 | 0.556829452514648 | 0.954 |
| 53           | alexnet    | 54                  | 99.4 | 0.277466416358948 | 0.954 |
| 54           | alexnet    | 55                  | 99.4 | 0.388734489679337 | 0.954 |
| 55           | alexnet    | 56                  | 99.2 | 0.333289265632629 | 0.954 |
| 56           | alexnet    | 57                  | 99.5 | 0.347097814083099 | 0.954 |
| 57           | alexnet    | 58                  | 99.4 | 0.326541125774384 | 0.954 |
| 58           | alexnet    | 59                  | 99.1 | 0.343861967325211 | 0.954 |
| 59           | alexnet    | 60                  | 99.2 | 0.309823334217072 | 0.954 |
| 60           | alexnet    | 61                  | 99.5 | 0.397097110748291 | 0.954 |
| 61           | alexnet    | 62                  | 99   | 0.289588332176208 | 0.954 |
| 62           | alexnet    | 63                  | 99   | 0.311627626419067 | 0.954 |
| 63           | alexnet    | 64                  | 99   | 0.290801674127579 | 0.954 |
| 64           | alexnet    | 65                  | 99.3 | 0.297502398490906 | 0.954 |
| 65           | alexnet    | 66                  | 99.3 | 0.32006961107254  | 0.954 |
| 66           | alexnet    | 67                  | 98.7 | 0.273110270500183 | 0.954 |
| 67           | alexnet    | 68                  | 99   | 0.41022053360939  | 0.954 |
| 68           | alexnet    | 69                  | 99.4 | 0.301292777061462 | 0.954 |
| 69           | alexnet    | 70                  | 99.2 | 0.370980560779572 | 0.954 |
| 70           | alexnet    | 71                  | 99.4 | 0.270659059286118 | 0.954 |
| 71           | alexnet    | 72                  | 99.1 | 0.294102430343628 | 0.954 |
| 72           | alexnet    | 73                  | 99.6 | 0.289619743824005 | 0.954 |
| 73           | alexnet    | 74                  | 99.3 | 0.354983627796173 | 0.954 |
| 74           | alexnet    | 75                  | 98.9 | 0.532078266143799 | 0.954 |
| 75           | alexnet    | 76                  | 98.9 | 0.456400990486145 | 0.954 |
| 76           | alexnet    | 77                  | 99   | 0.264904260635376 | 0.954 |
| 77           | alexnet    | 78                  | 99.1 | 0.303837120532989 | 0.954 |
| 78           | alexnet    | 79                  | 99.4 | 0.290091633796692 | 0.954 |
| 79           | alexnet    | 80                  | 99.5 | 0.275074660778046 | 0.954 |
| 80           | alexnet    | 81                  | 99   | 0.338803559541702 | 0.954 |
| 81           | alexnet    | 82                  | 98.3 | 0.440221726894379 | 0.954 |
| 82           | alexnet    | 83                  | 99.2 | 0.361238300800323 | 0.954 |
| 83           | alexnet    | 84                  | 98.4 | 0.443875730037689 | 0.954 |
| 84           | alexnet    | 85                  | 99   | 0.283835679292679 | 0.954 |
| 85           | alexnet    | 86                  | 98.7 | 0.532202184200287 | 0.954 |
| 86           | alexnet    | 87                  | 98.9 | 0.491292297840118 | 0.954 |
| 87           | alexnet    | 88                  | 99.4 | 0.272197782993317 | 0.954 |
| 88           | alexnet    | 89                  | 98.9 | 0.281462222337723 | 0.954 |
| 89           | alexnet    | 90                  | 99.6 | 0.260485202074051 | 0.954 |
| 90           | alexnet    | 91                  | 99.1 | 0.372459173202515 | 0.954 |
| 91           | alexnet    | 92                  | 99.5 | 0.298528790473938 | 0.954 |
| 92           | alexnet    | 93                  | 99   | 0.300070941448212 | 0.954 |
| 93           | alexnet    | 94                  | 99.2 | 0.338181167840958 | 0.954 |
| 94           | alexnet    | 95                  | 99.3 | 0.255412936210632 | 0.954 |
| 95           | alexnet    | 96                  | 99.1 | 0.517260432243347 | 0.954 |
| 96           | alexnet    | 97                  | 99.4 | 0.288031220436096 | 0.954 |
| 97           | alexnet    | 98                  | 98.8 | 0.392812669277191 | 0.954 |
| 98           | alexnet    | 99                  | 99.4 | 0.274012416601181 | 0.954 |
| 99           | alexnet    | 100                 | 98.8 | 0.292465299367905 | 0.954 |
| 100          | alexnet    | 101                 | 98.2 | 0.604820489883423 | 0.954 |
| 101          | alexnet    | 102                 | 99.6 | 0.253884643316269 | 0.954 |
| 102          | alexnet    | 103                 | 99.1 | 0.251197010278702 | 0.954 |
| 103          | alexnet    | 104                 | 99.3 | 0.324398636817932 | 0.954 |
| 104          | alexnet    | 105                 | 99.5 | 0.260720163583755 | 0.954 |
| 105          | alexnet    | 106                 | 98.9 | 0.339387714862823 | 0.954 |
| 106          | alexnet    | 107                 | 99.1 | 0.346414387226105 | 0.954 |
| 107          | alexnet    | 108                 | 98.9 | 0.286338806152344 | 0.954 |
| 108          | alexnet    | 109                 | 99.1 | 0.262272328138351 | 0.954 |
| 109          | alexnet    | 110                 | 98.8 | 0.512029767036438 | 0.954 |
| 110          | alexnet    | 111                 | 99.1 | 0.306622445583343 | 0.954 |
| 111          | alexnet    | 112                 | 99.1 | 0.420867383480072 | 0.954 |
| 112          | alexnet    | 113                 | 98.4 | 0.403161883354187 | 0.954 |
| 113          | alexnet    | 114                 | 99.3 | 0.26814067363739  | 0.954 |
| 114          | alexnet    | 115                 | 99.3 | 0.249728128314018 | 0.954 |
| 115          | alexnet    | 116                 | 99.4 | 0.251666814088821 | 0.954 |
| 116          | alexnet    | 117                 | 98.2 | 0.266170531511307 | 0.954 |
| 117          | alexnet    | 118                 | 99.5 | 0.279242634773254 | 0.954 |
| 118          | alexnet    | 119                 | 99.4 | 0.241375878453255 | 0.954 |
| 119          | alexnet    | 120                 | 98.8 | 0.374202251434326 | 0.954 |
| 120          | alexnet    | 121                 | 99.6 | 0.23839345574379  | 0.954 |
| 121          | alexnet    | 122                 | 98.9 | 0.37282919883728  | 0.954 |
| 122          | alexnet    | 123                 | 99.2 | 0.23824243247509  | 0.954 |
| 123          | alexnet    | 124                 | 99   | 0.271578907966614 | 0.954 |
| 124          | alexnet    | 125                 | 99.2 | 0.307115495204926 | 0.954 |
| 125          | alexnet    | 126                 | 98.9 | 0.283430576324463 | 0.954 |
| 126          | alexnet    | 127                 | 99   | 0.31258887052536  | 0.954 |
| 127          | alexnet    | 128                 | 99.3 | 0.281039386987686 | 0.954 |
| 128          | alexnet    | 129                 | 98.8 | 0.32544869184494  | 0.954 |
| 129          | alexnet    | 130                 | 99   | 0.293349087238312 | 0.954 |
| 130          | alexnet    | 131                 | 98.8 | 0.435329675674439 | 0.954 |
| 131          | alexnet    | 132                 | 99.5 | 0.283255726099014 | 0.954 |
| 132          | alexnet    | 133                 | 99.3 | 0.290360569953918 | 0.954 |
| 133          | alexnet    | 134                 | 99.3 | 0.235175564885139 | 0.954 |
| 134          | alexnet    | 135                 | 99.1 | 0.29720014333725  | 0.954 |
| 135          | alexnet    | 136                 | 99.5 | 0.280366539955139 | 0.954 |
| 136          | alexnet    | 137                 | 98.9 | 0.657987177371979 | 0.954 |
| 137          | alexnet    | 138                 | 99.4 | 0.271561682224274 | 0.954 |
| 138          | alexnet    | 139                 | 99.3 | 0.431677639484406 | 0.954 |
| 139          | alexnet    | 140                 | 97   | 0.476840704679489 | 0.954 |
| 140          | alexnet    | 141                 | 98.8 | 0.273041784763336 | 0.954 |
| 141          | alexnet    | 142                 | 99.5 | 0.241969734430313 | 0.954 |
| 142          | alexnet    | 143                 | 99.1 | 0.261145025491714 | 0.954 |
| 143          | alexnet    | 144                 | 99.1 | 0.50212550163269  | 0.954 |
| 144          | alexnet    | 145                 | 99.1 | 0.344536811113358 | 0.954 |
| 145          | alexnet    | 146                 | 99.4 | 0.236252471804619 | 0.954 |
| 146          | alexnet    | 147                 | 99.3 | 0.230109602212906 | 0.954 |
| 147          | alexnet    | 148                 | 98.7 | 0.352993696928024 | 0.954 |
| 148          | alexnet    | 149                 | 99.3 | 0.283613413572311 | 0.954 |
| 149          | alexnet    | 150                 | 98.9 | 0.522108316421509 | 0.954 |
| 0            | lenet      | 1                   | 85.2 | 5.00841569900513  | 0.943 |
| 1            | lenet      | 2                   | 92.8 | 2.44474077224731  | 0.943 |
| 2            | lenet      | 3                   | 94.8 | 1.63162648677826  | 0.943 |
| 3            | lenet      | 4                   | 96.2 | 1.13046026229858  | 0.943 |
| 4            | lenet      | 5                   | 96.7 | 0.895064830780029 | 0.943 |
| 5            | lenet      | 6                   | 97.1 | 0.682721138000488 | 0.943 |
| 6            | lenet      | 7                   | 97.2 | 0.624269962310791 | 0.943 |
| 7            | lenet      | 8                   | 97.5 | 0.530609250068665 | 0.943 |
| 8            | lenet      | 9                   | 97.8 | 0.426977276802063 | 0.943 |
| 9            | lenet      | 10                  | 98.1 | 0.404638409614563 | 0.943 |
| 10           | lenet      | 11                  | 97.8 | 0.444970548152924 | 0.943 |
| 11           | lenet      | 12                  | 98.1 | 0.428657084703445 | 0.943 |
| 12           | lenet      | 13                  | 98.1 | 0.39644068479538  | 0.943 |
| 13           | lenet      | 14                  | 98   | 0.42621710896492  | 0.943 |
| 14           | lenet      | 15                  | 98   | 0.396665543317795 | 0.943 |
| 15           | lenet      | 16                  | 98.2 | 0.439348161220551 | 0.943 |
| 16           | lenet      | 17                  | 98.2 | 0.374234318733215 | 0.943 |
| 17           | lenet      | 18                  | 98.7 | 0.291637897491455 | 0.943 |
| 18           | lenet      | 19                  | 98.4 | 0.374483644962311 | 0.943 |
| 19           | lenet      | 20                  | 98.2 | 0.36075234413147  | 0.943 |
| 20           | lenet      | 21                  | 98.3 | 0.286309361457825 | 0.943 |
| 21           | lenet      | 22                  | 98.4 | 0.317619383335113 | 0.943 |
| 22           | lenet      | 23                  | 98.5 | 0.343027472496033 | 0.943 |
| 23           | lenet      | 24                  | 98.5 | 0.312263071537018 | 0.943 |
| 24           | lenet      | 25                  | 98.9 | 0.291776359081268 | 0.943 |
| 25           | lenet      | 26                  | 98.5 | 0.301185548305511 | 0.943 |
| 26           | lenet      | 27                  | 98.4 | 0.370178818702698 | 0.943 |
| 27           | lenet      | 28                  | 98.2 | 0.247751712799072 | 0.943 |
| 28           | lenet      | 29                  | 98.7 | 0.274353086948395 | 0.943 |
| 29           | lenet      | 30                  | 98.4 | 0.351215302944183 | 0.943 |
| 30           | lenet      | 31                  | 98.7 | 0.232679039239883 | 0.943 |
| 31           | lenet      | 32                  | 98.6 | 0.270397961139679 | 0.943 |
| 32           | lenet      | 33                  | 98.8 | 0.353564709424973 | 0.943 |
| 33           | lenet      | 34                  | 98.5 | 0.348615407943726 | 0.943 |
| 34           | lenet      | 35                  | 98.6 | 0.36773818731308  | 0.943 |
| 35           | lenet      | 36                  | 98.8 | 0.286641538143158 | 0.943 |
| 36           | lenet      | 37                  | 98.8 | 0.232580989599228 | 0.943 |
| 37           | lenet      | 38                  | 98.8 | 0.311236262321472 | 0.943 |
| 38           | lenet      | 39                  | 99   | 0.268564909696579 | 0.943 |
| 39           | lenet      | 40                  | 98.8 | 0.26664400100708  | 0.943 |
| 40           | lenet      | 41                  | 98.4 | 0.254927307367325 | 0.943 |
| 41           | lenet      | 42                  | 98.6 | 0.295538663864136 | 0.943 |
| 42           | lenet      | 43                  | 99   | 0.285465657711029 | 0.943 |
| 43           | lenet      | 44                  | 99   | 0.295769572257996 | 0.943 |
| 44           | lenet      | 45                  | 99   | 0.237250983715057 | 0.943 |
| 45           | lenet      | 46                  | 98.3 | 0.280443370342255 | 0.943 |
| 46           | lenet      | 47                  | 98.8 | 0.271896362304687 | 0.943 |
| 47           | lenet      | 48                  | 99.1 | 0.215384244918823 | 0.943 |
| 48           | lenet      | 49                  | 98.9 | 0.271013528108597 | 0.943 |
| 49           | lenet      | 50                  | 99   | 0.40921226143837  | 0.943 |
| 50           | lenet      | 51                  | 98.6 | 0.366929352283478 | 0.943 |
| 51           | lenet      | 52                  | 99   | 0.253238916397095 | 0.943 |
| 52           | lenet      | 53                  | 98.6 | 0.342853248119354 | 0.943 |
| 53           | lenet      | 54                  | 99   | 0.23275700211525  | 0.943 |
| 54           | lenet      | 55                  | 98.9 | 0.246465474367142 | 0.943 |
| 55           | lenet      | 56                  | 98.8 | 0.266181349754333 | 0.943 |
| 56           | lenet      | 57                  | 99   | 0.239713951945305 | 0.943 |
| 57           | lenet      | 58                  | 98.9 | 0.351640701293945 | 0.943 |
| 58           | lenet      | 59                  | 98.7 | 0.280939638614655 | 0.943 |
| 59           | lenet      | 60                  | 98.9 | 0.191711202263832 | 0.943 |
| 60           | lenet      | 61                  | 98.7 | 0.217704355716705 | 0.943 |
| 61           | lenet      | 62                  | 98.1 | 0.255031853914261 | 0.943 |
| 62           | lenet      | 63                  | 99   | 0.299375712871552 | 0.943 |
| 63           | lenet      | 64                  | 98.8 | 0.245857536792755 | 0.943 |
| 64           | lenet      | 65                  | 98.7 | 0.289814472198486 | 0.943 |
| 65           | lenet      | 66                  | 98.3 | 0.272889316082001 | 0.943 |
| 66           | lenet      | 67                  | 99.2 | 0.235340505838394 | 0.943 |
| 67           | lenet      | 68                  | 98.8 | 0.32211235165596  | 0.943 |
| 68           | lenet      | 69                  | 98.8 | 0.31301337480545  | 0.943 |
| 69           | lenet      | 70                  | 98.1 | 0.384932637214661 | 0.943 |
| 70           | lenet      | 71                  | 98.9 | 0.265338867902756 | 0.943 |
| 71           | lenet      | 72                  | 98.6 | 0.247917175292969 | 0.943 |
| 72           | lenet      | 73                  | 98.2 | 0.354120045900345 | 0.943 |
| 73           | lenet      | 74                  | 98.6 | 0.222755640745163 | 0.943 |
| 74           | lenet      | 75                  | 99   | 0.188791900873184 | 0.943 |
| 75           | lenet      | 76                  | 98.8 | 0.214016824960709 | 0.943 |
| 76           | lenet      | 77                  | 98.8 | 0.250528246164322 | 0.943 |
| 77           | lenet      | 78                  | 98.7 | 0.287720710039139 | 0.943 |
| 78           | lenet      | 79                  | 98.7 | 0.27379047870636  | 0.943 |
| 79           | lenet      | 80                  | 98.5 | 0.264042884111404 | 0.943 |
| 80           | lenet      | 81                  | 98.8 | 0.283451229333878 | 0.943 |
| 81           | lenet      | 82                  | 98.9 | 0.246824130415916 | 0.943 |
| 82           | lenet      | 83                  | 98.8 | 0.299163520336151 | 0.943 |
| 83           | lenet      | 84                  | 98.8 | 0.200576186180115 | 0.943 |
| 84           | lenet      | 85                  | 99.1 | 0.29777181148529  | 0.943 |
| 85           | lenet      | 86                  | 98.8 | 0.294532001018524 | 0.943 |
| 86           | lenet      | 87                  | 99   | 0.253222495317459 | 0.943 |
| 87           | lenet      | 88                  | 98.7 | 0.201173394918442 | 0.943 |
| 88           | lenet      | 89                  | 99.1 | 0.231133610010147 | 0.943 |
| 89           | lenet      | 90                  | 99.1 | 0.232080519199371 | 0.943 |
| 90           | lenet      | 91                  | 98.5 | 0.298993498086929 | 0.943 |
| 91           | lenet      | 92                  | 98.9 | 0.390253007411957 | 0.943 |
| 92           | lenet      | 93                  | 98.6 | 0.22135192155838  | 0.943 |
| 93           | lenet      | 94                  | 98.9 | 0.436125963926315 | 0.943 |
| 94           | lenet      | 95                  | 99.2 | 0.301855593919754 | 0.943 |
| 95           | lenet      | 96                  | 98.8 | 0.364958107471466 | 0.943 |
| 96           | lenet      | 97                  | 98.6 | 0.244299426674843 | 0.943 |
| 97           | lenet      | 98                  | 98.9 | 0.240339934825897 | 0.943 |
| 98           | lenet      | 99                  | 98.7 | 0.301423490047455 | 0.943 |
| 99           | lenet      | 100                 | 99.3 | 0.202781021595001 | 0.943 |
| 100          | lenet      | 101                 | 99   | 0.216359212994575 | 0.943 |
| 101          | lenet      | 102                 | 98.9 | 0.300414443016052 | 0.943 |
| 102          | lenet      | 103                 | 98.7 | 0.237642556428909 | 0.943 |
| 103          | lenet      | 104                 | 98.8 | 0.270957112312317 | 0.943 |
| 104          | lenet      | 105                 | 99.1 | 0.25977885723114  | 0.943 |
| 105          | lenet      | 106                 | 99.3 | 0.248424887657166 | 0.943 |
| 106          | lenet      | 107                 | 99.2 | 0.287781953811646 | 0.943 |
| 107          | lenet      | 108                 | 98.9 | 0.306936502456665 | 0.943 |
| 108          | lenet      | 109                 | 98.7 | 0.255792379379272 | 0.943 |
| 109          | lenet      | 110                 | 98.8 | 0.29299983382225  | 0.943 |
| 110          | lenet      | 111                 | 99.1 | 0.302724570035934 | 0.943 |
| 111          | lenet      | 112                 | 99   | 0.277883529663086 | 0.943 |
| 112          | lenet      | 113                 | 98.9 | 0.216464012861252 | 0.943 |
| 113          | lenet      | 114                 | 99   | 0.193078517913818 | 0.943 |
| 114          | lenet      | 115                 | 99.1 | 0.281186401844025 | 0.943 |
| 115          | lenet      | 116                 | 99   | 0.32935157418251  | 0.943 |
| 116          | lenet      | 117                 | 99.2 | 0.222212642431259 | 0.943 |
| 117          | lenet      | 118                 | 99.2 | 0.206674009561539 | 0.943 |
| 118          | lenet      | 119                 | 98.9 | 0.232496410608292 | 0.943 |
| 119          | lenet      | 120                 | 99   | 0.222332626581192 | 0.943 |
| 120          | lenet      | 121                 | 99   | 0.242076739668846 | 0.943 |
| 121          | lenet      | 122                 | 98.6 | 0.379129946231842 | 0.943 |
| 122          | lenet      | 123                 | 99   | 0.24955852329731  | 0.943 |
| 123          | lenet      | 124                 | 98.9 | 0.200477689504623 | 0.943 |
| 124          | lenet      | 125                 | 99.2 | 0.185772687196732 | 0.943 |
| 125          | lenet      | 126                 | 98.9 | 0.251093149185181 | 0.943 |
| 126          | lenet      | 127                 | 98.6 | 0.328094720840454 | 0.943 |
| 127          | lenet      | 128                 | 98.9 | 0.380752891302109 | 0.943 |
| 128          | lenet      | 129                 | 99.2 | 0.28732818365097  | 0.943 |
| 129          | lenet      | 130                 | 99.2 | 0.251024812459946 | 0.943 |
| 130          | lenet      | 131                 | 99.2 | 0.234700724482536 | 0.943 |
| 131          | lenet      | 132                 | 98.7 | 0.368621051311493 | 0.943 |
| 132          | lenet      | 133                 | 99.2 | 0.273328542709351 | 0.943 |
| 133          | lenet      | 134                 | 98.9 | 0.3249172270298   | 0.943 |
| 134          | lenet      | 135                 | 99   | 0.2325589209795   | 0.943 |
| 135          | lenet      | 136                 | 98.8 | 0.295057117938995 | 0.943 |
| 136          | lenet      | 137                 | 99.1 | 0.201177895069122 | 0.943 |
| 137          | lenet      | 138                 | 98.9 | 0.255284786224365 | 0.943 |
| 138          | lenet      | 139                 | 99   | 0.317417204380035 | 0.943 |
| 139          | lenet      | 140                 | 99   | 0.181290209293366 | 0.943 |
| 140          | lenet      | 141                 | 99.3 | 0.236163824796677 | 0.943 |
| 141          | lenet      | 142                 | 98.2 | 0.259167551994324 | 0.943 |
| 142          | lenet      | 143                 | 99.2 | 0.264978140592575 | 0.943 |
| 143          | lenet      | 144                 | 99   | 0.349231779575348 | 0.943 |
| 144          | lenet      | 145                 | 98.9 | 0.225069969892502 | 0.943 |
| 145          | lenet      | 146                 | 98.8 | 0.285434186458588 | 0.943 |
| 146          | lenet      | 147                 | 98.8 | 0.312730193138123 | 0.943 |
| 147          | lenet      | 148                 | 99   | 0.279895007610321 | 0.943 |
| 148          | lenet      | 149                 | 99.2 | 0.226912885904312 | 0.943 |
| 149          | lenet      | 150                 | 99.3 | 0.222084075212479 | 0.943 |


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
