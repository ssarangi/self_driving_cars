# Background
Recently I decided to join the Udacity's Self Driving Car Nanodegree. With all the apprehensions about cost and whether it will be worth it or not I finally decided to take
the jump into this field. I had a few other things going on and I wasn't sure whether I could even devote the time to do it. However, after much speculation however I convinced
myself that this was the way to go and enrolled in the class. So this blog post is serving as a writeup for the first project.

As soon as you start the course, this project becomes due in a week. So you have to jump right in and get the project working asap in order to submit it.


# Finding Lane Lines on the Road
The first project from this course was lane line detection from images and videos. Udacity provided test images and videos for testing the overall pipeline out.
The requirements were to use OpenCV with Python and IPython Notebooks to demonstrate the solution. This turned out to be quite an interesting project overall for
me. It had the right amount of technicality along with some intuition needed to solve the problem.

# Terminology:

* ROI (Region of Interest): Defines the region in an image where we are likely to find the lane lines.
* Canny: Canny Edge Detection
* Hough: Probabilistic Hough Transform

# Assumptions:
* The region of interest is always assumed to be at the center which essentially
means that it's always assumed that the car starts between the 2 lanes.
* Filtering currently is a very naive, in which all lines whose slope falls between -0.5 to 0.5 are ignored.

# Results:
Currently, the same pipeline works for both white and yellow lines as well as the optional challenge with just changing parameters for ROI and hough. 


# Algorithm

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that identifies the lane lines on the road.
* Once the lane lines are identified, make it a single straight line which is uniform across frames.
* Identify outliers and make sure that they don't break the overall algorithm


[//]: # (Image References)

[grayscale_with_noise]: ./writeup/grayscale.jpg "Grayscale With Noise"

---

### Reflection

#### 1. Pipeline

The problem consisted of a combination of images as well as videos. The idea was first to first describe a pipeline which works on both images and videos. The images provided
were pretty straightforward and didn't present too much of a problem. However, the videos turned out to be trickier than I thought to deal with. Below I describe the pipeline
which I used to describe the problem.

##### Gaussian Blur:
The first step of the algorithm is to do a slight blur on the image. The most common form of this is the Gaussian Blur. Blurring the image reduces the noise in the image
which makes it easier for Canny edge detection to find the edges 

![alt text][image1]

The images above show the problem when using 


#### 2. Approach & Debugging

#### 3. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ... 

Another shortcoming could be ...


#### 4. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...