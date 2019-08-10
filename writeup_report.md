# **Behavioral Cloning**

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./examples/exp_center.jpg "Center Lane Drive"
[image2]: ./examples/left.jpg "Recover From Left"
[image3]: ./examples/center.jpg "Center"
[image4]: ./examples/right.jpg "Recover From Right"
[image5]: ./examples/normal.jpg "Normal Image"
[image6]: ./examples/flipped.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The model I used is an architecture published by NVIDIA Autonomous Vehicle Team, ["End to End Learning for Self-Driving Cars"](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layer in order to reduce overfitting (model.py line 82).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 61, 62). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 87).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and smooth cornering.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was:

* Firstly use a simple model to check if preprocessing is correct
* Then use a model created for a similar purpose: Behavior Cloning for Self Driving Cars
* If it underfits, I increased number of epochs
* If it overfits, I augmented dataset, used dropout layer and decrease number of epochs
* Then I test the model on simulator to check if it is correct

The final model I used is convolution neural network model of NVIDIA's End to End Learning for Self Driving Cars paper. I thought this model might be appropriate because it is created for the same purpose.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set.(model.py line 88). I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I add a Dropout Layer of 0.5.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track when cornering. To improve the driving behavior in these cases, I changed dropout rate to 0.2.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 68-85) consisted of a convolution neural network with the following layers and layer sizes

It starts with Normalization layer, then 5 Convolution Layers followed by 4 Fully-Connected Layers:

* Convolution layer: 24 Filters of 5x5 with RELU activation
* Convolution layer: 36 Filters of 5x5 with RELU activation
* Convolution layer: 48 Filters of 5x5 with RELU activation
* Convolution layer: 64 Filters of 3x3 with RELU activation
* Convolution layer: 64 Filters of 3x3 with RELU activation

* Flatten layer: 9600 neurons
* Fully Connected layer: 100 neurons with RELU activation and 0.3 Dropout
* Fully Connected layer: 50 neurons with RELU activation
* Fully Connected layer: 10 neurons with RELU activation

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 70).

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to get away from sides. These images show what a recovery looks like :

![alt text][image2]
![alt text][image3]
![alt text][image4]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would help model to generalize. For example, here is an image that has then been flipped:

![alt text][image5]
![alt text][image6]

After the collection process, I had 50232 number of data points. I then preprocessed this data by normalizing and mean centering (code line 70).


I finally randomly shuffled the data set and put 0.2% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 6 as evidenced by increasing validation error. I used an adam optimizer so that manually training the learning rate wasn't necessary.
