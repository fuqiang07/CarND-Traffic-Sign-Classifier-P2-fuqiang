# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[DataVisualization]: ./examples/DataVisualization.png "DataVisualization"
[preprocess]: ./examples/preprocess.png "preprocess"
[DataAugmentation]: ./examples/DataAugmentation.png "DataAugmentation"
[DataOriginal]: ./examples/DataOriginal.png "DataOriginal"
[DataAugmented]: ./examples/DataAugmented.png "DataAugmented"
[ModifiedLeNetModel]: ./examples/ModifiedLeNetModel.jpeg "ModifiedLeNetModel"
[MyImage]: ./examples/MyImage.png "MyImage"
[MyImageAccuracy]: ./examples/MyImageAccuracy.png "MyImageAccuracy"
[MyImageAnalysis]: ./examples/MyImageAnalysis.png "MyImageAnalysis"

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/fuqiang07/CarND-Traffic-Sign-Classifier-Project-fuqiang/blob/master/Traffic_Sign_Classifier_fuqiang.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799.
* The size of the validation set is 4410.
* The size of test set is 12630.
* The shape of a traffic sign image is (32, 32, 3).
* The number of unique classes/labels in the data set is 43.

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the distribution of the training, validation and test data. To view the distribution more conviently, I adjust the number of validation set by mulpiplying by 8 and the number of test set by multiplying by 2.5. From the chart below, we can see that:
1) the number of data for each class (43 in total) is not the same, ranging from a few hundreds to two thousands
2) the distribution of training, validation and test data are consisent.

![alt text][DataVisualization]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

There are two steps for this data preprocessing:
1) Converting the RGB images to grayscale. This works well for this task, since the color plays much less important role compated with the shape and the edge for signal classification. Moreover, this helps reduces traning time.
2) Normalizing the Gray images within the range (-1, 1), which can be realized by the simple code img = (img - 128)/128. As mentioned in Prof. Andrew Ng's deep learning course, the normalization is helpful for the learning to converge fast.

Here is an example of a traffic sign image before and after data preprocessing.

![alt text][preprocess]


I decided to generate additional data because data augmentation is helpful to solve the problem of overfitting. Moreover, as mentioned above, the number of data for each class is not equal, even with big difference. Data augmentation can make the data distribution more uniform.

To add more data to the the data set, I used the following techniques, translation, scale, warp and brightness, because these techniques can provide more examples for the deep networks.

Here is an example of the original image and the augmented image:

![alt text][DataAugmentation]

The difference between the original data set and the augmented data set is shown as below

Original Data | Augmented Data
 :---:  | :---:
![alt text][DataOriginal] | ![alt text][DataAugmented]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model is the modified LeNet network as shown in the figure below
![alt text][ModifiedLeNetModel]

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the method presented in the lessons and the following parameters:
* Optimizer: adam optimizer with default settings
* Batch size: 128
* Epochs: 20
* Learning rate: 0.001
* Keep Prob rate for Drop Out: 0.3


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.999
* validation set accuracy of 0.970
* test set accuracy of 0.950

I choose an iterative approach. In this way, I can realize one solution in shortest time and improve the accuracy based on the previous iteration result.
  1. For the initial step, I just did the data preprocessing and chose the LeNet model learned from the lesson. After implementation, I got a result with training accuracy 0.995 and validataion accuracy 0.929, which implis the overfitting problem. Moreover, it took me around 534.2s to complete the training process.
  2. Since my computer is not so powerful, I would like to try to shorten the training time by optimizing the model architecture. Accoding to the Deep Learning course of Porf. Andrew Ng, we can convert the fully-connected layers into convolutional layers to save training time. However, it seems that the time cost is almost the same as before. Converting fully connected networks to convolutional networks may be helpful for the case of convolution implementation of sliding windows, but is not helpful in this case.
  3. The most efficient way to solve the problem of overfitting is regularization, including drop out. After using the technique of drop out, I updated the result as training accuracy 0.995 and validataion accuracy 0.954. It seems that this technique works well, but there is still space to improve.
  4. Then I use data augmentation to get more uniform data for the training, resulting in training accuracy 0.991 and validataion accuracy 0.962. The validation accuray is improved a little bit, but not so much. One reason is that I just augment the training set, leadding to different distribution between tranining set and validation set.
  5. Since the problem is still overfitting, I try a modified LeNet Networks, with training accuracy 0.999 and validataion accuracy 0.970. Moreover, the test accurary is improved from 0.935 to 0.950.
  6. I think I can try more advanced networks but with limited time. I will keep trying other techniques when I am free in the future.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are nine German traffic signs that I found on the web:

![alt text][MyImage]

The only image that I think it might be difficult to classify is the eighth one because it is not so clear and warped.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:
![alt text][MyImageAccuracy]

The model was able to correctly guess 9 of the 9 traffic signs, which gives an accuracy of 100%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.
![alt text][MyImageAnalysis]

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


