# Scene-Recognition
Scene Recognition: Mini Project 3 of the Computer Vision Course Offered in Spring 2022 @ Zewail City 


The project is adopted from a similar course in Brown University.
The goal is to classify scenes into 15 different categories using three different methods for scene recognition. This will be achieved by training and testing on the 15-scene database. 

The three scene recognition schemes to be implemented are:

    1) Using tiny images representation (get_tiny_images()) and nearest neighbor classifier (nearest_neighbor_classify()).
    2) Using bag of words representation (build_vocabulary(), get_bags_of_words()) and nearest neighbor classifier.
    3) Using bag of words representation and linear SVM classifier (svm_classify()).


## Bag of Words Implementation:
A better way to characterize an image is using local features. We have already implemented SIFT-like features, but how can we use them for classification?

We can't possibly match every image with the entire dataset to get which class it belongs to, that is extremely not scalable.

A better approach is to use the Bag of words technique, which is:
* Build a vocabulary set of all possible feature descriptors in the training data
* Then for each training data example, build a histogram, describing the distribution of its own feature descriptors across the vocabulary
* This histogram is then a **feature vector representing the image itself**! Then, we can use simple classification techniques like SVMs and KNN. 

However, if we include all feature descriptors, the data will be huge, and this requires exact matches from the test data, which is unlikely. For that, we can instead cluster this vocabulary into a smaller set of vocabulary. We will then build a histogram for each image of what cluster its features descriptors belong to

The following functions are used in our implementation 

    1) get_tiny_images(): In this function, the images are rescaled into small sizes (16x16). The functions is used 
    to represent the whole images with less amount of information by keeping the low frequencies only and getting 
    rid of the high frequencies.
    
    2) build_vocabulary(): In this function, we build the vocabulary bag that will compare the feature vector and build the histogram.
    
    3) get_bags_of_words(): In this function, feature vectors are extracted from each image in our dataset and compare it with other 
    generated vocabularies to calculate the histogram of the image features.
    
    4) svm_classify(): In this function, linear support vector machine is used to model the data by training on the data and 
    fitting the test data to classify the images according to their feature vectors.
     
    5) nearest_neighbor_classify(): In this function, KNN is used as a classifier to model the data by training on the data 
    and fitting the test data to classify the images according to their feature vectors.

## Results

### Tiny Images, with SVM
Accuracy = 12%

### Tiny Images, with KNN
Accuracy = 21%

### Bag of words, with KNN (k = 3)
Accuracy = 54%

### Bag of words, with SVM
Accuracy = 63%

This is by far the best accuracy achieved. Files containing the confusion matrix and detailed summary of the model performance are attached with the project deliverables
