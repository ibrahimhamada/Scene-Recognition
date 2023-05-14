# Scene-Recognition
Scene Recognition: Mini Project 3 of the Computer Vision Course Offered in Spring 2022 @ Zewail City 


The project is adopted from a similar course in Brown University.
The goal is to classify scenes into 15 different categories using three different methods for scene recognition. This will be achieved by training and testing on the 15-scene database. 

The three scene recognition schemes to be implemented are:

    1) Using tiny images representation (get_tiny_images()) and nearest neighbor classifier (nearest_neighbor_classify()).
    2) Using bag of words representation (build_vocabulary(), get_bags_of_words()) and nearest neighbor classifier.
    3) Using bag of words representation and linear SVM classifier (svm_classify()).


## Implementation:
The following functions are used in our implementation 

    1) get_tiny_images(): In this function, the images are rescaled into small sizes (16x16). The functions is used to represent 
    the whole images with less amount of information by keeping the low frequencies only and getting rid of the high frequencies.
    
    2) build_vocabulary(): In this function, we build the vocabulary bag that will compare the feature vector wit to build the histogram.
    
    3) get_bags_of_words(): In this function, feature vectors are extracted from each image in our dataset and compare it with other 
    generated vocabularies to calculate the histogram of the image features.
    
    4) svm_classify(): In this function, linear support vector machine is used to model the data by training on the data and 
    fitting the test data to classify the images according to their feature vectors.
     
    5) nearest_neighbor_classify(): In this function, KNN is used as a classifier to model the data by training on the data 
    and fitting the test data to classify the images according to their feature vectors.

