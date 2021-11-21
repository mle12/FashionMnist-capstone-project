# FashionMnist-capstone-project
Capstone Project- Fashion MNIST Clothing Classification


 The ability to classify images correctly is particularly useful in the world of e commerce. Computer vision models can be used in similarity search where site visitors are given exact or similar matches to what they have searched for. This in turn can prolong the engagement of visitors and increase sales. Automatic image tagging is also a useful application enabling clothing to be correctly labelled and found on websites with large and extensive inventory.

The task of this project is to develop a Machine Learning model to correctly classify photos of clothing.
The model will be trained on the well known Fashion MNIST dataset. This comprises 60000 photos that will be used for the training dataset and 10000 images for the test dataset. The photos are 28X28 pixel grayscale images of 10 types of clothing belonging to the following categories 

0: T-shirt/top
1: Trouser
2: Pullover
3: Dress
4: Coat
5: Sandal
6: Shirt
7: Sneaker
8: Bag
9: Ankle boot


Conveniently the dataset can be loaded from the tensorflow_datasets module. The dataset comes with training and test partitioning. However I decided to further split the training data to create an additional validation set. The partitioning then is as follows:

Training set :50000 images
Validation set :10000 images
Test Set :10000 images

As this is a computer vision problem my plan was to implement a Convolutional Neural network(CNN). The Keras layers API within Tensorflow is a straightforward way to build the various sections of a CNN architecture. There is a vast number of possible architectures I could have used but I decided to settle with a relatively simple one and then see if I could improve on it.

To reduce overfitting (the inablity to generalise to unseen data) in my model I added a dropout layer. Also as this was a multi class classification problem with 10 outputs or categories of clothing, I used a softmax activation function as the final step of my CNN. The ideal loss function for this type of multi class classification is SparseCatgoricalCrossentropy and I chose to use the Adam optimizer to train the CNN. 

Next I trained the model and used the validation dataset to monitor how the model was learning. 
I decided to use 50 epochs or cycles through the data to make sure the accuracy metric would converge. I then plotted training and validation curves for 'loss' and 'accuracy' to see how the model was learning. The goal was for 'loss' to be as close to 0 as possible and 'accuracy' to be as close to 100 as possible.

The graphs revealed that the initial model did very well on the training data getting around 99% accuracy. However the model did not iperform as well on the validation or test data getting around 91% accuracy. Also the validation accuracy failed to improve after 5 epochs. It looked like the model had low bias but high variance indicating overfitting or lack of generalisation to unseen data.  I then took a batch of 12 examples from the test data and visualised the input image and the predicted labels. This revealed the model only incorrectly classified 1 out of 12 images.

I next decided to use a more complex architecture with the aim of improving accuracy and to reduce overfitting. I settled on the well known ResNet50 architecture. This new model had some 26 million parameters to train versus the initial model with only around 3 million. This meant this model would take longer to train and be more computationally expensive but would it improve the accuracy?

Again I used 50 epochs to train and plotted the same graphs. The ResNet50 model produced a similar accuracy on the training data but slightly worse on the validation and test data at 88% accuracy. It also took 5 times longer to train than the first CNN. It also demonstrated a similar issue of overfitting. The first model would therefore be the better choice out of the two.

How else could accuracy be improved and overfitting reduced? Data augmentation could be explored which involves rotating, cropping, and flipping the images. Different well known CNN architectures could be tried such as AlexNet,VGG16, and InceptionV3. Transfer learning models could be used with parameters derived from training on other datasets. Also the dropout parameter could be modified( I used 0.5 in my models). However pursuing many of these techniques would require time and additional resources. A company must access whether this would be worth pursuing for potentially only a small incremental improvement in accuracy.

Images in the real world will likely be very different to the ones used to train the model. Fashion MNIST only has 70,000 images whereas they are millions of clothing images on the internet. Real world images could have different resolution, be in color and be taken from different angles. The clothing could also be of different style and color. Therefore when the model goes into production it may struggle to generalise to new images.

