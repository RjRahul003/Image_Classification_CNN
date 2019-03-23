# Image_Classification_CNN
# Objective :-
  1. Classification of fashionable clothes according to their sleeve types using Images of Myntra Fashion Image Dataset .
# Note :-
  1. Image dataset contains 1482 train images and 520 test images .
# Methods :-
# 1.Using the bottleneck features of a pre-trained network: 90% accuracy in a minute
We will use the VGG16 architecture, pre-trained on the ImageNet dataset --a model previously featured on this blog. Because the ImageNet dataset contains several "cat" classes (persian cat, siamese cat...) and many "dog" classes among its total of 1000 classes, this model will already have learned features that are relevant to our classification problem. In fact, it is possible that merely recording the softmax predictions of the model over our data rather than the bottleneck features would be enough to solve our dogs vs. cats classification problem extremely well. However, the method we present here is more likely to generalize well to a broader range of problems, including problems featuring classes absent from ImageNet.
Our strategy will be as follow: we will only instantiate the convolutional part of the model, everything up to the fully-connected layers. We will then run this model on our training and validation data once, recording the output (the "bottleneck features" from th VGG16 model: the last activation maps before the fully-connected layers) in two numpy arrays. Then we will train a small fully-connected model on top of the stored features.
The reason why we are storing the features offline rather than adding our fully-connected model directly on top of a frozen convolutional base and running the whole thing, is computational effiency. Running VGG16 is expensive, especially if you're working on CPU, and we want to only do it once. Note that this prevents us from using data augmentation.

**Accuracy , Loss :-**
1. Train :- 0.9363 , 0.1847
2. Test :- 0.9217 , 0.2475

# 2.Fine-tuning the top layers of a a pre-trained network
To further improve our previous result, we can try to "fine-tune" the last convolutional block of the VGG16 model alongside the top-level classifier. Fine-tuning consist in starting from a trained network, then re-training it on a new dataset using very small weight updates. In our case, this can be done in 3 steps:
1) instantiate the convolutional base of VGG16 and load its weights 2) add our previously defined fully-connected model on top, and load its weights 3) freeze the layers of the VGG16 model up to the last convolutional block
Note that:
1) We choose to only fine-tune the last convolutional block rather than the entire network in order to prevent overfitting, since the entire network would have a very large entropic capacity and thus a strong tendency to overfit. The features learned by low-level convolutional blocks are more general, less abstract than those found higher-up, so it is sensible to keep the first few blocks fixed (more general features) and only fine-tune the last one (more specialized features). 2) Fine-tuning should be done with a very slow learning rate, and typically with the SGD optimizer rather than an adaptative learning rate optimizer such as RMSProp. This is to make sure that the magnitude of the updates stays very small

**Accuracy , Loss :-**
1. Train :- 0.9315 , 0.2049
2. Test :- 0.9241 , 0.2369

**Final Table**
+--------------------------------------------------------------+----------------+------------+---------------+-----------+
|                           Approach                           | Train Accuracy | Train Loss | Test Accuracy | Test Loss |
+--------------------------------------------------------------+----------------+------------+---------------+-----------+
| Using the bottleneck features of a pre-trained VGG16 network |     0.9363     |   0.1847   |     0.9217    |   0.2475  |
|    Fine-tuning the top layers of a a pre-trained network     |     0.9315     |   0.2049   |     0.9241    |   0.2369  |
+--------------------------------------------------------------+----------------+------------+---------------+-----------+
