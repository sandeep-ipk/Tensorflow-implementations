# Ensemble learning

This code creates an ensemble of 5 Convnets for classifying the MNIST dataset. The ensemble works by averaging the predicted class-labels of the 5 individual Convnets. This results in  improved classification accuracy on the test-set.

However, the ensemble did not always perform better than the individual neural networks, which sometimes classified images correctly while the ensemble misclassified those images. This might be mainly because the dataset is not that big, which resulted in random results. By performing 2000 iterations per network is enough to get a high classification accuracy per network. If the network was trained on a relatively bigger dataset, then ensemble works very well.

The form of ensemble learning used here is called bagging (or Bootstrap Aggregating), which is mainly useful for avoiding overfitting the data.