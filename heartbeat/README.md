# Heart beat anomaly detection

In the recent years, I have seen an increase in the applications of data-driven methods in healthcare, especially deep learning methods.

This work aims to classify abnormal heartbeat from the normal heartbeat. This abnormal beat is termed as "heart murmurs". Abnormal heart murmurs in adults are related to defective heart valves. Abnormal heart murmurs in adults may be related to Valve calcification, Endocarditis, Rheumatic fever. 

In this work, I propose a deep learning approach to this problem. I find this behaviour by training a deep convolutional neural network on recorded audio files (in .wav format) of the heartbeats. I use Python and Keras (using TensorFlow backend) for building and training the model. I convert the '.wav' files to time-series using Scipy's 'wavfile' library. 

Then I apply the model to these time-series data of each audio file, I use 8 layers of 1-D convolutional layers to capture the temporal features. The final layer is a densely connected layer classifying the sample as normal or murmur type. The proposed model is then validated on the test dataset, I achieve an accuracy of _92.5%_.