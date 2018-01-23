# Neural Art

This algorithm is for artistic style transfer.    
Original paper: https://arxiv.org/abs/1508.06576

If we understand the mechanism of painting by an artist it goes like:    
The artist has a scene in mind which he wants to paint, he also thinks of a style in his brain by which he wants to paint the scene on canvas.

If we make the same analogy for this algorithm, the artist is the computer,    
the brain is the neural network and the style is computed by "Gram matrices".

The gram matrices compute a good measure of pairwise statistics between the channels of a feature maps yielding a representation of the style.    
Gram matrices: http://mlwiki.org/index.php/Gram_Matrices

In this algorithm we have a content image which we want to paint according to a style image.  
So our initial canvas is some random noise sampled from a gaussian. Then we input the noise image through the "model"   
then we compute the feature maps and gram matrices at one layer or at multiple layers.

We compare these computed feature maps and gram matrices with the pre computed feature maps of content image and gram matrices  
of style image at those layers. We will compare them using a mean squared error loss function and take the gradient of the  
loss function w.r.t the input image. Now just perform gradient descent on the input image.

Aditionally adding the total variational denoising loss reduces some of the noise in the generated images.  
https://en.m.wikipedia.org/wiki/Total_variation_denoising

The model used in this code is the pre trained VGG16 model. The code is written in TensorFlow and Python.  
Perform the iterations number of times for better results. Again, my CPU did not have enough compute.
