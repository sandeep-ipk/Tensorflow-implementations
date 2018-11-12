# Deep dream

Deep dreaming is a very popular algorithm, we can say that it is  
a by-product of Conv-Nets.  
(https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html)

In this code, I use a pre trained model of inception5h for the feature extraction.  
The model extracts features from a inputted image.

Now here is the trick part, since we are using a pre-trained model, we do not want to  
fine tune it for classifying out inputted image, rather we want it to extract feature    
that it thinks are  in the image.  

Now we simply calculate the gradient of these features w.r.t the input image and  
apply gradient-ascent. As simple as that.

What i use in my code is an idea of recursive optimization i.e: first downsample the image  
to a smaller size and then apply the algorithm on that image, now upsample it and again apply the  
algorithm and then upsample the image until it reaches the original size by applying the algorithm.  
Ofcourse, this idea is not mine originally and i found it on the YouTube.  
(https://www.youtube.com/watch?v=ws-ZbiFV1Ms)

I use Tensorflow and Python 3.5.2.
I was not able to perform much iterations as i do not have a GPU,  
performing 50 iterations was more than what my CPU could handle!





