# Deep convolutional GAN

Generative adversarial networks have become very popular in the recent times.

These kind of generative models assume that the input data follows an implicit latent
distribution, and directly sample the latent representation from that distribution.

The GANs contain two networks: Generator and the Discriminator.
Generator generates fake images that the discriminator tries to classify as real/fake.
It eventually becomes a minimax game b/w these two.

The code is in Keras using Tensorflow backend in Python 3.5.2
The results are not as good since i have don't have a GPU for training longer iterations.



But here are some tips on how to train a GAN:

> Normalize the inputs.

> Keep the last layer output of generator as 'tanh'. (-1, 1)

> Different batches for real and fake for the generator ~ Counter intuitive but works (:

> Avoid sparse gradients and use:
   > LeakyRelu
   > Avg pooling / Conv2d + stride
   > PixelShuffle / Conv2dTranspose + stride

> Soft/ Noisy labels for the Discriminator.

> Hybrid models: KL divergence + GAN or Var auto-encoder + GAN

> Some stability tricks from Reinforcement learning

> optimizer: ADAM rules !

> Tracks of failure are:
   > Disc loss reduces to 0 quickly.
   > If Gen loss decreases steadily it is fooling the Disc with garbage.

> Labeled classification helps.

> Add some noise to inputs and decay noise over time.

> Train Disc K times per training of Gen.


Some of these methods were not used in my code, but try them for better results!