title: Time Series Forecasting with Convolutional Neural Networks - a Look at WaveNet

**Note**: if you're interested in learning more and building a simple WaveNet-style CNN time series model yourself using keras, check out the [accompanying notebook - ADD LINK]() that I've posted on github. For an introductory look at high-dimensional time series forecasting with neural networks, you can read my previous [blog post -- ADD LINK]().

If you're reading this blog, it's likely that you're familiar with some of the classic applications of convolutional neural networks to tasks like image recognition and text classification. Convolutions are a very natural and powerful tool for capturing spacially invariant patterns. It matters little *where* in the image whiskers occur when we're identifying a cat. Similarly, in classifying a document as a court case transcript, the *presence* of legal jargon phrases matters much more to us than their *position* in the document. But what about temporal patterns? By a similar token, might there be recurring patterns like weekly cyclicality and certain autocorrelation structures that convolutions are well-suited to model?

The answer is a resounding yes! It turns out that there are specialized convolutional architectures that perform quite well at time series prediction tasks. In this post I'll discuss one in particular, [DeepMind's WaveNet](https://deepmind.com/blog/wavenet-generative-model-raw-audio/), which was designed to advance the state of the art for text-to-speech systems. The WaveNet model's architecture allows it to exploit the efficiencies of convolution layers while simultaneously alleviating the challenge of learning long-term dependencies across a large number of timesteps (1000+). The latter is a frequent pain point for recurrent neural networks, even those that include some long-term memory mechanism like LSTMs. 

At the heart of WaveNet's magic is the **dilated causal convolution layer**, which allows it to properly treat temporal order and handle long-term dependencies without an explosion in model complexity. Here's a nice visualization of its structure from DeepMind's post:   

![WaveNet](/images/ts_conv/WaveNet_gif.gif)

The visual is helpful, but let's try to gain more insight by breaking this down. First of all, what makes a convolution *causal*? In a traditional 1-dimensional convolution layer, we slide a filter of weights across an input series, sequentially applying it to (usually overlapping) regions of the series. But when we're using the history of a time series to predict its future, we have to be careful. As we form layers that eventually connect input steps to outputs, we must make sure that inputs do not influence output steps that proceed them in time. Otherwise, we would be using the future to predict the past, so our model would be cheating!       

To ensure that we don't cheat in this way, we adjust our convolution design to explicitly prohibit the future from influencing the past. In other words, we only allow inputs to connect to future time step outputs in a **causal** structure, as pictured below in a visualization from the WaveNet paper. In practice, this causal 1D structure is easy to implement by shifting traditional convolutional outputs by a number of timesteps. Keras handles it via setting ```padding = 'causal'```.


![dilated_conv](/images/ts_conv/WaveNet_causalconv.png)


![dilated_conv](/images/ts_conv/WaveNet_dilatedconv.png)

![ts_preds](/images/ts_conv/conv_preds.png)

