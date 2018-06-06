title: Time Series Forecasting with Convolutional Neural Networks - a Look at WaveNet

**Note**: if you're interested in building a simple WaveNet-style CNN time series model yourself using keras, check out the [accompanying notebook - ADD LINK]() that I've posted on github. For an introductory look at high-dimensional time series forecasting with neural networks, you can read my previous [blog post -- ADD LINK]().

If you're reading this blog, it's likely that you're familiar with some of the classic applications of convolutional neural networks to tasks like image recognition and text classification. Convolutions are a very natural and powerful tool for capturing spacially invariant patterns. It matters little *where* in the image whiskers occur when we're identifying a cat. Similarly, in classifying a document as a court case transcript, the *presence* of legal jargon phrases matters much more to us than their *position* in the document. But what about temporal patterns? By a similar token, might there be recurring patterns like weekly cyclicality and certain autocorrelation structures that convolutions are well-suited to model?

The answer is a resounding yes! It turns out that there are specialized convolutional architectures that perform quite well at time series prediction tasks. In this post I'll discuss one in particular, [DeepMind's WaveNet](https://deepmind.com/blog/wavenet-generative-model-raw-audio/), which was designed to advance the state-of-the-art for text-to-speech systems. 

  

![WaveNet](/images/ts_conv/WaveNet_gif.gif)



![dilated_conv](/images/ts_conv/WaveNet_causalconv.png)


![dilated_conv](/images/ts_conv/WaveNet_dilatedconv.png)

![ts_preds](/images/ts_conv/conv_preds.png)

