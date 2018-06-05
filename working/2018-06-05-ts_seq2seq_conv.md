title: Time Series Forecasting with Convolutional Neural Networks 

**Note**: if you're interested in building a simple WaveNet-style CNN time series model yourself using keras, check out the [accompanying notebook - ADD LINK]() that I've posted on github. For an introductory look at high-dimensional time series forecasting with neural networks, you can read my previous [blog post -- ADD LINK]().

Using data from the past to try to get a glimpse into the future has been around since humans have been, and should only become increasingly prevalent as computational and data resources expand. Companies can use forecasting methods to anticipate trends that are core to their business, improving their decision making and resource allocation. Examples abound: grocery chains and online retailers predict product demand for inventory stocking, popular websites predict page visits to manage server demand, and rideshare apps predict trip volume by area to distribute their drivers more effectively.

  
  

![WaveNet](/images/ts_conv/WaveNet_gif.gif)



![dilated_conv](/images/ts_conv/WaveNet_causalconv.png)


![dilated_conv](/images/ts_conv/WaveNet_dilatedconv.png)

![ts_preds](/images/ts_conv/conv_preds.png)

