title: Forecasting with Neural Networks - An Introduction to Sequence-to-Sequence Modeling Of Time Series 

**Note**: if you're interested in building seq2seq time series models yourself using keras, check out the [introductory notebook](https://github.com/JEddy92/TimeSeries_Seq2Seq/blob/master/notebooks/TS_Seq2Seq_Intro.ipynb) that I've posted on github.

Using data from the past to try to get a glimpse into the future has been around since humans have been, and should only become increasingly prevalent as computational and data resources expand. Companies can use forecasting methods to anticipate trends that are core to their business, improving their decision making and resource allocation. Examples abound: grocery chains and online retailers predict product demand for inventory stocking, popular websites predict page visits to manage server demand, and rideshare apps predict trip volume by area to distribute their drivers more effectively.

These diverse applications share a common quantitative framework under the umbrella of time series forecasting. Each unit of interest (item, webpage, location) has a regularly measured value (purchases, visits, rides) that changes over time, giving rise to a large collection of time series.  

![random_series](/images/ts_intro/random_series.png)

In traditional time series forecasting, series are often considered on an individual basis, and predictive models are then fit with series-specific parameters. An example of this style is the classic **Autoregressive Integrated Moving Average (ARIMA)** model. Series-specific models can often make quite good predictions, but unfortunately they do not scale well to problems where the number of series to forecast extends to thousands or even hundreds of thousands of series. Additionally, fitting series-specific models fails to capture the expressive general patterns that can be learned from studying many fundamentally related series. From the examples above, we can see that this challenging “high-dimensional” time series setting is faced by many companies.

Luckily, multi-step time series forecasting can be expressed as a sequence-to-sequence supervised prediction problem, a framework amenable to modern neural network models. At the cost of added complexity in constructing and tuning the model, it's possible to capture the entire predictive problem across all the series with one model. Since neural networks are natural feature learners, it's also possible to take a minimalistic approach to feature engineering when preparing the model. And when exogenous variables do need to be integrated into the model (e.g. product category, website language, day of week, etc.), it's simple due to the flexibility of neural network architectures. If you're not already sold on the potential power of this approach, check out the [DeepAR](https://arxiv.org/pdf/1704.04110.pdf) model that Amazon uses to forecast demand across a massive quantity of products.     

So how does seq2seq work exactly? Let's first consider it in its original application domain as described by this [2014 paper](https://arxiv.org/abs/1409.3215), machine translation.   

![architecture_lang](/images/ts_intro/seq2seq_lang.png)

Encoder - decoder framework

![architecture_ts](/images/ts_intro/seq2seq_ts.png)

In translation we condition on the entirety of an input sentence to generate a corresponding output sentence. Similarly, in a time series problem we can condition on the entire history of a series in order to make predictions about the future. The encoder’s final hidden state then becomes the decoder’s initial hidden state, and this vector serves as a learned representation of history.  
