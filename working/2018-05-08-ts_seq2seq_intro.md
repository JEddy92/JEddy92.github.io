title: Forecasting with Neural Networks - An Introduction to Sequence-to-Sequence Modeling Of Time Series 

In traditional time series forecasting, series are often considered on an individual basis, and predictive models are then fit with series-specific parameters (e.g. ARIMA models). 
This style of forecasting does not scale well to problems where the number of series to forecast extends to thousands or even hundreds of thousands of series. 
Additionally, fitting series-specific models fails to capture the expressive general patterns that can be learned from studying many fundamentally related series. 
This “high-dimensional” time series setting is faced by many companies, in forms ranging from store/product demand forecasting (think Amazon) to webpage traffic forecasting (think web advertising).

![random_series](/images/ts_seq2seq_intro/random_series.png)

Luckily, multi-step time series forecasting can be expressed as a sequence-to-sequence supervised prediction problem, a framework amenable to modern neural network models.

one more viz here? illustrating setup?

Intro to seq2seq NN - frequently used for NLP problems like machine translation (original source of the architecture). Encoder - decoder framework

In translation we condition on the entirety of an input sentence to generate a corresponding output sentence. Similarly, in a time series problem we can condition on the entire history of a series in order to make predictions about the future. The encoder’s final hidden state then becomes the decoder’s initial hidden state, and this vector serves as a learned representation of history.  
