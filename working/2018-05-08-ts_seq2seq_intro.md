In traditional time series forecasting, series are often considered on an individual basis, and predictive models are then fit with series-specific parameters (e.g. ARIMA models). 
This style of forecasting does not scale well to problems where the number of series to forecast extends to thousands or even hundreds of thousands of series. 
Additionally, fitting series-specific models fails to capture the expressive general patterns that can be learned from studying many fundamentally related series. 
This “high-dimensional” time series setting is faced by many companies, in forms ranging from store/product demand forecasting (think Amazon) to webpage traffic forecasting (think web advertising).
