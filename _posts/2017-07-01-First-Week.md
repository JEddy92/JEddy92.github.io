---
layout: post
title: Week 1 At Metis Bootcamp - EDA on the MTA, Crazy Turnstiles, & Perfectionism
---

My first week at Metis Bootcamp has flown by, and so far the biggest challenge has been getting this blog set up (just kidding, but it definitely took me much longer than it should have). This week's focus was getting up to speed on several of Python's linchpin data manipulation and analysis packages like numpy, pandas, and matplotlib, with our workflow anchored by a project on MTA subway traffic data. 


"Exploratory Data Analysis" (EDA) is a catch-all description of the process, meaning pretty much any studying and visualization of a dataset in order to extract meaning from it up until the the point of doing extensive statistical modeling and/or hypothesis testing. As an example of an end result, the chart below summarizes traffic trends by day of the week for the most frequented stations  

![plot1](/images/Line_Volume.png)

But before getting there, you usually have to spend some time battling the data to get it into a usable format with meaningful information. Here's a neat visual guide of how it's done: 

![gif](/images/cat_comp.gif)

Just imagine that the cat is doing a bunch of pandas groupby operations and screening out nonsensical data in a frenzy. True fact, this is what the cat is writing:

<pre>
  <code class="python">
df_MTA_byDate = df_MTA.groupby(['C/A','UNIT','SCP','STATION','DATE','DAY_OF_WEEK']) \
                .ENTRIES.agg({'MIN_ENTRIES':'min'})
df_MTA_byDate = df_MTA_byDate.reset_index()

df_MTA_byDate['DAILY_ENTRIES'] = df_MTA_byDate.groupby(['C/A','UNIT','SCP','STATION']) \
                                 .MIN_ENTRIES.diff().shift(-1)
df_MTA_byDate.drop('MIN_ENTRIES',axis=1,inplace=True) 
df_MTA_byDate.loc[df_MTA_byDate['DAILY_ENTRIES'] < 0, 'DAILY_ENTRIES'] = np.nan
df_MTA_byDate.loc[df_MTA_byDate['DAILY_ENTRIES'] > 100000, 'DAILY_ENTRIES'] = np.nan
  </code>
</pre>

The idea here is to convert the raw turnstile entry data, which are cumulative counts, into daily entry values that will actually be useful for analysis. You can do this by taking the difference between cumulative entries at the start of the next day and cumulative entries at the start of the current day. But of course the data is not your friend, so sometimes the cumulative counts randomly reset or jump forward by hundreds of thousands. Either the MTA has found a way to demolish the space-time continuum (plausible, would explain the wait times and the current "state of emergency"), or the data... isn't quite accurate. So you want to NA out data points that are clearly invalid, and it takes some time to identify exactly all the ways that things can go wrong and what figure out what cutoffs you should use.

People in my cohort have referred to these cases as "crazy turnstiles", and have pointed out that you could do an entire analysis just on how often turnstiles are crazy, in what way they tend to be crazy, and whether the same turnstiles tend to be consistently crazy. 
