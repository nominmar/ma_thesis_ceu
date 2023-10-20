## Simulation codes for CEU MA thesis

#### "MAIN.ipynb" calls all functions.
#### Thesis text can be found [here](https://github.com/nominmar/MA_thesis_ceu/blob/main/other/thesis.pdf)

### Abstract:

Causal Trees leverage the supervised machine learning algorithm decision trees to estimate heterogeneous treatment effects across data-driven groups in a randomized treatment assignment setting. In my thesis, I modify the Causal Tree estimator by introducing a parameter theta that lets the user control allocation of data into training and estimation subsamples. The estimator implements honest sample splitting by default, which divides the sample into two equal parts: training and estimation subsamples. The new input parameter theta lets the user select the portion of data to be allocated to the estimation subsample. I test the performance of the estimator under various data allocations through Monte-Carlo simulations. 

### Flowchart:
![](https://github.com/nominmar/MA_thesis_ceu/blob/main/other/flowchart.png)
