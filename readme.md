# WQU Machine Learning in Finance Group Work Project

Using dollar bars of Bitcoin data from the Bitstamp exchange, a trading strategy was created that would take long and short positions on the basis of momentum and driven by technical indicators such exponential moving average, relative strength index, and commodity channel index.  The model is then fitted using decision tree, random forest, logistic regression, and support vector machine.  Each model is then tuned and evaluated for accuracy and robustness.  Finally, return and risk metrics are generated for the trading system to quantify expectations and to evaluate if the system should be used. 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them
```

pip install numpy==1.14.5 pandas==0.23.1 tensorflow==1.8.0 statsmodels==0.9.0 scikit-image==0.14.0 scikit-learn==0.19.1 Keras==2.2.0 kmeans==1.0.2 jupyterlab==0.32.1 ipython==6.4.0 ipywidgets==7.2.1 ipython-genutils==0.2.0 ipykernel==4.8.2 h5py==2.8.0 Faker==0.9.1 Babel==2.6.0 matplotlib==2.2.2 nodejs

pip install https://github.com/matplotlib/mpl_finance/archive/master.zip
pip install backtrader
pip install pyfolio
pip install seaborn
pip install scipy
pip install statsmodels
pip install theano
pip install pymc3
pip install sklearn
```


## Running the notebooks

- Start a Command Prompt on Windows or Terminal on MacOs
- Browse to the folder you stored the .ipynb file then run 
```
jupyter notebook
```

## Solution overview

Below is an overview of the different steps performed in this project

* Read tick data from bitstampUSD 
  * time_stamp (in unix time)
  * volume (BTC) - Volume of BTC transacted
  * price (Currency) - Bitcoin price in Currency units
  * data collected over 21 days starts from 28th September 2018 to 18th October 2018

* Create dollar bars from this tick data, since as was discovered earlier, dollar bars have the least serial correlation and closest distribution to a Normal Distribution

* Calculate the returns

* Calculate 20 period volatility as the 20 period std deviation of the log of returns at each bar

* Define a target as 1 if price moves up more than periodVol value, and 0 otherwise

* Feature engineering and Determing input variable
  * Adding technical indicators
  * Standard Scaling the data 

* ML Models
  * ML Models used to fit
    * Decision Tree
    * Random Forest
    * Logistic Regression
    * Support Vector Machine

* Accuracy, Precision, Recall, F-Score, ROC curves and Confusion Matrices are calculated 

* Backtrader and PyFolio are used to backtest the trading system and generate the fund factsheet

## Built With

Python 3.7

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

Hat tip to everyonen below whose code was used in or helped inspire this work

* [BlackArbsCEO](https://github.com/BlackArbsCEO/Adv_Fin_ML_Exercises)
* [Jackal08](https://github.com/Jackal08/Adv_Fin_ML_Exercises)
* [quantopian](https://github.com/quantopian/pyfolio)
* [backtrader](https://github.com/backtrader/backtrader)
* [Rachnog](https://github.com/Rachnog/Deep-Trading)

