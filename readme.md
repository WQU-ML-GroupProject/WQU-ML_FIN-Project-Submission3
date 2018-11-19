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
- Browse to the folder you unzipped the .ipynb file.
- Type 
```
jupyter notebook
```

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

Python 3.7

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to everyonen below whose code was used in or helped inspire this work

[BlackArbsCEO](https://github.com/BlackArbsCEO/Adv_Fin_ML_Exercises)

[Jackal08](https://github.com/Jackal08/Adv_Fin_ML_Exercises)

[quantopian](https://github.com/quantopian/pyfolio)

[backtrader](https://github.com/backtrader/backtrader)

[backtrader](https://github.com/backtrader/Deep-Trading)

