\# Model overview



This document summarises the main modelling components of the project and where their results are stored.



\## Forecasting models



The primary forecasting task is to predict the next day log return of the S and P five hundred index.



Data representation:

• sliding windows of thirty trading days

• features include log returns, moving averages, rolling volatility and aggregated news sentiment



Models implemented:

• simple baselines (naive last return, mean of last five returns)

• linear regression on flattened windows

• LSTM based sequence model



The resulting comparison table is stored in:



`05\_Results/model\_comparison\_lstm\_vs\_baselines.csv`



This file contains validation and test mean absolute error, root mean squared error and directional accuracy for each model.



\## Generative model



A variational autoencoder (VAE) is trained on sequences of daily log returns with a window length of thirty days.



Architecture:

• encoder and decoder as fully connected networks

• latent space of dimension twelve

• hidden layer size one hundred twenty eight



The best model weights are stored at:



`04\_Src/models/checkpoints/returns\_vae.pt`



Synthetic return windows sampled from the prior and from specific reference windows are saved in:



`05\_Results/vae\_generated\_return\_windows.csv`

`05\_Results/vae\_example\_scenarios\_from\_ref\_window.csv`



Training and validation loss curves for the VAE are saved as:



`05\_Results/vae\_train\_val\_loss.png`



\## Explainable model



For explainability, a random forest regression model is trained on a daily tabular dataset where features describe day t and the target is the next day log return.



The tabular splits are stored at:



`02\_Data/Processed/sp500\_tabular\_train.csv`

`02\_Data/Processed/sp500\_tabular\_val.csv`

`02\_Data/Processed/sp500\_tabular\_test.csv`



Global SHAP importance values are stored at:



`05\_Results/rf\_shap\_global\_importance.csv`



An example local explanation for a specific validation date is stored at:



`05\_Results/rf\_shap\_example\_local\_explanation\_script.csv`

with metadata in:

`05\_Results/rf\_shap\_example\_meta.json`



These components will later be wired into the Streamlit dashboard via the service layer.



