\# Dashboard overview



This document explains the structure and purpose of the Streamlit dashboard used in the thesis.



\## Purpose



The dashboard is a proof of concept for a decision support tool based on:

\* LSTM forecasting of next day S and P five hundred log returns

\* VAE based scenario generation for future return and price paths

\* SHAP based explainability of a random forest benchmark model



It is not a trading system. The focus is transparency, interpretability and a clear separation between forecasting, generative modelling and explanations.



\## Code structure



The main entry point is:



\* `06\_App/streamlit\_app.py`



The app relies on a small service layer:



\* `04\_Src/services/forecasting\_service.py`

&nbsp;   \* loads the trained LSTM model and metadata

&nbsp;   \* reads the latest feature window from `sp500\_features\_daily.csv`

&nbsp;   \* predicts the next day log return



\* `04\_Src/services/generative\_service.py`

&nbsp;   \* loads the trained VAE from `returns\_vae.pt`

&nbsp;   \* reads the most recent thirty day window of log returns

&nbsp;   \* generates synthetic return scenarios around that window



\* `04\_Src/services/explainability\_service.py`

&nbsp;   \* loads the tabular daily dataset for train and validation

&nbsp;   \* trains a random forest regression model for next day returns

&nbsp;   \* computes local SHAP explanations for a selected validation day



The dashboard itself defines three pages:

\* `page\_forecast()` – LSTM forecast and implied next day price

\* `page\_scenarios()` – VAE return and price scenarios

\* `page\_explainability()` – local and global SHAP explanations



