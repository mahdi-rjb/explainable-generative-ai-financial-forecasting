\# Thesis Scope – Minimum Viable Thesis (MVT)



\## 1. Project Title



\*\*Explainable Generative AI Dashboard for Real-Time Financial Forecasting and Decision Support\*\*



\## 2. Objective (What this system does)



The goal of this thesis is to build an end-to-end prototype that:



1\. Forecasts the \*\*next-day log return\*\* of the S\&P 500 index using historical price data and news sentiment.

2\. Generates \*\*plausible future scenarios\*\* for the S\&P 500 using a generative model (VAE).

3\. Provides \*\*explainable forecasts\*\* via SHAP, so that users can see \*why\* the model predicts a certain movement.

4\. Presents predictions, explanations, and scenarios in an \*\*interactive Streamlit dashboard\*\* with pseudo–real-time updates.



This is a \*\*prototype\*\* focused on methodological clarity and explainability, not on high-frequency trading or production deployment.



---



\## 3. Data Scope



\### 3.1 Market Data



\- \*\*Asset:\*\* S\&P 500 index (ticker: `^GSPC`)

\- \*\*Frequency:\*\* Daily

\- \*\*Period:\*\* From 2020-01-01 to the most recent available date

\- \*\*Fields:\*\* Open, High, Low, Close, Adjusted Close, Volume



From this, the following are derived:



\- Daily \*\*log returns\*\*:

&nbsp; \\\[

&nbsp; r\_{t} = \\log\\left(\\frac{Close\_{t}}{Close\_{t-1}}\\right)

&nbsp; \\]

\- Simple technical indicators such as:

&nbsp; - Moving averages (e.g. 5-day, 20-day)

&nbsp; - Rolling volatility over a fixed window



\### 3.2 News Data



\- \*\*Source:\*\* A programmatic news API or dataset (e.g., financial news headlines).

\- \*\*Content:\*\* Headlines and timestamps related to global financial markets / S\&P 500.

\- \*\*Granularity:\*\* Article-/headline-level, with publication timestamp.



From this, the following are derived:



\- \*\*Headline sentiment scores\*\* using a lexicon-based method (e.g. VADER).

\- \*\*Daily aggregated sentiment features\*\*, e.g.:

&nbsp; - Mean sentiment per day

&nbsp; - Number of articles per day

&nbsp; - Counts of positive vs. negative headlines



---



\## 4. Prediction Task



\### 4.1 Main Target



\- \*\*Target variable:\*\* Next-day log return of the S\&P 500 index.

\- \*\*Formal definition:\*\*

&nbsp; \\\[

&nbsp; y\_{t+1} = r\_{t+1} = \\log\\left(\\frac{Close\_{t+1}}{Close\_{t}}\\right)

&nbsp; \\]



\### 4.2 Input Features



For each trading day \\( t \\), input features will include:



\- Recent \*\*price-based features\*\*:

&nbsp; - Past \\( N \\) days of returns (e.g. last 30 days)

&nbsp; - Technical indicators (moving averages, volatility, etc.)

\- Recent \*\*news-based features\*\*:

&nbsp; - Daily aggregated sentiment (mean sentiment)

&nbsp; - News volume (number of articles)



\### 4.3 Forecasting Setup



\- \*\*Input window length:\*\* 30 previous trading days (configurable).

\- \*\*Forecast horizon:\*\* 1 day ahead (\\( t+1 \\)).

\- \*\*Prediction type:\*\* Regression (real-valued return).

\- \*\*Evaluation also includes:\*\* Directional accuracy (whether the sign of the return was predicted correctly).



---



\## 5. Models in Scope



\### 5.1 Baseline Models



Used for comparison:



\- Naive “last value” baseline (predict next-day return = today’s return).

\- Simple rolling mean baseline (e.g., mean of last 5 days).

\- Linear regression and/or simple tree-based model on tabular features.



\### 5.2 Forecasting Model (Non-generative)



\- Sequence model (e.g. LSTM or GRU) that:

&nbsp; - Inputs: sequences of past 30 days of features.

&nbsp; - Output: next-day return.

\- Trained on:

&nbsp; - Training period: approximately 2020–2022.

&nbsp; - Validation period: approximately 2023.

&nbsp; - Test period: 2024 onward.



\### 5.3 Generative Model (VAE)



\- \*\*Model type:\*\* Variational Autoencoder (VAE) for time series.

\- \*\*Input:\*\* Historical windows of returns.

\- \*\*Output:\*\* Reconstructed or sampled future return trajectories.

\- \*\*Use case:\*\* Generate multiple plausible future scenarios (e.g., next 5–10 days) starting from a recent historical window.



---



\## 6. Explainability (XAI)



\- \*\*Primary method:\*\* SHAP (SHapley Additive exPlanations).

\- \*\*Application:\*\*

&nbsp; - Mainly to a tree-based or tabular model trained on daily aggregated features.

&nbsp; - Used to provide:

&nbsp;   - \*\*Global explanations:\*\* which features are generally most important.

&nbsp;   - \*\*Local explanations:\*\* why the model made a specific prediction on a specific date.

\- Output will be visualized via:

&nbsp; - SHAP summary plots.

&nbsp; - Local bar/waterfall charts highlighting top contributing features.



---



\## 7. Dashboard (Streamlit)



\- \*\*Framework:\*\* Streamlit.

\- \*\*Core views:\*\*

&nbsp; 1. \*\*Historical and forecast view\*\*

&nbsp;    - Historical S\&P 500 price and returns.

&nbsp;    - Model predictions vs actual values on the test set.

&nbsp; 2. \*\*Explainability view\*\*

&nbsp;    - SHAP-based explanation for the latest prediction.

&nbsp;    - Feature importance and contributions.

&nbsp; 3. \*\*Scenario simulation view\*\*

&nbsp;    - VAE-generated scenarios for future returns/price paths.

\- \*\*Real-time mode:\*\* Pseudo–real-time

&nbsp; - A script updates data and re-runs inference (e.g. hourly or on demand).

&nbsp; - The dashboard refreshes from the latest available CSV/model outputs.



---



\## 8. Out of Scope (Not included in the MVT)



To keep the project feasible within 10 weeks, the following are \*\*explicitly out of scope\*\* for the Minimum Viable Thesis:



\- Multiple asset classes beyond S\&P 500 (e.g. DAX, FX, crypto).

\- True high-frequency data (intraday or tick-level data).

\- Social media sentiment (e.g. Twitter, Reddit).

\- Advanced transformer-based forecasting models (e.g. Temporal Fusion Transformer) as a core requirement.

\- Production-grade deployment (Docker/Kubernetes, cloud scaling).

\- Complex portfolio optimization and risk-parity strategies.



Some of these items (e.g. DAX index, transformer model, LIME for XAI) may be discussed as \*\*future work\*\* or optional enhancements if time allows.



---



\## 9. Success Criteria for the MVT



The MVT will be considered successful if:



1\. A reproducible data pipeline exists for:

&nbsp;  - S\&P 500 daily data.

&nbsp;  - News sentiment features.

2\. At least one forecasting model outperforms naive baselines on the test set.

3\. A VAE-based generative model can produce realistic return trajectories.

4\. SHAP-based explanations are available for the predictive model(s).

5\. A Streamlit dashboard integrates:

&nbsp;  - Forecasts,

&nbsp;  - Explanations,

&nbsp;  - Scenario simulations,

&nbsp;  - With pseudo–real-time updates.



