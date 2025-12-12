\# Dashboard demo script



This is a short script I can follow when presenting the system live.



1\. Intro (home / title)

&nbsp;  \* Mention that the dashboard combines forecasting, scenario generation and explainability.

&nbsp;  \* Explain that all results are based on S and P five hundred daily data since 2020 plus news sentiment features.



2\. Forecast page

&nbsp;  \* Show the last available date and last close price.

&nbsp;  \* Highlight the predicted next day log return and the implied next day price.

&nbsp;  \* Briefly mention the LSTM inputs: 30 day window, price based features, sentiment aggregates.



3\. Scenarios page

&nbsp;  \* Explain that the VAE is trained on 30 day sequences of log returns.

&nbsp;  \* Show how changing the number of scenarios and noise scale affects the return and price paths.

&nbsp;  \* Emphasise that these are not point forecasts, but plausible future paths conditioned on recent market behaviour.



4\. Explainability page

&nbsp;  \* Select a recent validation date via the date selector.

&nbsp;  \* Point out the difference between true and predicted next day return.

&nbsp;  \* Scroll the local feature contribution table and mention which features drive the prediction.

&nbsp;  \* Show the global SHAP importance plot and connect it to domain intuition

&nbsp;    (for example, recent volatility, recent returns or sentiment levels).



5\. Closing remarks

&nbsp;  \* State that the system demonstrates how generative models and explainability methods

&nbsp;    can be combined into a unified dashboard for financial decision support.

&nbsp;  \* Mention that the architecture is modular and can be extended with other assets or model classes.



