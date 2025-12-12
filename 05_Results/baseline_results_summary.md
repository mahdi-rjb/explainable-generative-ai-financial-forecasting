\# Baseline forecasting results



This document summarises the behaviour of simple baseline models for predicting the next day log return of the S and P five hundred index.



The following models were evaluated:



\* Naive last return

  \* Predicts the next day return as equal to the most recent observed return in the input window.

\* MA5 last returns

  \* Predicts the next day return as the mean of the last five returns in the input window.

\* Linear regression

  \* Uses the full thirty day window of features, flattened into a single vector, followed by standardisation and a linear regression model.



\## Validation set performance



Insert your real numbers here, for example:



\* Naive last return

  mae about …

  rmse about …

  directional accuracy about …



\* MA5 last returns

  mae about …

  rmse about …

  directional accuracy about …



\* Linear regression

  mae about …

  rmse about …

  directional accuracy about …



\## Test set performance



Same structure, for example:



\* Naive last return

  mae about …

  rmse about …

  directional accuracy about …



\* MA5 last returns

  mae about …

  rmse about …

  directional accuracy about …



\* Linear regression

  mae about …

  rmse about …

  directional accuracy about …



\## Interpretation



Here you write short, honest comments such as:



\* The naive and MA5 baselines already capture a part of the signal, but their directional accuracy is close to chance over the test period.

\* Linear regression slightly improves error metrics compared to the naive baselines, which suggests that there is some value in using the full window of features.

\* Since the current sentiment features are synthetic placeholders, they do not contribute real information yet. The baselines mainly reflect the behaviour of price based features.

\* These baselines provide a reference level that more advanced sequence models must improve on to be considered useful.



\## Planned sequence model



The next step after establishing baselines is a sequence model based on an LSTM architecture. The model will take a sliding window of thirty days of features as input and will output a prediction for the next day log return. The same data representation that was used for the baselines is reused, which keeps the comparison fair.



\## LSTM sequence model



After training the LSTM based sequence model on the same data representation, its performance on the validation and test sets can be directly compared to the baselines.



Here you will insert your real numbers, for example:



\* On the validation set the LSTM achieved a mean absolute error of … and a root mean squared error of … which is better than the linear regression baseline.

\* On the test set the LSTM reached a directional accuracy of … which is compared against the naive and MA5 baselines.

\* This indicates whether the additional temporal modelling capacity of the LSTM provides a measurable benefit over simple linear and naive models.



At this stage sentiment features are still synthetic, so the improvements primarily reflect the use of richer price based patterns in the sequence.



\## Comparison of baselines and LSTM



The combined results table model\_comparison\_lstm\_vs\_baselines shows the performance of the naive, MA5, linear regression, and LSTM models on the validation and test sets.



Here you can summarise with your real numbers, for example:



\* On the test set, the LSTM achieves a lower mean absolute error and root mean squared error than the naive and MA5 baselines, and slightly improves over the linear regression model.

\* Directional accuracy may still be close to chance, which is expected given the difficulty of short term return prediction.

\* The time series plots and scatter plots confirm that predictions remain noisy, but the LSTM tracks larger movements somewhat better than the simple baselines.



These observations will later be refined and included in the thesis results chapter.



\# LSTM sequence model results



This document summarises the performance of the LSTM based forecasting model for next day S and P five hundred log returns.



\## Model configuration



\* Input window length: thirty trading days

\* Features: price based features and synthetic sentiment features from the merged dataset

\* Architecture: two layer LSTM with hidden size sixty four and dropout zero point one

\* Loss function: mean squared error

\* Optimiser: Adam with learning rate zero point zero zero one

\* Early stopping: patience of five epochs based on validation loss



\## Training behaviour



The training and validation loss curves in `lstm\_train\_val\_loss.png` show how the model converges. The validation loss stabilises after several epochs, and early stopping prevents overfitting.



\## Validation and test performance



Here fill in your actual metrics, for example:



\* Validation set  

&nbsp; mean absolute error about  

&nbsp; root mean squared error about  

&nbsp; directional accuracy about  



\* Test set  

&nbsp; mean absolute error about  

&nbsp; root mean squared error about  

&nbsp; directional accuracy about  



These values can be read from `lstm\_metrics.json` or from the printed output in the notebook.



\## Comparison against baselines



Compared with the naive last return, MA5 and linear regression baselines recorded in `model\_comparison\_lstm\_vs\_baselines.csv`:



\* The LSTM reduces error metrics on the validation and test sets by a certain margin

\* Directional accuracy may remain close to chance, which is typical for short horizon equity index returns

\* Since sentiment input is still synthetic, most of the signal exploited by the LSTM comes from price dynamics and simple technical features



These points will be refined and integrated into the thesis results chapter.



