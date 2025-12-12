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



