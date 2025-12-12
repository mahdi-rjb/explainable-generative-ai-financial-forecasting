Explainability setup



To obtain interpretable explanations, a random forest regression model is trained on a daily tabular representation of the S and P five hundred data. The input features include log returns, simple technical indicators and aggregated news sentiment. The target is the next day log return.



SHAP TreeExplainer is used to compute Shapley values on a subset of the validation set. Summary plots provide global feature importance across time, and dependence plots show how changes in individual features affect the predicted next day return. These explanations are later integrated into the dashboard so that users can inspect why the model predicts a certain movement on a given day.



