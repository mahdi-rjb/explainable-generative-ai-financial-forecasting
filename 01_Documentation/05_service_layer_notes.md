Service layer overview



To make the future Streamlit dashboard simple and focused, the project defines a small service layer:



\* forecasting\_service

&nbsp; \* load\_lstm\_model\_and\_metadata

&nbsp; \* load\_latest\_feature\_window

&nbsp; \* predict\_next\_return\_with\_lstm



\* generative\_service

&nbsp; \* load\_trained\_vae

&nbsp; \* load\_recent\_return\_window

&nbsp; \* generate\_return\_scenarios



\* explainability\_service

&nbsp; \* load\_tabular\_data

&nbsp; \* train\_random\_forest\_for\_explainability

&nbsp; \* compute\_local\_shap\_for\_index



The dashboard will call these functions rather than working directly with raw files or models, which keeps the user interface code short and easier to maintain.



