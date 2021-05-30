# Shared App

https://share.streamlit.io/nico-facto/leukemia-apl-classification/main/Leucemie_app.py

# Run App Locally 

streamlit run Leucemie_app.py

# Goal 

**Try to have the best acc on classe 1,  with the least possible false negative. Having false positives could be very serious problem, so we don't want them.**

Acute promyelocytic leukemia (APL) has an excellent prognosis, but still a high rate of early death due to severe coagulopathy. In order to avoid these early deaths, immediate initiation of treatment is recommended, which depends on a fast and accurate recognition of this acute myeloid leukemia (AML) subtype. Early diagnosis relies on cytology, which requires a high level of expertise and can be misleading. Hence, the development of a simple diagnostic tool to diagnose APL based on routine biological parameters might represent an important step to improve prompt and accurate recognition of APL on a 7/24 basis. To tackle this challenge, we have developed a classifier using machine learning on a large cohort of 222 APL and non-APL AML. Based on only 8 parameters, it achieves very high discriminating capacities on the test cohort (area under the receiver operator curve of 0.95). We have validated this tool on 3 external retrospective cohorts and one prospective cohort (n=415 patients, AUC ROC = 0.96). The accuracy of the tool is above 99.5% for two third of patients with a confidence score above 99%. Finally, we have created a user-friendly web interface to make this predictor accessible all over the world.


# Data and research

 - Données projet LAP Cas-témoin 02-10-20.xlsx : Initial dataset with all features. Target = Témoin/Cas (0/1). Non-APL = 0.
 - cohorte_full_data.csv : all the validation data collected with only features slectionned. Target = target. Non-APL = 0.

 All the research code is in Notebook_Prepa_Leukemia folder. 
  - 01 : data exploration.
  - 02 : Model exploration.
  - 03 : code for drop databricks and model, testing model on validation data.
  - 05 : Create anomalies detection for app, and testing retrain with validation data.

Most of the code was refactoringin lib folder on python.py files and import on notebook,
it can be re-used with other bionological data, among others ...

# Custom Virtual Env 

    Made with Python 3.7 on windows 10
    At the root of a folder : 
    - python -m venv YourEnvName
    - Activate : cd ... YourEnvName\Scripts\  cmd = activate
    - Move back to root of folder
    - python -m pip install --upgrade pip
    - pip install -r requirements.txt
    - desactivate : cd ... YourEnvName\Scripts\ cmd = deactivate


# Contributor 

Guillaume Meurisse (Big Thanks)

# Author

Nicolas Autexier 
nicolas.atx@gmx.fr