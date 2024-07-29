import os
from pathlib import Path

# Paths
BASE_DIR = './'
DATA = os.path.join(BASE_DIR, 'data/')
TEST_FILE = os.path.join(DATA, 'Paitients_Files_Test.csv')
HISTORY = os.path.join(DATA, 'history/')
HISTORY_FILE = os.path.join(HISTORY, 'history.csv')

# Urls
TEST_FILE_URL = "https://raw.githubusercontent.com/D0nG4667/sepsis_prediction_full_stack/model_development/dev/data/Paitients_Files_Test.csv"


# ENV when using standalone streamlit server
ENV_PATH = Path('../../env/online.env')

ALL_MODELS = [
    "AdaBoostClassifier",
    "CatBoostClassifier",
    "DecisionTreeClassifier",
    "KNeighborsClassifier",
    "LGBMClassifier",
    "LogisticRegression",
    "RandomForestClassifier",
    "SupportVectorClassifier",
    "XGBoostClassifier",
]

BEST_MODELS = ["RandomForestClassifier", "XGBoostClassifier"]

markdown_table_all = """    
| Column   Name                | Attribute/Target | Description                                                                                                                                                                                                  |
|------------------------------|------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ID                           | N/A              | Unique number to represent patient ID                                                                                                                                                                        |
| PRG           | Attribute1       |  Plasma glucose|
| PL               | Attribute 2     |   Blood Work Result-1 (mu U/ml)                                                                                                                                                |
| PR              | Attribute 3      | Blood Pressure (mm Hg)|
| SK              | Attribute 4      | Blood Work Result-2 (mm)|
| TS             | Attribute 5      |     Blood Work Result-3 (mu U/ml)|                                                                                  
| M11     | Attribute 6    |  Body mass index (weight in kg/(height in m)^2|
| BD2             | Attribute 7     |   Blood Work Result-4 (mu U/ml)|
| Age              | Attribute 8      |    patients age  (years)|
| Insurance | N/A     | If a patient holds a valid insurance card|
| Sepsis                 | Target           | Positive: if a patient in ICU will develop a sepsis , and Negative: otherwise |
"""
