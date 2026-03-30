import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import xgboost as xgb
from xgboost import XGBClassifier
import joblib 

def main():
    data = pd.read_csv("./data/diabetic_data.csv")
    print(data.head())

    X = data.iloc[:,0:50]
    y = data.iloc[:,-1].apply(lambda x: 0 if x == 'NO' else 1)

    print(y)
    # define features
    meds_col = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
       'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
       'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
       'tolazamide', 'examide', 'citoglipton', 'insulin',
       'glyburide-metformin', 'glipizide-metformin',
       'glimepiride-pioglitazone', 'metformin-rosiglitazone',
       'metformin-pioglitazone']
    cat_col = ['age', 'gender']
    num_col = ['num_lab_procedures', 'num_procedures',
       'num_medications', 'number_outpatient', 'number_emergency',
       'number_inpatient', 'number_diagnoses']
    
    # split into train test sizes
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # build pipelines for med, cat and num columns

    cat_transformer = Pipeline(steps = [
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('one_hot_encode', OneHotEncoder()),
    ])

    num_tranformer = Pipeline(steps=[
        ('impute', SimpleImputer(strategy='mean')),
        ('scale', StandardScaler())
    ])

    meds_transformer = Pipeline(steps=[
        ('binary_encode', OrdinalEncoder())
    ])

    # define column transformer
    preprocessor = ColumnTransformer(transformers=[
        ('cat',cat_transformer, cat_col),
        ('num',num_tranformer,num_col),
        ('ord',meds_transformer,meds_col)
    ],
    remainder="drop"
    )

    model = XGBClassifier(n_estimators=100,
                        learning_rate=0.1,
                        max_depth=5,
                        random_state=42)

    pipeline = make_pipeline(preprocessor, model)


    pipeline.fit(X_train, y_train)

    print(pipeline.score(X_test,y_test))

    joblib.dump(pipeline, './models/xgb_model_v2.pkl')
    
main()