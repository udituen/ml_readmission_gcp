import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline



model = XGBClassifier(n_estimators=100,
                        learning_rate=0.1,
                        max_depth=5,
                        random_state=42)

pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", model)
])