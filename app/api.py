import joblib
import pandas as pd

model = joblib.load("./models/xgb_model_v1.pkl")

@app.post("/predict")
def predict(dat: InputSchema):
    input_df = pd.Dataframe([data.dict()])