from pydantic import BaseModel, Field
from typing import Literal

class PatientInput(BaseModel):
    # Categorical
    age: str = Field(example="[70-80)")
    gender: str = Field(example="Female")

    # Numerical
    num_lab_procedures: int = Field(example=41)
    num_procedures: int = Field(example=0)
    num_medications: int = Field(example=12)
    number_outpatient: int = Field(example=0)
    number_emergency: int = Field(example=0)
    number_inpatient: int = Field(example=0)
    number_diagnoses: int = Field(example=9)

    # Medication columns
    metformin: str = Field(example="No")
    repaglinide: str = Field(example="No")
    nateglinide: str = Field(example="No")
    chlorpropamide: str = Field(example="No")
    glimepiride: str = Field(example="No")
    acetohexamide: str = Field(example="No")
    glipizide: str = Field(example="No")
    glyburide: str = Field(example="No")
    tolbutamide: str = Field(example="No")
    pioglitazone: str = Field(example="No")
    rosiglitazone: str = Field(example="No")
    acarbose: str = Field(example="No")
    miglitol: str = Field(example="No")
    troglitazone: str = Field(example="No")
    tolazamide: str = Field(example="No")
    examide: str = Field(example="No")
    citoglipton: str = Field(example="No")
    insulin: str = Field(example="No")
    glyburide_metformin: str = Field(alias="glyburide-metformin", example="No")
    glipizide_metformin: str = Field(alias="glipizide-metformin", example="No")
    glimepiride_pioglitazone: str = Field(alias="glimepiride-pioglitazone", example="No")
    metformin_rosiglitazone: str = Field(alias="metformin-rosiglitazone", example="No")
    metformin_pioglitazone: str = Field(alias="metformin-pioglitazone", example="No")

    model_config = {"populate_by_name": True}


class PredictionResponse(BaseModel):
    prediction: int
    prediction_label: str
    probability_readmitted: float
    model_version: str = "1.0.0"