# validation of input request

from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    age: int = Field(..., ge=0, le=120)
    num_procedures: int
    num_medications: int
    gender: str
    admission_type: str