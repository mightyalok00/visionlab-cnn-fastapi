from pydantic import BaseModel, Field


class TopPrediction(BaseModel):
    class_name: str
    probability: float = Field(ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float = Field(ge=0.0, le=1.0)
    top_predictions: list[TopPrediction]
    probabilities: dict[str, float]
    inverted_background: bool = False


class ExplainResponse(BaseModel):
    predicted_class: str
    confidence: float = Field(ge=0.0, le=1.0)
    gradcam_base64: str
    preprocessed_base64: str
    inverted_background: bool = False


class BatchItemPrediction(BaseModel):
    filename: str
    predicted_class: str
    confidence: float = Field(ge=0.0, le=1.0)
    top_predictions: list[TopPrediction]
    inverted_background: bool = False


class BatchPredictionResponse(BaseModel):
    total_processed: int
    predictions: list[BatchItemPrediction]
