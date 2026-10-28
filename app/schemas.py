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
    status: str = Field(default="success", description="Processing status: 'success' or 'error'")
    predicted_class: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    top_predictions: list[TopPrediction] | None = None
    inverted_background: bool = False
    error: str | None = None


class BatchPredictionResponse(BaseModel):
    total_images: int
    successful: int
    failed: int
    predictions: list[BatchItemPrediction]
