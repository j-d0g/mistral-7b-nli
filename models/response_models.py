from pydantic import BaseModel, Field, field_validator
from typing import Optional

class NLIResponse(BaseModel):
    """Pydantic model for validating NLI task responses"""
    thought_process: str = Field(..., description="Step-by-step reasoning process")
    label: int = Field(..., description="Entailment label (0=no, 1=yes)")
    
    @field_validator('label')
    def validate_label(cls, v):
        if v not in [0, 1]:
            raise ValueError('Label must be 0 or 1')
        return v

class ScoringResponse(BaseModel):
    """Pydantic model for validating scoring responses"""
    score: int = Field(..., ge=0, le=5, description="Quality score (0-5)")
    label: int = Field(..., description="Entailment label (0=no, 1=yes)")
    improved_thoughts: str = Field(..., description="Improved reasoning")
    
    @field_validator('label')
    def validate_label(cls, v):
        if v not in [0, 1]:
            raise ValueError('Label must be 0 or 1') 
        return v
    
    @field_validator('score')
    def validate_score(cls, v):
        if not (0 <= v <= 5):
            raise ValueError('Score must be between 0 and 5')
        return v 