from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union, Any
from datetime import datetime
from enum import Enum

class ModelType(str, Enum):
    GENERATIVE = "generative"
    EMBEDDING = "embedding"
    MULTIMODAL = "multimodal"
    OTHER = "other"

class AccessType(str, Enum):
    OPEN_SOURCE = "open_source"
    API_ONLY = "api_only"
    CLOSED = "closed"
    OTHER = "other"

class ComplianceLevel(str, Enum):
    COMPLIANT = "compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NON_COMPLIANT = "non_compliant"
    UNKNOWN = "unknown"

class QAPair(BaseModel):
    """Estructura para almacenar una pregunta y su respuesta"""
    question: str
    response: str
    category: str
    timestamp: datetime = Field(default_factory=datetime.now)
    
class ModelMetadata(BaseModel):
    """Estructura para los metadatos del modelo"""
    model_id: str
    name: str
    version: Optional[str] = None
    provider: str
    release_date: Optional[datetime] = None
    model_type: ModelType
    access_type: AccessType
    description: Optional[str] = None
    training_data_description: Optional[str] = None
    parameters_count: Optional[int] = None
    context_window: Optional[int] = None
    environmental_impact: Optional[Dict[str, Any]] = None
    documentation_url: Optional[str] = None
    papers_url: Optional[List[str]] = None
    additional_metadata: Optional[Dict[str, Any]] = None

class PrincipleEvaluation(BaseModel):
    """Evaluación de un principio específico"""
    principle_id: int
    principle_name: str
    compliance_level: ComplianceLevel
    evidence: List[str]
    qa_pairs: List[QAPair]
    recommendations: List[str]
    notes: Optional[str] = None

class ModelCompliance(BaseModel):
    """Evaluación completa de un modelo"""
    metadata: ModelMetadata
    evaluations: List[PrincipleEvaluation]
    overall_compliance: ComplianceLevel
    evaluation_date: datetime = Field(default_factory=datetime.now)
    evaluator: Optional[str] = None
    summary: Optional[str] = None 