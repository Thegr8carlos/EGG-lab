from backend.classes.FeatureExtracture.FeatureExtracture import Transform
from typing import Optional, Literal
from pydantic import  Field


# ------------------- DCT Transform -------------------

class DCTTransform(Transform):
    type: Optional[Literal[1, 2, 3, 4]] = Field(
        2,
        description="Tipo de DCT: 1, 2, 3 o 4"
    )
    norm: Optional[Literal["ortho", None]] = Field(
        None,
        description="Tipo de normalización: 'ortho' o None"
    )
    axis: Optional[int] = Field(
        -1,
        ge=-3,
        le=3,
        description="Eje sobre el que se aplica la DCT (ej. -1 para último eje)"
    )