from backend.classes.FeatureExtracture.FeatureExtracture import Transform
from typing import Optional, Literal
from pydantic import  Field



class FFTTransform(Transform):
    window: Literal["hann", "hamming", "blackman", "rectangular"] = Field(
        "hann",
        description="Tipo de ventana: hann, hamming, blackman, rectangular"
    )
    nfft: Optional[int] = Field(
        None,
        ge=1,
        description="Número de puntos para la FFT (opcional)"
    )
    overlap: Optional[float] = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Porcentaje de solapamiento (entre 0.0 y 1.0)"
    )