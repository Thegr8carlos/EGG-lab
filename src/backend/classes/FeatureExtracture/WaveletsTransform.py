from backend.classes.FeatureExtracture.FeatureExtracture import Transform
from typing import Optional
from pydantic import  Field

class WaveletTransform(Transform):
    wavelet: str = Field(
        ...,
        description="Nombre de la wavelet a usar (ej. db4, coif5, etc.)"
    )
    level: Optional[int] = Field(
        None,
        ge=1,
        le=10,
        description="Nivel de descomposición (opcional)"
    )
    mode: Optional[str] = Field(
        "symmetric",
        description="Modo de extensión de bordes: symmetric, periodic, etc."
    )
    threshold: Optional[float] = Field(
        None,
        ge=0.0,
        description="Valor de umbral para denoising (si aplica)"
    )
    @classmethod
    def apply(cls, instance: "WaveletTransform") -> None:
        """
        Simula la aplicación de la transformación Wavelet.
        """
        print(f"[SIMULACIÓN] Aplicando transformación Wavelet:")
        print(f"  Wavelet: {instance.wavelet}")
        print(f"  Nivel: {instance.level}")
        print(f"  Modo: {instance.mode}")
        print(f"  Umbral: {instance.threshold}")
