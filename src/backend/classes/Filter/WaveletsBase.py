from backend.classes.Filter.Filter import Filter
from pydantic import Field
from typing import Optional


class WaveletsBase(Filter):
    wavelet: str = Field(
        ...,
        description="Nombre de la wavelet a usar (por ejemplo: db4, coif5, etc.)"
    )
    level: Optional[int] = Field(
        None,
        ge=1,
        le=10,
        description="Nivel de descomposición (opcional)"
    )
    mode: Optional[str] = Field(
        'symmetric',
        description="Modo de extensión de bordes: symmetric, periodic, etc."
    )
    threshold: Optional[float] = Field(
        None,
        ge=0.0,
        description="Valor de umbral para denoising (si aplica)"
    )

    @classmethod
    def apply(cls, instance: "WaveletsBase") -> None:
        """
        Simula la aplicación del filtro WaveletsBase.
        """
        print(f"[SIMULACIÓN] Aplicando filtro WaveletsBase:")
        print(f"  Wavelet: {instance.wavelet}")
        print(f"  Nivel: {instance.level}")
        print(f"  Modo: {instance.mode}")
        print(f"  Umbral: {instance.threshold}")
