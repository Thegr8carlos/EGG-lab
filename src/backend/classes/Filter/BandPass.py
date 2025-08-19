
from backend.classes.Filter.Filter import Filter
from pydantic import Field
from typing import Optional,  Literal, Union, Tuple


# --------------------- Bandpass ---------------------

class BandPass(Filter):
    filter_type: Literal['lowpass', 'highpass', 'bandpass'] = Field(
        'bandpass',
        description="Tipo de filtro: lowpass, highpass o bandpass"
    )
    freq: Union[float, Tuple[float, float]] = Field(
        ...,
        description="Frecuencia de corte: una sola (float) o un par (low, high)"
    )
    method: Literal['fir', 'iir'] = Field(
        'fir',
        description="Método de diseño del filtro"
    )
    order: Optional[int] = Field(
        None,
        ge=1,
        le=1000,
        description="Orden del filtro (opcional, depende del método)"
    )
    phase: Literal['zero', 'minimum'] = Field(
        'zero',
        description="Tipo de fase para el filtro FIR"
    )
    fir_window: Optional[str] = Field(
        'hamming',
        description="Ventana para diseño FIR: hamming, hann, blackman, etc."
    )
    @classmethod
    def apply(cls, instance: "BandPass") -> None:
        """
        Simula la aplicación del filtro BandPass.
        """
        print(f"[SIMULACIÓN] Aplicando filtro BandPass:")
        print(f"  Tipo de filtro: {instance.filter_type}")
        print(f"  Frecuencia(s): {instance.freq}")
        print(f"  Método: {instance.method}")
        print(f"  Orden: {instance.order}")
        print(f"  Fase: {instance.phase}")
        print(f"  Ventana FIR: {instance.fir_window}")


