
from backend.classes.Filter.Filter import Filter
from pydantic import  Field
from typing import Optional,  Literal, Union, List



class Notch(Filter):
    freqs: Union[float, List[float]] = Field(
        ...,
        description="Frecuencia o lista de frecuencias a atenuar"
    )
    quality: Optional[float] = Field(
        30.0,
        ge=1.0,
        le=100.0,
        description="Factor de calidad del filtro (Q). A mayor valor, mayor selectividad"
    )
    method: Literal['fir', 'iir'] = Field(
        'fir',
        description="Método para aplicar el filtro notch"
    )