from pydantic import BaseModel
from typing import Optional, Tuple, Literal, Union, List, Dict, Any


# ---------------------------- MODELOS ----------------------------
class Signal(BaseModel):
    path: str
    name: str


class Filter(BaseModel):
    sp: float


class ICA(Filter):
    numeroComponentes: Optional[int] = None
    method: Literal['fastica', 'picard', 'infomax'] = 'fastica'
    random_state: Optional[int] = None
    max_iter: Optional[int] = 200


class WaveletsBase(Filter):
    wavelet: str
    level: Optional[int] = None
    mode: Optional[str] = 'symmetric'
    threshold: Optional[float] = None


class Bandpass(Filter):
    filter_type: Literal['lowpass', 'highpass', 'bandpass'] = 'bandpass'
    freq: Union[float, Tuple[float, float]]
    method: Literal['fir', 'iir'] = 'fir'
    order: Optional[int] = None
    phase: Literal['zero', 'minimum'] = 'zero'
    fir_window: Optional[str] = 'hamming'


class Notch(Filter):
    freqs: Union[float, List[float]]
    quality: Optional[float] = 30.0
    method: Literal['fir', 'iir'] = 'fir'


# ---------------------------- FACTORY ----------------------------

class FilterSchemaFactory:
    """
    Genera esquemas simplificados para filtros y sus hijos.
    """
    available_filters = {
        'ica': ICA,
        'wavelets': WaveletsBase,
        'bandpass': Bandpass,
        'notch': Notch
    }

    @classmethod
    def get_base_schema(cls) -> Dict[str, Any]:
        return Filter.model_json_schema()

    @classmethod
    def get_all_filter_schemas(cls) -> Dict[str, Dict[str, Any]]:
        schemas = {}
        for key, model in cls.available_filters.items():
            schema = model.model_json_schema()
          
            schemas[key] = schema
        return schemas


# ---------------------------- USO ----------------------------

if __name__ == "__main__":
    from pprint import pprint

    print("🧱 Esquema base Filter:")
    pprint(FilterSchemaFactory.get_base_schema())

    print("\n🔧 Esquemas simplificados de filtros hijos:")
    pprint(FilterSchemaFactory.get_all_filter_schemas())
