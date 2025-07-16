from pydantic import BaseModel
from typing import Optional, Literal, Union, Tuple, List,Dict


# ---------------------------- BASE ----------------------------

class Transform(BaseModel):
    sp: float  # puntos por segundo


# ---------------------------- TRANSFORMADAS ----------------------------

class WaveletTransform(Transform):
    wavelet: str
    level: Optional[int] = None
    mode: Optional[str] = "symmetric"
    threshold: Optional[float] = None


class FFTTransform(Transform):
    window: Optional[Literal["hann", "hamming", "blackman", "rectangular"]] = "hann"
    nfft: Optional[int] = None
    overlap: Optional[float] = 0.0  # porcentaje entre 0.0 y 1.0


class DCTTransform(Transform):
    type: Optional[Literal[1, 2, 3, 4]] = 2
    norm: Optional[Literal["ortho", None]] = None
    axis: Optional[int] = -1


class TransformSchemaFactory:
    """
    Genera esquemas detallados para transformadas.
    """
    available_transforms = {
        "wavelet": WaveletTransform,
        "fft": FFTTransform,
        "dct": DCTTransform
    }

    @classmethod
    def get_all_transform_schemas(cls) -> Dict[str, Dict]:
        schemas = {}
        for key, model in cls.available_transforms.items():
            schema = model.model_json_schema()
            schemas[key] = schema
        return schemas
if __name__ == "__main__":
    from pprint import pprint

    print("\n🎯 Esquemas de Transformadas:")
    pprint(TransformSchemaFactory.get_all_transform_schemas())
