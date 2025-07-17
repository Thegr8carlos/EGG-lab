from pydantic import BaseModel, Field, model_validator, ValidationError
from typing import Optional, Literal, Union, Tuple, List,Dict
from backend.classes.mapaValidacion import generar_mapa_validacion_inputs
from dash import callback, Input, Output, State, no_update

# ---------------------------- BASE ----------------------------

class Transform(BaseModel):
    sp: float  # puntos por segundo


# ---------------------------- TRANSFORMADAS ----------------------------


# ------------------- Wavelet Transform -------------------

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


# ------------------- FFT Transform -------------------

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
class TransformSchemaFactory:
    """
    Genera esquemas detallados para transformadas.
    """
    available_transforms = {
        "WaveletTransform": WaveletTransform,
        "FFTTransform": FFTTransform,
        "DCTTransform": DCTTransform
    }

    @classmethod
    def get_all_transform_schemas(cls) -> Dict[str, Dict]:
        schemas = {}
        for key, model in cls.available_transforms.items():
            schema = model.model_json_schema()
            schemas[key] = schema
        return schemas
    








def registrar_callback(boton_id: str, inputs_map: dict):

    available_transforms = {
        "WaveletTransform": WaveletTransform,
        "FFTTransform": FFTTransform,
        "DCTTransform": DCTTransform
    }
    input_ids = list(inputs_map.keys())

    @callback(
        Output(boton_id, "children"),
        Input(boton_id, "n_clicks"),
        [State(input_id, "value") for input_id in input_ids]
    )
    def manejar_formulario(n_clicks, *values, input_ids=input_ids, validadores=inputs_map):
        if not n_clicks:
            return no_update

        filtro_nombre = boton_id.replace("btn-aplicar-", "")
        clase_validadora = available_transforms.get(filtro_nombre)

        datos = {}
        for input_id, value in zip(input_ids, values):
            _, field = input_id.split("-", 1)
            datos[field] = value

        try:
            instancia_valida = clase_validadora(**datos) ##             Aqui se instancia la clase para validar de manera automatica.
            print(f"✅ Datos válidos para {filtro_nombre}: {instancia_valida}")
            """
            -------------------------------------------------------------------------------------------------------------
            Aqui se puede definir Cómo queremos maneajr los experimentos. 
            
            Yo propongo que generemos un auxiliar json llamado Experimeento o algo así
            y para no meternos en problemas de qué y cómo guardar las coasas
            mejor que se mantengan estatica.
            Ya solo haría falta hacer una clase de experimento y su mandero. Esto es basicamente un CRUD
            -------------------------------------------------------------------------------------------------------------
            """
            return no_update
        except ValidationError as e:
            print(f"❌ Errores en {filtro_nombre}: {e}")
            errores = e.errors()
            msg = "\n".join(f"{err['loc'][0]}: {err['msg']}" for err in errores)
            return no_update
for grupo in generar_mapa_validacion_inputs(TransformSchemaFactory.get_all_transform_schemas()):
    for boton_id, inputs_map in grupo.items():
        registrar_callback(boton_id, inputs_map)














if __name__ == "__main__":
    from pprint import pprint

    print("\n🎯 Esquemas de Transformadas:")
    pprint(TransformSchemaFactory.get_all_transform_schemas())
