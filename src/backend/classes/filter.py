from pydantic import BaseModel, Field, model_validator, ValidationError
from typing import Optional, Tuple, Literal, Union, List, Dict, Any
from dash import callback, Input, Output, State, no_update
from backend.classes.mapaValidacion import generar_mapa_validacion_inputs

# ---------------------------- MODELOS ----------------------------
class Signal(BaseModel):
    path: str
    name: str


class Filter(BaseModel):
    sp: float



# --------------------- ICA ---------------------

class ICA(Filter):
    numeroComponentes: Optional[int] = Field(
        None,
        ge=1,
        description="Número de componentes independientes (opcional)"
    )
    method: Literal['fastica', 'picard', 'infomax'] = Field(
        'fastica',
        description="Método ICA: fastica, picard o infomax"
    )
    random_state: Optional[int] = Field(
        None,
        description="Semilla aleatoria para reproducibilidad"
    )
    max_iter: Optional[int] = Field(
        200,
        ge=1,
        le=10000,
        description="Número máximo de iteraciones"
    )

# --------------------- Wavelets ---------------------

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

# --------------------- Bandpass ---------------------

class Bandpass(Filter):
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
    

# ---------------------------- FACTORY ----------------------------

class FilterSchemaFactory:
    """
    Genera esquemas simplificados para filtros y sus hijos.
    """
    
    available_filters = {
        'ICA': ICA,
        'WaveletsBase': WaveletsBase,
        'Bandpass': Bandpass,
        'Notch': Notch
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













#---------------------------------------------Call backs-----------------------------------


"ejemplo de call bach"

#Para hacer los callbacks de cada uno de los componentes generados dinamicamente, usamos la misma función para generar 
# el esquema json, con la cual: Obtenedremos los id y los campos especificos. Luego, también se generan dinamicamente
# los states del callback, los cuales son guardados en un diciconario y valiadados por las clas. 
# 




def registrar_callback(boton_id: str, inputs_map: dict):

    available_filters = {
        'ICA': ICA,
        'WaveletsBase': WaveletsBase,
        'Bandpass': Bandpass,
        'Notch': Notch
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
        clase_validadora = available_filters.get(filtro_nombre)

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
for grupo in generar_mapa_validacion_inputs(FilterSchemaFactory.get_all_filter_schemas()):
    for boton_id, inputs_map in grupo.items():
        registrar_callback(boton_id, inputs_map)

# ---------------------------- USO ----------------------------

if __name__ == "__main__":
    from pprint import pprint

    print("🧱 Esquema base Filter:")
    pprint(FilterSchemaFactory.get_base_schema())

    print("\n🔧 Esquemas simplificados de filtros hijos:")
    pprint(FilterSchemaFactory.get_all_filter_schemas())
