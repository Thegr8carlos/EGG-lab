
from backend.classes.Filter.Filter import Filter
from pydantic import Field
from typing import Optional,  Literal


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
        
    @classmethod
    def apply(cls, instance: "ICA") -> None:
        """
        Simula la aplicación del filtro ICA.
        """
        print(f"[SIMULACIÓN] Aplicando filtro ICA:")
        print(f"  Método: {instance.method}")
        print(f"  Número de Componentes: {instance.numeroComponentes}")
        print(f"  Random State: {instance.random_state}")
        print(f"  Máx Iteraciones: {instance.max_iter}")
