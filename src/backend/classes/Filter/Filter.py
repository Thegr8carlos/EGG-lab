from pydantic import BaseModel, Field

class Filter(BaseModel):
    id: str
    sp: float = Field(..., gt=0, description="Frecuencia de muestreo en Hz (debe ser > 0)")

    
    def get_sp(self) -> float:
        return self.sp
    
    def get_id(self) -> str:
        return self.id