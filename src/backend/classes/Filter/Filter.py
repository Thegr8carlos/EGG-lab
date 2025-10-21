from pydantic import BaseModel

class Filter(BaseModel):
    id: str
    sp: float

    
    def get_sp(self) -> float:
        return self.sp
    
    def get_id(self) -> str:
        return self.id