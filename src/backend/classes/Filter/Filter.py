from pydantic import BaseModel
# ---------------------------- MODELOS ----------------------------
class Signal(BaseModel):
    path: str
    name: str


class Filter(BaseModel):
    sp: float

