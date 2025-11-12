from pydantic import BaseModel

# ---------------------------- BASE ----------------------------

class Transform(BaseModel):
    sp: float  # puntos por segundo
    id: str   # identificador único (dentro del experimento)


    def get_sp(self) -> float:
        return self.sp
    def get_id(self) -> str:
        return self.id









# if __name__ == "__main__":
#     from pprint import pprint

#     print("\n🎯 Esquemas de Transformadas:")
#     pprint(TransformSchemaFactory.get_all_transform_schemas())
