from pydantic import BaseModel, Field

# ---------------------------- BASE ----------------------------

class Classifier(BaseModel):
    epochs: int = Field(
        50,
        ge=1,
        le=1000,
        description="Número de épocas para el entrenamiento"
    )
    batch_size: int = Field(
        32,
        ge=1,
        le=512,
        description="Tamaño del batch (lote) para entrenamiento"
    )


# ---------------------------- MODELOS CLASIFICADORES ----------------------------
































# if __name__ == "__main__":
#     from pprint import pprint

#     print("🧠 Esquemas de Modelos de Clasificación:")
#     pprint(ClassifierSchemaFactory.get_all_classifier_schemas())
