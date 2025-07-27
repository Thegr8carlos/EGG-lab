

def generar_mapa_validacion_inputs(all_schemas: dict) -> list[dict]:
    
    resultado = []
    #print(all_schemas)

    for tipo_filtro, schema in all_schemas.items():
        propiedades = schema.get("properties", {})
        input_map = {}

        for nombre_campo, info in propiedades.items():
            # This matches exactly the ID used in build_configuration_ui
            input_id = f"{tipo_filtro}-{nombre_campo}"

            # Determine type
            tipo = info.get("type")
            any_of = info.get("anyOf")
            enum = info.get("enum")

            if any_of:
                # Prefer non-null type from anyOf
                tipos = [a.get("type") for a in any_of if a.get("type") and a["type"] != "null"]
                if tipos:
                    tipo = tipos[0]

            # Map evaluation function
            if enum:
                evaluacion = f"validar_enum({enum})"
            elif tipo == "number":
                evaluacion = "validar_float(valor)"
            elif tipo == "integer":
                evaluacion = "validar_int(valor)"
            elif tipo == "array":
                evaluacion = "validar_array_de_numeros(valor)"
            elif tipo == "string":
                evaluacion = "validar_str(valor)"
            else:
                evaluacion = "valor  # sin validación definida"

            input_map[input_id] = evaluacion

        # Ensure the button ID matches the one in build_configuration_ui
        resultado.append({
            f"btn-aplicar-{tipo_filtro}": input_map
        })
    #print(resultado)

    return resultado