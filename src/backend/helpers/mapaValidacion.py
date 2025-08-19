def generar_mapa_validacion_inputs(all_schemas: dict) -> list[dict]:
    resultado = []

    for key, schema in all_schemas.items():
        # Usar el mismo prefijo que usa la UI para los IDs
        tipo_ui = schema.get("title", schema.get("type", key))

        propiedades = schema.get("properties", {}) or {}
        input_map = {}

        for nombre_campo, info in propiedades.items():
            # Debe coincidir con build_configuration_ui: f"{type}-{field_name}"
            input_id = f"{tipo_ui}-{nombre_campo}"

            # Determinar tipo
            tipo = info.get("type")
            any_of = info.get("anyOf")
            enum = info.get("enum")

            if any_of:
                tipos = [a.get("type") for a in any_of if a.get("type") and a["type"] != "null"]
                if tipos:
                    tipo = tipos[0]

            # Map de validación
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

        # Botón debe coincidir con build_configuration_ui: f"btn-aplicar-{type}"
        resultado.append({f"btn-aplicar-{tipo_ui}": input_map})

    return resultado
