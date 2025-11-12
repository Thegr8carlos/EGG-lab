# Documentación UML - EGG-lab

Esta carpeta contiene todos los diagramas UML del proyecto, organizados por tipo.

## Estructura

```
doc/
├── class_diagrams/      - Diagramas de Clases (estructura estática)
├── sequence_diagrams/   - Diagramas de Secuencia (interacciones temporales)
├── activity_diagrams/   - Diagramas de Actividad (flujos de trabajo)
├── use_case_diagrams/   - Diagramas de Casos de Uso (interacción usuario-sistema)
└── component_diagrams/  - Diagramas de Componentes (arquitectura del sistema)
```

## Diagramas Actuales

### Diagramas de Clases
- `dataset_class.puml` - Diagrama de clases de Dataset (backend/classes/dataset.py)

## Cómo visualizar los diagramas

### Opción 1: Editor Web de PlantUML
1. Abre http://www.plantuml.com/plantuml/uml/
2. Copia el contenido del archivo `.puml`
3. Pégalo en el editor

### Opción 2: VSCode con extensión PlantUML
1. Instala la extensión "PlantUML" (jebbs.plantuml)
2. Abre cualquier archivo `.puml`
3. Presiona `Alt+D` para preview

### Opción 3: Kroki.io
1. Visita https://kroki.io/
2. Selecciona "PlantUML"
3. Pega el código y visualiza

## Convenciones de nombres

- **Diagramas de clases**: `<nombre_clase>_class.puml`
- **Diagramas de secuencia**: `<funcionalidad>_sequence.puml`
- **Diagramas de actividad**: `<proceso>_activity.puml`
- **Diagramas de casos de uso**: `<modulo>_usecase.puml`
- **Diagramas de componentes**: `<sistema>_component.puml`
