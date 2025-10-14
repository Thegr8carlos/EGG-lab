# EGG-Lab
Una interfaz interactiva basada en **Dash** para la exploración y el modelado de datos EEG de extremo a extremo.

## Tabla de contenidos
1. [Descripción](#descripción)
2. [Características principales](#características-principales)
3. [Requisitos previos](#requisitos-previos)
4. [Instalación y puesta en marcha](#instalación-y-puesta-en-marcha)
5. [Estructura del proyecto](#estructura-del-proyecto)
6. [Contribuir](#contribuir)
7. [Licencia](#licencia)

---

## Descripción
EGG-Lab provee una UI ligera y extensible que combina el poder de **Dash/Plotly** con utilidades específicas para señales EEG, permitiendo desde la inspección visual hasta la prueba rápida de modelos de machine-learning.  

## Características principales
- 📊 Dashboard interactivo hecho con Dash.  
- ⚡ Capa de utilidades para pre-procesamiento y visualización de señales EEG.  
- 🔌 Diseño modular: nuevos componentes se añaden fácilmente bajo `src/components`.  
- 📝 Configurado para ejecutar con **Python 3.12** en un entorno virtual aislado.

## Requisitos previos
| Software | Versión mínima | Notas |
|----------|----------------|-------|
| Python   | 3.12           | Probado en 3.12.x |
| Git      | 2.25           | Para clonar el repositorio |

> **Nota**: Se recomienda usar un *virtual environment* (`python -m venv`) para mantener las dependencias aisladas del resto del sistema. :contentReference[oaicite:0]{index=0}

## Instalación y puesta en marcha
```bash
# 1. Clonar el repositorio
git clone https://github.com/<usuario>/EGG-lab.git
cd EGG-lab

# 2. Ir al directorio fuente
cd src

# 3. Crear y activar un entorno virtual
python3.12 -m venv .venv        # crea la carpeta .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows PowerShell

# 4. Instalar dependencias
pip install -r requirements.txt

# 5. Ejecutar la aplicación
python3.12 main.py


