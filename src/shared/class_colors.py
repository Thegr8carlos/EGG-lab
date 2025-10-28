"""
Sistema centralizado de colores para clases de eventos EEG.

Este módulo asegura que cada clase tenga siempre el mismo color
en todas las páginas de la aplicación, independientemente del orden.
"""

# Mapa de colores fijos por nombre de clase
# Usando colores vibrantes y diferenciados en HSL
CLASS_COLORS = {
    # Clases de inner speech (4 direcciones)
    "abajo": "hsl(0, 75%, 55%)",      # Rojo vibrante
    "arriba": "hsl(120, 70%, 50%)",   # Verde brillante
    "derecha": "hsl(210, 75%, 55%)",  # Azul cielo
    "izquierda": "hsl(45, 85%, 55%)", # Amarillo/dorado

    # Clases P300 (target/non-target)
    "target": "hsl(270, 70%, 55%)",     # Morado
    "non-target": "hsl(180, 65%, 50%)", # Cyan/turquesa
    "nontarget": "hsl(180, 65%, 50%)",  # Alias para non-target

    # Clases adicionales comunes
    "left": "hsl(45, 85%, 55%)",        # Amarillo (igual que izquierda)
    "right": "hsl(210, 75%, 55%)",      # Azul (igual que derecha)
    "up": "hsl(120, 70%, 50%)",         # Verde (igual que arriba)
    "down": "hsl(0, 75%, 55%)",         # Rojo (igual que abajo)

    # Colores adicionales para otras clases
    "rest": "hsl(300, 60%, 60%)",       # Magenta
    "baseline": "hsl(30, 70%, 60%)",    # Naranja
    "task": "hsl(160, 65%, 50%)",       # Verde azulado
    "control": "hsl(220, 60%, 60%)",    # Azul periwinkle
}

def get_class_color(class_name: str, index: int = 0) -> str:
    """
    Obtiene el color para una clase específica.

    Args:
        class_name: Nombre de la clase
        index: Índice de la clase (usado como fallback si no hay color fijo)

    Returns:
        Color en formato HSL (ej: "hsl(0, 75%, 55%)")
    """
    # Normalizar nombre (lowercase, sin espacios)
    normalized_name = str(class_name).lower().strip().replace(" ", "_")

    # Buscar en el mapa de colores fijos
    if normalized_name in CLASS_COLORS:
        return CLASS_COLORS[normalized_name]

    # Fallback: generar color basado en hash del nombre para consistencia
    # Usar hash del nombre para que siempre dé el mismo color
    name_hash = abs(hash(normalized_name))
    hue = (name_hash * 37) % 360  # 37 para buena distribución
    return f"hsl({hue}, 70%, 50%)"


def get_class_colors_map(classes: list) -> dict:
    """
    Genera un mapa de colores para una lista de clases.

    Args:
        classes: Lista de nombres de clases

    Returns:
        Diccionario {clase: color}
    """
    color_map = {}
    for idx, class_name in enumerate(classes):
        color_map[str(class_name)] = get_class_color(class_name, idx)
    return color_map


def get_class_color_rgb(class_name: str, index: int = 0) -> str:
    """
    Obtiene el color en formato RGB para una clase específica.
    Útil para Plotly u otras librerías que no soportan HSL.

    Args:
        class_name: Nombre de la clase
        index: Índice de la clase (usado como fallback)

    Returns:
        Color en formato RGB (ej: "rgb(255, 100, 100)")
    """
    import re

    hsl_color = get_class_color(class_name, index)

    # Extraer valores HSL
    match = re.match(r'hsl\((\d+),\s*(\d+)%,\s*(\d+)%\)', hsl_color)
    if not match:
        return "rgb(128, 128, 128)"  # Gris por defecto

    h, s, l = int(match.group(1)), int(match.group(2)) / 100, int(match.group(3)) / 100

    # Convertir HSL a RGB
    def hsl_to_rgb(h, s, l):
        c = (1 - abs(2 * l - 1)) * s
        x = c * (1 - abs((h / 60) % 2 - 1))
        m = l - c / 2

        if 0 <= h < 60:
            r, g, b = c, x, 0
        elif 60 <= h < 120:
            r, g, b = x, c, 0
        elif 120 <= h < 180:
            r, g, b = 0, c, x
        elif 180 <= h < 240:
            r, g, b = 0, x, c
        elif 240 <= h < 300:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x

        r, g, b = int((r + m) * 255), int((g + m) * 255), int((b + m) * 255)
        return f"rgb({r}, {g}, {b})"

    return hsl_to_rgb(h, s, l)
