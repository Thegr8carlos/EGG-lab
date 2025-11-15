"""
Configuración GPU Global para TensorFlow

DEBE llamarse al inicio de la aplicación, ANTES de importar cualquier modelo.
Este módulo configura la GPU para evitar fragmentación de memoria.
"""

import os
import sys


def configure_gpu_for_training():
    """
    Configura GPU para entrenamiento secuencial de modelos.

    Características:
    - Memory Growth: Aloca memoria gradualmente en vez de reservar toda
    - CUDA Async Allocator: Reduce fragmentación de memoria
    - Optimizado para entrenamientos múltiples sin reiniciar proceso

    Returns:
        bool: True si GPU configurada exitosamente, False si no hay GPU o falló
    """

    print("=" * 70)
    print("🔧 [GPU CONFIG] Inicializando configuración GPU...")
    print("=" * 70)

    # ======================================================================
    # PASO 1: Configurar variables de ambiente (ANTES de import tensorflow)
    # ======================================================================

    # Memory Growth: Permite que TensorFlow aloque memoria gradualmente
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    print("   ✓ TF_FORCE_GPU_ALLOW_GROWTH = true")

    # CUDA Async Allocator: Reduce fragmentación de memoria
    # Este es el nuevo allocator de TensorFlow que maneja mejor la fragmentación
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    print("   ✓ TF_GPU_ALLOCATOR = cuda_malloc_async")

    # Suprimir mensajes de info de TensorFlow (opcional)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error

    # ======================================================================
    # PASO 2: Importar y configurar TensorFlow programáticamente
    # ======================================================================

    try:
        import tensorflow as tf

        # Verificar GPUs disponibles
        gpus = tf.config.list_physical_devices('GPU')

        if not gpus:
            print("   ⚠️ No se detectaron GPUs físicas")
            print("   ℹ️ Los modelos se entrenarán en CPU")
            print("=" * 70)
            return False

        # Configurar cada GPU detectada
        for i, gpu in enumerate(gpus):
            try:
                # CRÍTICO: Habilitar memory growth
                # Esto DEBE hacerse antes de cualquier operación GPU
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"   ✓ Memory Growth habilitado en GPU {i}: {gpu.name}")

            except RuntimeError as e:
                # Si falla, probablemente GPU ya fue inicializada
                print(f"   ❌ Error configurando GPU {i}: {e}")
                print(f"   ⚠️ GPU ya inicializada, configuración puede no aplicarse")
                return False

        # Verificar GPUs lógicas (después de configuración)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"\n   📊 Resumen:")
        print(f"      - GPUs Físicas: {len(gpus)}")
        print(f"      - GPUs Lógicas: {len(logical_gpus)}")

        # Mostrar información de cada GPU
        for i, gpu in enumerate(gpus):
            print(f"      - GPU {i}: {gpu.name}")

        print("\n   ✅ Configuración GPU completada exitosamente")
        print("   ℹ️ Los modelos intentarán usar GPU primero, CPU como fallback")
        print("=" * 70)
        return True

    except ImportError:
        print("   ❌ ERROR: TensorFlow no instalado")
        print("=" * 70)
        return False

    except Exception as e:
        print(f"   ❌ ERROR inesperado configurando GPU: {e}")
        print("   ⚠️ Los modelos se entrenarán en CPU")
        print("=" * 70)
        return False


def print_gpu_memory_info():
    """
    Muestra información de memoria GPU (útil para debugging).

    Llamar DESPUÉS de configure_gpu_for_training()
    """
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            print("[GPU INFO] No hay GPUs disponibles")
            return

        print("\n" + "=" * 70)
        print("📊 [GPU MEMORY INFO] Estado de memoria GPU")
        print("=" * 70)

        for i, gpu in enumerate(gpus):
            try:
                # Intentar obtener estadísticas de memoria
                memory_info = tf.config.experimental.get_memory_info(f'GPU:{i}')

                current_mb = memory_info['current'] / (1024**2)
                peak_mb = memory_info['peak'] / (1024**2)

                print(f"\nGPU {i} ({gpu.name}):")
                print(f"   - Memoria Actual: {current_mb:.2f} MB")
                print(f"   - Memoria Pico:   {peak_mb:.2f} MB")

            except Exception as e:
                print(f"\nGPU {i}: No se pudo obtener info de memoria ({e})")

        print("=" * 70 + "\n")

    except Exception as e:
        print(f"[GPU INFO] Error: {e}")


# ======================================================================
# INICIALIZACIÓN AUTOMÁTICA (cuando se importa este módulo)
# ======================================================================

if __name__ != "__main__":
    # Solo configurar si NO estamos ejecutando este archivo directamente
    # (es decir, si lo están importando)
    _gpu_configured = configure_gpu_for_training()
else:
    # Si ejecutan este archivo directamente, mostrar info
    print(__doc__)
    _gpu_configured = configure_gpu_for_training()
    if _gpu_configured:
        print_gpu_memory_info()
