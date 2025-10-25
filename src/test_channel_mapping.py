#!/usr/bin/env python3.12
"""
Script de prueba para el sistema de mapeo de canales.

Uso:
    cd /mnt/CarlosSSD/Carlos/tt/EGG-lab/src
    python3.12 test_channel_mapping.py
"""

from backend.classes.dataset import Dataset
import numpy as np
from pathlib import Path

def test_channel_mapping():
    """Prueba las funciones de mapeo de canales"""
    dataset_name = "nieto_inner_speech"

    print("="*80)
    print("TEST 1: get_channel_mapping()")
    print("="*80)

    mapping = Dataset.get_channel_mapping(dataset_name)
    print(f"Total de canales: {len(mapping)}")
    print(f"Primeros 5 canales: {dict(list(mapping.items())[:5])}")
    print(f"Últimos 3 canales: {dict(list(mapping.items())[-3:])}")

    print("\n" + "="*80)
    print("TEST 2: get_channel_index()")
    print("="*80)

    test_channels = ["A1", "A2", "B5", "Status", "FAKE_CHANNEL"]
    for ch in test_channels:
        idx = Dataset.get_channel_index(dataset_name, ch)
        print(f"Canal '{ch}' → índice {idx}")

    print("\n" + "="*80)
    print("TEST 3: get_channels_indices()")
    print("="*80)

    selected_channels = ["A1", "A2", "B5", "C10", "Status"]
    indices = Dataset.get_channels_indices(dataset_name, selected_channels)
    print(f"Canales seleccionados: {selected_channels}")
    print(f"Índices obtenidos: {indices}")

    print("\n" + "="*80)
    print("TEST 4: get_all_channel_names()")
    print("="*80)

    all_channels = Dataset.get_all_channel_names(dataset_name)
    print(f"Total de canales: {len(all_channels)}")
    print(f"Primeros 10: {all_channels[:10]}")
    print(f"Últimos 3: {all_channels[-3:]}")

    print("\n" + "="*80)
    print("TEST 5: extract_channels()")
    print("="*80)

    # Buscar primer evento disponible
    events_dir = Path(f"Aux/{dataset_name}").rglob("Events")
    event_file = None

    for events_path in events_dir:
        if events_path.is_dir():
            npy_files = list(events_path.glob("*.npy"))
            if npy_files:
                event_file = npy_files[0]
                break

    if event_file:
        print(f"Cargando evento: {event_file.name}")
        evento_full = np.load(event_file)
        print(f"Shape original: {evento_full.shape}")

        # Filtrar por canales
        channels_to_extract = ["A1", "A2", "B5"]
        evento_filtrado = Dataset.extract_channels(evento_full, channels_to_extract, dataset_name)
        print(f"Canales extraídos: {channels_to_extract}")
        print(f"Shape filtrado: {evento_filtrado.shape}")

        # Verificar que los datos son correctos
        print(f"\nVerificación: comparando datos originales con filtrados")
        indices_expected = Dataset.get_channels_indices(dataset_name, channels_to_extract)
        for i, (ch_name, idx) in enumerate(zip(channels_to_extract, indices_expected)):
            original_row = evento_full[idx, :]
            filtered_row = evento_filtrado[i, :]
            match = np.array_equal(original_row, filtered_row)
            print(f"  Canal {ch_name} (idx {idx}): {'✓ MATCH' if match else '✗ NO MATCH'}")

        print("\n" + "="*80)
        print("TEST 6: load_event_with_channels()")
        print("="*80)

        result = Dataset.load_event_with_channels(str(event_file), channels_to_extract, dataset_name)
        print(f"Evento cargado: {result['event_path']}")
        print(f"Canales solicitados: {result['channel_names']}")
        print(f"Índices: {result['channel_indices']}")
        print(f"Shape original: {result['original_shape']}")
        print(f"Shape filtrado: {result['data'].shape}")

    else:
        print("⚠️ No se encontraron eventos para probar extract_channels()")

    print("\n" + "="*80)
    print("✅ TODOS LOS TESTS COMPLETADOS")
    print("="*80)

if __name__ == "__main__":
    test_channel_mapping()
