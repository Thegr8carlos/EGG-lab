"""
Componente de Visualización en Tiempo Real para Simulación P300 + Inner Speech
================================================================================

Adaptado de DataSimulation.py, este componente procesa ventanas con modelos
y muestra resultados en tiempo real usando clientside callbacks para performance.
"""

from dash import html, dcc, clientside_callback, Output, Input, State, callback, no_update
import dash_bootstrap_components as dbc
from typing import Optional, Dict, Any, List
import numpy as np


def create_realtime_simulation_view(
    container_id: str = "sim-realtime-container",
    store_results_id: str = "sim-results-store",
    store_metadata_id: str = "sim-metadata-store",
    graph_id: str = "sim-realtime-graph",
    stats_id: str = "sim-realtime-stats",
    interval_id: str = "sim-realtime-interval",
    progress_id: str = "sim-realtime-progress",
    interval_ms: int = 2000,  # 2 segundos por batch
):
    """
    Crea el contenedor de visualización en tiempo real para simulación.

    Args:
        container_id: ID del contenedor principal
        store_results_id: ID del Store para resultados procesados
        store_metadata_id: ID del Store para metadata (modelos, señal, etc)
        graph_id: ID del gráfico principal
        stats_id: ID del panel de estadísticas
        interval_id: ID del Interval component
        progress_id: ID de la barra de progreso
        interval_ms: Intervalo en ms (default 2000 = 2 seg)

    Returns:
        html.Div con todos los componentes
    """

    return html.Div(
        id=container_id,
        children=[
            # Stores para datos y resultados
            dcc.Store(id=store_results_id, data={
                "results": [],
                "current_window": 0,
                "total_windows": 0,
                "is_complete": False
            }),
            dcc.Store(id=store_metadata_id),  # Para guardar config de modelos y señal

            # Barra de progreso
            html.Div(id=progress_id, children=[
                dbc.Alert([
                    html.H5([
                        html.I(className="fas fa-play-circle me-2"),
                        "⚡ Modo Tiempo Real - Preparando..."
                    ]),
                    html.P("Esperando inicio de simulación..."),
                    dbc.Progress(value=0, striped=True, animated=True, color="info")
                ], color="info")
            ]),

            # Panel de estadísticas en tiempo real
            html.Div(id=stats_id, style={"marginBottom": "20px"}),

            # Gráfico principal (renderizado con clientside callback)
            dcc.Graph(
                id=graph_id,
                config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToRemove': ['lasso2d', 'select2d']
                },
                style={"height": "600px"}
            ),

            # Interval para procesamiento incremental
            dcc.Interval(
                id=interval_id,
                interval=interval_ms,
                n_intervals=0,
                disabled=True  # Se habilita cuando se inicia simulación
            ),
        ],
        style={
            "padding": "20px",
            "backgroundColor": "rgba(0,0,0,0.15)",
            "borderRadius": "12px",
            "border": "1px solid rgba(255,255,255,0.1)"
        }
    )


def register_realtime_simulation_callbacks(
    interval_id: str,
    store_results_id: str,
    store_metadata_id: str,
    progress_id: str,
    stats_id: str,
    graph_id: str
):
    """
    Registra todos los callbacks necesarios para la simulación en tiempo real.

    Args:
        interval_id: ID del Interval component
        store_results_id: ID del Store de resultados
        store_metadata_id: ID del Store de metadata
        progress_id: ID de la barra de progreso
        stats_id: ID del panel de estadísticas
        graph_id: ID del gráfico
    """

    # ========================================================================
    # CALLBACK 1: Procesamiento de batches (SERVER-SIDE)
    # ========================================================================
    @callback(
        Output(store_results_id, "data", allow_duplicate=True),
        Output(progress_id, "children", allow_duplicate=True),
        Output(interval_id, "disabled", allow_duplicate=True),
        Input(interval_id, "n_intervals"),
        State(store_results_id, "data"),
        State(store_metadata_id, "data"),
        prevent_initial_call=True
    )
    def process_batch(n_intervals, results_data, metadata):
        """
        Procesa un batch de ventanas con los modelos P300 e Inner Speech.
        """
        print(f"\n[process_batch] ⚡ Callback triggered! n_intervals={n_intervals}")
        print(f"[process_batch] results_data: {results_data}")
        print(f"[process_batch] metadata: {metadata}")

        if not metadata or not results_data:
            print(f"[process_batch] ⚠️ Datos faltantes! metadata={metadata is not None}, results_data={results_data is not None}")
            return no_update, no_update, no_update

        # Extraer metadata
        from backend.helpers.simulation_engine import SimulationEngine

        engine = metadata.get('engine_cache_key')  # Usaremos cache global
        batch_size = metadata.get('batch_size', 100)

        # Acceder al cache global
        from app.pages.simulation import _realtime_cache

        if not _realtime_cache.get("engine"):
            error_view = dbc.Alert("Error: Motor de simulación no disponible", color="danger")
            return results_data, error_view, True

        engine = _realtime_cache["engine"]
        current_window = results_data["current_window"]
        total_windows = results_data["total_windows"]
        accumulated_results = results_data["results"].copy()  # ¡IMPORTANTE! Copiar lista

        # Si ya terminamos
        if current_window >= total_windows:
            # CREAR NUEVO DICCIONARIO (no mutar el existente)
            new_results_data = {
                "results": accumulated_results,
                "current_window": current_window,
                "total_windows": total_windows,
                "is_complete": True
            }

            final_progress = dbc.Alert([
                html.H5([
                    html.I(className="fas fa-check-circle me-2"),
                    "✅ Procesamiento Completado"
                ], className="mb-2"),
                html.P(f"Total de ventanas procesadas: {len(accumulated_results)}"),
                dbc.Progress(value=100, color="success", style={"height": "25px"})
            ], color="success")

            print(f"[process_batch] ✅ Simulación completada: {len(accumulated_results)} ventanas")

            return new_results_data, final_progress, True  # Deshabilitar interval

        # Procesar siguiente batch
        end_window = min(current_window + batch_size, total_windows)

        print(f"\n[process_batch] 🔄 Procesando ventanas {current_window}-{end_window-1} (batch #{n_intervals+1})")
        print(f"[process_batch] 📝 accumulated_results tiene {len(accumulated_results)} resultados ANTES de procesar")

        try:
            for i in range(current_window, end_window):
                result = engine.process_window(i)
                accumulated_results.append(result)
                if (i - current_window) % 5 == 0:  # Log cada 5 ventanas
                    print(f"[process_batch] ⏳ Procesadas {i - current_window + 1}/{end_window - current_window} ventanas...")
        except Exception as e:
            print(f"[process_batch] ❌ ERROR procesando ventana {i}: {e}")
            import traceback
            traceback.print_exc()

        print(f"[process_batch] 📝 accumulated_results tiene {len(accumulated_results)} resultados DESPUÉS de procesar")

        # CREAR NUEVO DICCIONARIO (Dash necesita nuevo objeto para detectar cambio)
        new_results_data = {
            "results": accumulated_results,
            "current_window": end_window,
            "total_windows": total_windows,
            "is_complete": False
        }

        # Calcular progreso
        progress_pct = (end_window / total_windows) * 100

        print(f"[process_batch] ✅ Batch procesado. Progreso: {end_window}/{total_windows} ({progress_pct:.1f}%)")
        print(f"[process_batch] 📊 new_results_data: current_window={new_results_data['current_window']}, total_results={len(new_results_data['results'])}")

        # Vista de progreso
        progress_view = dbc.Alert([
            html.H5([
                html.I(className="fas fa-sync fa-spin me-2"),
                f"⚡ Procesando en Tiempo Real"
            ], className="mb-2"),
            html.P(f"Ventanas: {end_window}/{total_windows} ({progress_pct:.1f}%) | Batch #{n_intervals+1}", className="mb-2"),
            dbc.Progress(
                value=progress_pct,
                striped=True,
                animated=True,
                color="success",
                style={"height": "25px"}
            )
        ], color="info")

        print(f"[process_batch] 🎨 Actualizando Store con {len(accumulated_results)} resultados")
        print(f"[process_batch] 🔄 Retornando nuevo dict: current_window={new_results_data['current_window']}, disabled=False\n")

        return new_results_data, progress_view, False  # Continuar


    # ========================================================================
    # CALLBACK 2: Actualizar estadísticas (SERVER-SIDE)
    # ========================================================================
    @callback(
        Output(stats_id, "children"),
        Input(store_results_id, "data"),
        State(store_metadata_id, "data"),
        prevent_initial_call=True
    )
    def update_stats(results_data, metadata):
        """
        Actualiza el panel de estadísticas con los resultados procesados.
        """
        if not results_data or not results_data.get("results"):
            return html.Div()

        results = results_data["results"]

        if len(results) == 0:
            return html.Div()

        # Calcular métricas
        p300_detected = sum(1 for r in results if r.get('p300_prediction') == 1)
        inner_predictions = [r.get('inner_prediction') for r in results if r.get('inner_prediction') is not None]

        # Obtener clases desde metadata
        inner_classes = metadata.get('inner_classes', []) if metadata else []

        # Contar por clase
        from collections import Counter
        inner_counts = Counter(inner_predictions)

        # Crear vista de estadísticas
        stats_view = dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H3(f"{len(results)}", className="text-primary mb-0"),
                        html.Small("Ventanas", className="text-muted")
                    ])
                ], className="text-center")
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H3(f"{p300_detected}", className="text-info mb-0"),
                        html.Small("P300 Detectados", className="text-muted")
                    ])
                ], className="text-center")
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H3(f"{len(inner_predictions)}", className="text-success mb-0"),
                        html.Small("Clasificaciones", className="text-muted")
                    ])
                ], className="text-center")
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            dbc.Badge(
                                f"{inner_classes[pred_idx] if pred_idx < len(inner_classes) else f'C{pred_idx}'}: {count}",
                                color="primary",
                                className="me-1 mb-1",
                                style={"fontSize": "11px"}
                            )
                            for pred_idx, count in sorted(inner_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                        ]) if inner_predictions else html.Small("Sin datos", className="text-muted")
                    ])
                ], className="text-center")
            ], width=3)
        ], className="mb-3")

        return stats_view


    # ========================================================================
    # CALLBACK 3: Renderizar gráfico (CLIENT-SIDE para performance)
    # ========================================================================
    clientside_callback(
        """
        function(results_data, metadata) {
            // Validar datos
            if (!results_data || !results_data.results || results_data.results.length === 0) {
                return {
                    data: [],
                    layout: {
                        title: "⏳ Esperando datos...",
                        xaxis: {title: "Tiempo (s)"},
                        yaxis: {title: "Señal EEG"},
                        height: 500
                    }
                };
            }

            if (!metadata) {
                return window.dash_clientside.no_update;
            }

            const results = results_data.results;
            const sfreq = metadata.sfreq || 1024.0;
            const inner_classes = metadata.inner_classes || [];

            // Crear trazas
            const traces = [];

            // TRAZA 1: Marcadores de P300
            const p300_windows = results.filter(r => r.p300_prediction === 1);
            const p300_times = p300_windows.map(r => r.time_sec);
            const p300_confidences = p300_windows.map(r => r.p300_confidence);

            traces.push({
                x: p300_times,
                y: Array(p300_times.length).fill(1),
                mode: 'markers',
                name: '🔍 P300 Detectado',
                marker: {
                    color: 'rgba(0, 255, 240, 0.8)',
                    size: 12,
                    symbol: 'diamond',
                    line: {color: 'white', width: 1}
                },
                text: p300_confidences.map(c => `Confianza: ${(c * 100).toFixed(1)}%`),
                hovertemplate: '%{text}<extra></extra>',
                yaxis: 'y'
            });

            // TRAZA 2: Clasificaciones Inner Speech
            const inner_windows = results.filter(r => r.inner_prediction !== null && r.inner_prediction !== undefined);
            const inner_times = inner_windows.map(r => r.time_sec);
            const inner_preds = inner_windows.map(r => {
                const idx = r.inner_prediction;
                return idx < inner_classes.length ? inner_classes[idx] : `Class ${idx}`;
            });
            const inner_confidences = inner_windows.map(r => r.inner_confidence);

            // Colores por clase
            const class_colors = {
                'rest': 'rgba(128, 128, 128, 0.8)',
                'arriba': 'rgba(255, 0, 0, 0.8)',
                'abajo': 'rgba(0, 255, 0, 0.8)',
                'izquierda': 'rgba(0, 0, 255, 0.8)',
                'derecha': 'rgba(255, 255, 0, 0.8)'
            };

            traces.push({
                x: inner_times,
                y: Array(inner_times.length).fill(0.5),
                mode: 'markers+text',
                name: '🎯 Clasificación Inner',
                marker: {
                    color: inner_preds.map(pred => class_colors[pred] || 'rgba(255, 0, 229, 0.8)'),
                    size: 10,
                    symbol: 'square'
                },
                text: inner_preds,
                textposition: 'top center',
                textfont: {size: 9, color: 'white'},
                hovertemplate: '%{text}<br>Confianza: %{customdata:.1f}%<extra></extra>',
                customdata: inner_confidences.map(c => c * 100),
                yaxis: 'y'
            });

            // TRAZA 3: Timeline de ventanas procesadas
            const all_times = results.map(r => r.time_sec);
            const all_labels = results.map(r => r.label_real);

            // Agrupar por label para colorear
            const label_colors = {
                'rest': 'rgba(100, 100, 100, 0.3)',
                'arriba': 'rgba(255, 0, 0, 0.3)',
                'abajo': 'rgba(0, 255, 0, 0.3)',
                'izquierda': 'rgba(0, 0, 255, 0.3)',
                'derecha': 'rgba(255, 255, 0, 0.3)'
            };

            traces.push({
                x: all_times,
                y: Array(all_times.length).fill(0),
                mode: 'markers',
                name: '📊 Timeline',
                marker: {
                    color: all_labels.map(label => label_colors[label] || 'rgba(200, 200, 200, 0.3)'),
                    size: 4,
                    symbol: 'line-ns-open'
                },
                text: all_labels,
                hovertemplate: 'Label real: %{text}<extra></extra>',
                yaxis: 'y',
                showlegend: false
            });

            // Layout
            const layout = {
                title: `🎮 Simulación en Tiempo Real - ${results.length} ventanas procesadas`,
                xaxis: {
                    title: 'Tiempo (segundos)',
                    gridcolor: 'rgba(255, 255, 255, 0.1)',
                    showgrid: true
                },
                yaxis: {
                    title: 'Detecciones',
                    range: [-0.2, 1.5],
                    gridcolor: 'rgba(255, 255, 255, 0.1)',
                    showticklabels: false
                },
                plot_bgcolor: 'rgba(10, 1, 36, 0.95)',
                paper_bgcolor: 'rgba(10, 1, 36, 0.95)',
                font: {color: 'white', size: 12},
                height: 500,
                showlegend: true,
                legend: {
                    x: 1.02,
                    y: 1,
                    xanchor: 'left',
                    bgcolor: 'rgba(10, 1, 36, 0.8)',
                    bordercolor: 'rgba(255, 255, 255, 0.3)',
                    borderwidth: 1
                },
                hovermode: 'closest',
                margin: {t: 60, b: 60, l: 80, r: 150}
            };

            return {data: traces, layout: layout};
        }
        """,
        Output(graph_id, "figure"),
        [Input(store_results_id, "data"), Input(store_metadata_id, "data")]
    )

    print(f"[SimulationRealtime] ✅ Callbacks registrados para IDs: {interval_id}, {graph_id}")
