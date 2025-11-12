# app/components/TransformationViewer.py
"""
Componente reusable para visualización de transformaciones.
Puede ser usado en P300, Inner Speech, o extractores legacy.
"""
from dash import html, dcc, callback, Output, Input, State, no_update, ALL, ctx, clientside_callback
import dash_bootstrap_components as dbc
from typing import Optional
import time
import numpy as np


def get_transformation_viewer(
    prefix: str = "p300",
    model_type: str = "p300",
    show_playground: bool = True,
    show_sidebar: bool = True
):
    """
    Crea un visor de transformaciones reusable con IDs únicos.

    Args:
        prefix: Prefijo para IDs (ej: "p300", "inner", "extractores")
        model_type: Tipo de modelo para el pipeline ("p300", "inner", o "legacy")
        show_playground: Si mostrar el área de gráficas
        show_sidebar: Si mostrar la barra lateral izquierda

    Returns:
        html.Div con toda la estructura de visualización
    """

    # IDs únicos con prefijo
    DATA_STORE = f"signal-store-{prefix}"
    TRANSFORMED_DATA_STORE = f"transformed-signal-store-{prefix}"
    PIPELINE_UPDATE_TRIGGER = f"pipeline-update-trigger-{prefix}"
    AUTO_APPLY_PIPELINE = f"auto-apply-pipeline-{prefix}"
    PLOTS_CONTAINER = f"plots-container-{prefix}"
    CHANNEL_RANGE_STORE = f"channel-range-store-{prefix}"
    SELECTED_CLASS_STORE = f"selected-class-store-{prefix}"
    SELECTED_CHANNELS_STORE = f"selected-channels-store-{prefix}"

    # Navegación de canales
    BTN_PREV_CHANNELS = f"btn-prev-channels-{prefix}"
    BTN_NEXT_CHANNELS = f"btn-next-channels-{prefix}"
    CHANNEL_NAV_INFO = f"channel-nav-info-{prefix}"

    # Filtro por clase
    BTN_ALL_CLASSES = f"btn-all-classes-{prefix}"
    BTN_FILTER_CLASS = f"btn-filter-class-{prefix}"

    # Selector de canales
    CHECKLIST_CHANNELS = f"checklist-channel-selection-{prefix}"
    BTN_SELECT_ALL = f"btn-select-all-channels-{prefix}"
    BTN_CLEAR = f"btn-clear-channels-{prefix}"
    BTN_ONLY_EEG = f"btn-only-eeg-channels-{prefix}"
    CHANNEL_COUNT_DISPLAY = f"channel-count-display-{prefix}"

    # Historial de pipeline
    PIPELINE_HISTORY_VIEWER = f"pipeline-history-viewer-{prefix}"
    MODAL_HISTORY_JSON = f"modal-history-json-{prefix}"
    MODAL_HISTORY_JSON_TITLE = f"modal-history-json-title-{prefix}"
    MODAL_HISTORY_JSON_CONTENT = f"modal-history-json-content-{prefix}"
    BTN_VIEW_JSON = f"btn-view-json-{prefix}"

    # Toggle auto-apply
    TOGGLE_AUTO_APPLY = f"toggle-auto-apply-{prefix}"

    # Stores
    stores = [
        dcc.Store(id=DATA_STORE),
        dcc.Store(id=TRANSFORMED_DATA_STORE),
        dcc.Store(id=PIPELINE_UPDATE_TRIGGER, data=0),
        dcc.Store(id=AUTO_APPLY_PIPELINE, data=True),
        dcc.Store(id=CHANNEL_RANGE_STORE, data={"start": 0, "count": 8}),
        dcc.Store(id=SELECTED_CLASS_STORE, data=None),
        dcc.Store(id=SELECTED_CHANNELS_STORE, data=None),
        dcc.Store(id=f"model-type-{prefix}", data=model_type),  # Guardar el model_type
    ]

    # Controles de navegación (sidebar izquierdo)
    navigation_controls = create_navigation_controls(
        prefix=prefix,
        CHANNEL_NAV_INFO=CHANNEL_NAV_INFO,
        BTN_PREV_CHANNELS=BTN_PREV_CHANNELS,
        BTN_NEXT_CHANNELS=BTN_NEXT_CHANNELS,
        BTN_ALL_CLASSES=BTN_ALL_CLASSES,
        BTN_FILTER_CLASS=BTN_FILTER_CLASS,
        PIPELINE_HISTORY_VIEWER=PIPELINE_HISTORY_VIEWER,
        TOGGLE_AUTO_APPLY=TOGGLE_AUTO_APPLY,
        CHECKLIST_CHANNELS=CHECKLIST_CHANNELS,
        BTN_SELECT_ALL=BTN_SELECT_ALL,
        BTN_CLEAR=BTN_CLEAR,
        BTN_ONLY_EEG=BTN_ONLY_EEG,
        CHANNEL_COUNT_DISPLAY=CHANNEL_COUNT_DISPLAY
    )

    # Modal para JSON
    modal = dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle(id=MODAL_HISTORY_JSON_TITLE)),
        dbc.ModalBody(html.Pre(id=MODAL_HISTORY_JSON_CONTENT, style={
            "fontSize": "10px",
            "maxHeight": "500px",
            "overflow": "auto",
            "background": "var(--card-bg)",
            "padding": "12px",
            "borderRadius": "var(--radius-sm)",
            "color": "var(--text)"
        })),
    ], id=MODAL_HISTORY_JSON, size="lg", is_open=False)

    # Área de gráficas (dos columnas)
    plots_area = html.Div(
        id=PLOTS_CONTAINER,
        children=[
            html.Div("Cargando visualización...", style={
                "textAlign": "center",
                "padding": "40px",
                "color": "rgba(255,255,255,0.5)"
            })
        ],
        style={
            "flex": "1",
            "minWidth": 0,
            "overflowY": "auto",
            "padding": "12px"
        }
    )

    # Layout completo
    layout_parts = stores + [modal]

    if show_sidebar and show_playground:
        # Layout con sidebar + plots
        layout_parts.append(
            html.Div([
                # Sidebar izquierdo con controles
                html.Div(
                    navigation_controls,
                    style={
                        "width": "260px",
                        "padding": "1rem",
                        "overflowY": "auto",
                        "borderRight": "1px solid var(--border-weak)"
                    }
                ),
                # Área de plots
                plots_area
            ], style={
                "display": "flex",
                "height": "100%",
                "width": "100%"
            })
        )
    elif show_playground:
        # Solo plots (sin sidebar)
        layout_parts.append(plots_area)
    else:
        # Solo controles (para integrar en otra página)
        layout_parts.append(navigation_controls)

    return html.Div(layout_parts, id=f"transformation-viewer-{prefix}")


def register_transformation_callbacks(prefix: str = "p300", model_type: str = "p300"):
    """
    Registra todos los callbacks necesarios para el visor de transformaciones.
    Debe llamarse DESPUÉS de crear el layout con get_transformation_viewer().

    Args:
        prefix: Prefijo para IDs (debe coincidir con get_transformation_viewer)
        model_type: Tipo de modelo para el pipeline ("p300", "inner", o "legacy")
    """
    from backend.classes.Experiment import Experiment
    from backend.classes.dataset import Dataset
    from shared.fileUtils import get_dataset_metadata
    import json

    # IDs (deben coincidir con get_transformation_viewer)
    DATA_STORE = f"signal-store-{prefix}"
    TRANSFORMED_DATA_STORE = f"transformed-signal-store-{prefix}"
    PIPELINE_UPDATE_TRIGGER = f"pipeline-update-trigger-{prefix}"
    AUTO_APPLY_PIPELINE = f"auto-apply-pipeline-{prefix}"
    PLOTS_CONTAINER = f"plots-container-{prefix}"
    CHANNEL_RANGE_STORE = f"channel-range-store-{prefix}"
    SELECTED_CLASS_STORE = f"selected-class-store-{prefix}"
    SELECTED_CHANNELS_STORE = f"selected-channels-store-{prefix}"
    PIPELINE_HISTORY_VIEWER = f"pipeline-history-viewer-{prefix}"
    MODAL_HISTORY_JSON = f"modal-history-json-{prefix}"
    MODAL_HISTORY_JSON_TITLE = f"modal-history-json-title-{prefix}"
    MODAL_HISTORY_JSON_CONTENT = f"modal-history-json-content-{prefix}"
    TOGGLE_AUTO_APPLY = f"toggle-auto-apply-{prefix}"
    CHECKLIST_CHANNELS = f"checklist-channel-selection-{prefix}"
    CHANNEL_COUNT_DISPLAY = f"channel-count-display-{prefix}"

    # ===== CALLBACK: Cargar evento y auto-aplicar pipeline =====
    @callback(
        [
            Output(DATA_STORE, "data"),
            Output(TRANSFORMED_DATA_STORE, "data", allow_duplicate=True),
        ],
        [
            Input("selected-file-path", "data"),
            Input(SELECTED_CLASS_STORE, "data"),
            Input(SELECTED_CHANNELS_STORE, "data")
        ],
        [
            State("selected-dataset", "data"),
            State(AUTO_APPLY_PIPELINE, 'data')
        ],
        prevent_initial_call=True
    )
    def load_event_and_apply_pipeline(selected_file_path, selected_class, selected_channels, dataset_name, auto_apply_enabled):
        """Carga evento y opcionalmente aplica pipeline automáticamente"""
        if selected_file_path is None:
            return no_update, no_update

        if isinstance(selected_file_path, dict):
            candidate = selected_file_path.get("path") or selected_file_path.get("file") or ""
        else:
            candidate = str(selected_file_path)

        candidate = candidate.strip()
        if not candidate:
            return no_update, no_update

        data_payload = no_update
        transformed_payload = {"ts": time.time(), "pipeline_applied": False}

        try:
            # Obtener evento por clase
            res = Dataset.get_events_by_class(candidate, class_name=selected_class)
            first_evt = res.get("first_event_file") if isinstance(res, dict) else None

            if first_evt:
                # Cargar evento (con filtro de canales si aplica)
                if selected_channels and len(selected_channels) > 0 and dataset_name:
                    print(f"[{prefix}] Cargando evento con {len(selected_channels)} canales específicos")
                    result = Dataset.load_event_with_channels(first_evt, selected_channels, dataset_name)
                    arr = result["data"]
                else:
                    print(f"[{prefix}] Cargando evento completo")
                    arr = np.load(first_evt, allow_pickle=False)

                # Metadata básica
                try:
                    meta = get_dataset_metadata(candidate.split('/')[0])
                    sfreq = meta.get("sampling_frequency_hz", 1024.0)
                except Exception:
                    sfreq = 1024.0

                import os, re
                file_name = os.path.basename(first_evt)
                event_class = file_name.split('[')[0].strip() if '[' in file_name else file_name.replace('.npy', '')
                n_samples = arr.shape[1] if arr.ndim == 2 else arr.shape[0]
                duration_sec = n_samples / sfreq

                # Colores de clase y nombres de canales (consistencia con filtros)
                try:
                    meta_full = get_dataset_metadata(dataset_name) if dataset_name else {}
                    classes_meta = meta_full.get("classes", []) or []
                    from shared.class_colors import get_class_color
                    class_color_map = {str(lbl): get_class_color(str(lbl), idx) for idx, lbl in enumerate(classes_meta)}
                except Exception:
                    class_color_map = {}

                if selected_channels and len(selected_channels) > 0:
                    channel_names_for_plots = selected_channels
                else:
                    try:
                        all_channel_names = Dataset.get_all_channel_names(dataset_name) if dataset_name else []
                        channel_names_for_plots = all_channel_names if all_channel_names else [f"Ch{i}" for i in range(arr.shape[0])]
                    except Exception:
                        channel_names_for_plots = [f"Ch{i}" for i in range(arr.shape[0])]

                data_payload = {
                    "source": first_evt,
                    "shape": list(arr.shape),
                    "matrix": arr.tolist(),
                    "ts": time.time(),
                    "sfreq": sfreq,
                    "file_name": file_name.replace('.npy', ''),
                    "event_class": event_class,
                    "duration_sec": round(duration_sec, 3),
                    "class_filter": selected_class,
                    "n_events": res.get("n_events", 0) if isinstance(res, dict) else None,
                    "channel_names": channel_names_for_plots,
                    "selected_channels": selected_channels,
                    "n_channels_selected": len(selected_channels) if selected_channels else arr.shape[0],
                    "class_colors": class_color_map
                }

                if auto_apply_enabled:
                    try:
                        print(f"[{prefix}] Auto-aplicando pipeline (model_type={model_type})…")
                        pipeline_result = Experiment.apply_history_pipeline(
                            file_path=first_evt,
                            model_type=model_type,
                            force_recalculate=False,
                            save_intermediates=True,
                            verbose=True
                        )
                        if pipeline_result and "signal" in pipeline_result:
                            arr_processed = pipeline_result["signal"]
                            transformed_payload = dict(data_payload)
                            transformed_payload.update({
                                "matrix": arr_processed.tolist(),
                                "shape": list(arr_processed.shape),
                                "pipeline_applied": True,
                                "cache_used": pipeline_result.get("cache_used", False),
                                "ts": time.time()
                            })
                            # Attach optional visualization payload (domain-aware cube) if provided
                            if isinstance(pipeline_result.get("viz"), dict):
                                transformed_payload["viz"] = pipeline_result["viz"]
                            # Adjuntar nombre de transform aplicada si viene del backend
                            if isinstance(pipeline_result.get("applied_transform_name"), str):
                                transformed_payload["applied_transform_name"] = pipeline_result["applied_transform_name"]
                            print(f"[{prefix}] Pipeline aplicado: {arr_processed.shape}")
                        else:
                            print(f"[{prefix}] Pipeline no devolvió señal, limpiando transformada")
                            transformed_payload = {"ts": time.time(), "pipeline_applied": False}
                    except Exception as e:
                        print(f"[{prefix}] Error aplicando pipeline: {e}")
                        transformed_payload = {"ts": time.time(), "pipeline_applied": False}
                else:
                    transformed_payload = {"ts": time.time(), "pipeline_applied": False}

        except Exception as e:
            print(f"[{prefix}] ERROR cargando evento: {e}")

        return data_payload, transformed_payload

    # ===== CALLBACK: Toggle auto-apply =====
    @callback(
        Output(AUTO_APPLY_PIPELINE, 'data'),
        Input(TOGGLE_AUTO_APPLY, 'n_clicks'),
        State(AUTO_APPLY_PIPELINE, 'data'),
        prevent_initial_call=True
    )
    def toggle_auto_apply(n_clicks, current_state):
        if n_clicks:
            new_state = not current_state
            print(f"[{prefix}] Auto-apply: {'ON' if new_state else 'OFF'}")
            return new_state
        return current_state

    # ===== CALLBACK: Actualizar estilo del botón toggle =====
    @callback(
        [
            Output(TOGGLE_AUTO_APPLY, 'children'),
            Output(TOGGLE_AUTO_APPLY, 'style')
        ],
        Input(AUTO_APPLY_PIPELINE, 'data')
    )
    def update_toggle_button_style(is_enabled):
        if is_enabled:
            return "ON", {
                "padding": "2px 8px",
                "fontSize": "9px",
                "borderRadius": "3px",
                "border": "1px solid #4CAF50",
                "background": "#4CAF50",
                "color": "white",
                "cursor": "pointer",
                "fontWeight": "600"
            }
        else:
            return "OFF", {
                "padding": "2px 8px",
                "fontSize": "9px",
                "borderRadius": "3px",
                "border": "1px solid #888",
                "background": "#333",
                "color": "#888",
                "cursor": "pointer",
                "fontWeight": "600"
            }

    # ===== CALLBACK: Actualizar historial del pipeline =====
    @callback(
        Output(PIPELINE_HISTORY_VIEWER, 'children'),
        [
            Input('selected-dataset', 'data'),
            Input(PIPELINE_UPDATE_TRIGGER, 'data')
        ]
    )
    def update_pipeline_history(selected_dataset, trigger):
        """Muestra el historial del pipeline con botones JSON (estilo filtros)."""
        if not selected_dataset:
            return html.Div("Selecciona un dataset", style={
                "fontSize": "9px",
                "color": "#888",
                "textAlign": "center",
                "padding": "6px"
            })

        try:
            # Obtener resumen del experimento (sin recalcular caché)
            summary = Experiment.get_experiment_summary(calculate_cache=False)
            experiment_id = summary.get("experiment_id", "N/A")
            filters = summary.get("filters", [])
            transforms = summary.get("transforms", [])
            total_steps = summary.get("total_steps", 0)

            # Determinar transformación actual por modelo (p300/inner)
            current_transform_name = None
            try:
                if str(model_type).lower() == "p300":
                    model_transform = Experiment.get_P300_transform()
                elif str(model_type).lower() == "inner":
                    model_transform = Experiment.get_inner_speech_transform()
                else:
                    model_transform = None
                if isinstance(model_transform, dict) and len(model_transform) > 0:
                    current_transform_name = list(model_transform.keys())[0]
            except Exception:
                current_transform_name = None

            if total_steps == 0:
                current_box = html.Div([
                    html.Div(f"Exp {experiment_id}", style={
                        "fontSize": "9px",
                        "fontWeight": "600",
                        "color": "#ddd",
                        "marginBottom": "3px"
                    }),
                    html.Div([
                        html.Span(
                            f"Transformación por modelo ({model_type.upper()}): ",
                            style={"color": "#bbb", "fontSize": "10px", "marginRight": "4px"}
                        ),
                        html.Span(
                            current_transform_name or "No configurada",
                            style={"color": "#fff", "fontSize": "12px", "fontWeight": "700"}
                        )
                    ], style={
                        "padding": "6px",
                        "border": "1px solid #333",
                        "borderLeft": "3px solid #4A90E2",
                        "borderRadius": "4px",
                        "background": "#141414",
                        "marginBottom": "6px"
                    }),
                    html.Div("Sin pasos aplicados", style={
                        "fontSize": "8px",
                        "color": "#888",
                        "textAlign": "center"
                    })
                ])
                return current_box

            all_steps = []
            # Filtros
            for f in filters:
                filter_id = f.get("id", "?")
                filter_name = f.get("name", "Unknown")
                all_steps.append(
                    html.Div([
                        html.Span(f"F{filter_id}: {filter_name}", style={
                            "fontSize": "8px",
                            "color": "#ddd",
                            "flex": "1"
                        }),
                        html.Button(
                            "JSON",
                            id={"type": f"btn-view-json-{prefix}", "category": "filter", "index": filter_id},
                            n_clicks=0,
                            style={
                                "padding": "1px 4px",
                                "fontSize": "7px",
                                "borderRadius": "2px",
                                "border": "1px solid #555",
                                "background": "#2a2a2a",
                                "color": "#aaa",
                                "cursor": "pointer"
                            }
                        )
                    ], style={
                        "display": "flex",
                        "alignItems": "center",
                        "gap": "4px",
                        "marginBottom": "2px",
                        "padding": "2px 4px",
                        "background": "#1a1a1a",
                        "borderRadius": "3px",
                        "border": "1px solid #333"
                    })
                )

            # Transformadas
            for t in transforms:
                transform_id = t.get("id", "?")
                transform_name = t.get("name", "Unknown")
                all_steps.append(
                    html.Div([
                        html.Span(f"T{transform_id}: {transform_name}", style={
                            "fontSize": "8px",
                            "color": "#ddd",
                            "flex": "1"
                        }),
                        html.Button(
                            "JSON",
                            id={"type": f"btn-view-json-{prefix}", "category": "transform", "index": transform_id},
                            n_clicks=0,
                            style={
                                "padding": "1px 4px",
                                "fontSize": "7px",
                                "borderRadius": "2px",
                                "border": "1px solid #555",
                                "background": "#2a2a2a",
                                "color": "#aaa",
                                "cursor": "pointer"
                            }
                        )
                    ], style={
                        "display": "flex",
                        "alignItems": "center",
                        "gap": "4px",
                        "marginBottom": "2px",
                        "padding": "2px 4px",
                        "background": "#1a1a1a",
                        "borderRadius": "3px",
                        "border": "1px solid #333"
                    })
                )

            # Bloque de encabezado (experimento + transform actual por modelo)
            header = html.Div([
                html.Div([
                    html.Span(f"Exp {experiment_id}", style={
                        "fontSize": "9px",
                        "fontWeight": "600",
                        "color": "#ddd"
                    }),
                    html.Span(f"({total_steps})", style={
                        "fontSize": "8px",
                        "color": "#888",
                        "marginLeft": "4px"
                    })
                ], style={"marginBottom": "4px"}),
                html.Div([
                    html.Span(
                        f"Transformación por modelo ({model_type.upper()}): ",
                        style={"color": "#bbb", "fontSize": "10px", "marginRight": "4px"}
                    ),
                    html.Span(
                        current_transform_name or "No configurada",
                        style={"color": "#fff", "fontSize": "12px", "fontWeight": "700"}
                    )
                ], style={
                    "padding": "6px",
                    "border": "1px solid #333",
                    "borderLeft": "3px solid #4A90E2",
                    "borderRadius": "4px",
                    "background": "#141414",
                    "marginBottom": "6px"
                })
            ])

            return html.Div([
                header,
                html.Div(all_steps, style={
                    "maxHeight": "120px",
                    "overflowY": "auto",
                    "overflowX": "hidden",
                    "marginBottom": "4px"
                })
            ], style={
                "background": "#0f0f0f",
                "padding": "6px",
                "borderRadius": "4px",
                "border": "1px solid #2a2a2a"
            })

        except Exception as e:
            print(f"[{prefix}] Error en historial: {e}")
            return html.Div("Error", style={"fontSize": "8px", "color": "#f88"})

    # ===== MODAL JSON: Abrir y mostrar configuración =====
    @callback(
        [
            Output(MODAL_HISTORY_JSON, 'is_open'),
            Output(MODAL_HISTORY_JSON_TITLE, 'children'),
            Output(MODAL_HISTORY_JSON_CONTENT, 'children')
        ],
        Input({'type': f'btn-view-json-{prefix}', 'category': ALL, 'index': ALL}, 'n_clicks'),
        State(MODAL_HISTORY_JSON, 'is_open'),
        prevent_initial_call=True
    )
    def toggle_json_modal(n_clicks_list, is_open):
        import json as _json
        if not any(n_clicks_list):
            return no_update, no_update, no_update

        triggered = ctx.triggered_id
        if not triggered:
            return no_update, no_update, no_update

        category = triggered.get('category')
        index = triggered.get('index')

        try:
            summary = Experiment.get_experiment_summary()
            item_config = None
            item_name = "Unknown"

            if category == 'filter':
                for f in summary.get('filters', []):
                    if f.get('id') == index:
                        item_config = f.get('config', {})
                        item_name = f.get('name', 'Filter')
                        break
            elif category == 'transform':
                for t in summary.get('transforms', []):
                    if t.get('id') == index:
                        item_config = t.get('config', {})
                        item_name = t.get('name', 'Transform')
                        break

            if item_config is None:
                return True, "Error", "Configuración no encontrada"

            json_str = _json.dumps(item_config, indent=2, ensure_ascii=False)
            title = f"{item_name} [ID: {index}]"
            return True, title, json_str
        except Exception as e:
            print(f"[{prefix}] Error abriendo JSON: {e}")
            return True, "Error", str(e)

    # ===== CALLBACK: Populate channel checklist =====
    @callback(
        Output(CHECKLIST_CHANNELS, 'options'),
        Input('selected-dataset', 'data')
    )
    def populate_channel_checklist(selected_dataset):
        if not selected_dataset:
            return []
        try:
            channel_names = Dataset.get_all_channel_names(selected_dataset)
            return [{"label": ch, "value": ch} for ch in channel_names] if channel_names else []
        except:
            return []

    # ===== CALLBACK: Save selected channels =====
    @callback(
        Output(SELECTED_CHANNELS_STORE, 'data'),
        Input(CHECKLIST_CHANNELS, 'value')
    )
    def save_selected_channels(selected_channels):
        return selected_channels if selected_channels and len(selected_channels) > 0 else None

    # ===== CALLBACK: Update channel count =====
    @callback(
        Output(CHANNEL_COUNT_DISPLAY, 'children'),
        Input(CHECKLIST_CHANNELS, 'value')
    )
    def update_channel_count(selected_channels):
        count = len(selected_channels) if selected_channels else 0
        if count == 0:
            return "Todos los canales"
        elif count == 1:
            return "1 canal seleccionado"
        else:
            return f"{count} canales seleccionados"

    # ===== CALLBACK: Generar botones de filtro por clase =====
    @callback(
        Output(f"class-filter-container-{prefix}", 'children'),
        Input('selected-dataset', 'data')
    )
    def populate_class_filter_buttons(selected_dataset):
        """Genera botones de filtro por clase según el dataset seleccionado"""
        if not selected_dataset:
            return html.Div("Cargando clases...", style={"fontSize": "9px", "color": "#888"})

        try:
            meta = get_dataset_metadata(selected_dataset)
            classes = meta.get("classes", []) or []

            if not classes:
                return html.Div("Sin clases disponibles", style={"fontSize": "9px", "color": "#888"})

            # Botón "Todas"
            btn_all = html.Button(
                'Todas',
                id=f"btn-all-classes-{prefix}",
                n_clicks=0,
                style={
                    "padding": "3px 6px",
                    "flex": "1",
                    "borderRadius": "var(--radius-sm)",
                    "border": "1px solid var(--accent-1)",
                    "background": "var(--accent-1)",
                    "color": "var(--text)",
                    "cursor": "pointer",
                    "fontSize": "10px",
                    "fontWeight": "600",
                    "opacity": "1",
                    "whiteSpace": "nowrap"
                }
            )

            # Botones de clases individuales
            class_buttons = [
                html.Button(
                    str(cls),
                    id={'type': f'btn-filter-class-{prefix}', 'index': idx},
                    n_clicks=0,
                    style={
                        "padding": "3px 6px",
                        "flex": "1",
                        "borderRadius": "var(--radius-sm)",
                        "border": "1px solid var(--border-weak)",
                        "background": "var(--card-bg)",
                        "color": "var(--text)",
                        "cursor": "pointer",
                        "fontSize": "10px",
                        "fontWeight": "500",
                        "opacity": "0.8",
                        "whiteSpace": "nowrap"
                    }
                ) for idx, cls in enumerate(classes)
            ]

            return html.Div([btn_all] + class_buttons, style={
                "display": "flex",
                "gap": "4px",
                "marginBottom": "12px",
                "flexWrap": "wrap"
            })

        except Exception as e:
            print(f"[{prefix}] Error generando botones de clase: {e}")
            return html.Div("Error cargando clases", style={"fontSize": "9px", "color": "#f88"})

    # ===== CALLBACKS DE FILTRO POR CLASE =====

    # Seleccionar todas las clases
    @callback(
        Output(SELECTED_CLASS_STORE, "data", allow_duplicate=True),
        Input(f"btn-all-classes-{prefix}", "n_clicks"),
        prevent_initial_call=True
    )
    def select_all_classes(n_clicks):
        if not n_clicks:
            return no_update
        print(f"[{prefix}] Seleccionado: TODAS las clases")
        return None

    # Seleccionar clase específica
    @callback(
        Output(SELECTED_CLASS_STORE, "data", allow_duplicate=True),
        Input({'type': f'btn-filter-class-{prefix}', 'index': ALL}, 'n_clicks'),
        State("selected-dataset", "data"),
        prevent_initial_call=True
    )
    def select_specific_class(n_clicks_list, selected_dataset):
        if not any(n_clicks_list):
            return no_update

        try:
            meta = get_dataset_metadata(selected_dataset)
            classes = meta.get("classes", [])
        except:
            return no_update

        triggered = ctx.triggered_id
        if not triggered:
            return no_update

        class_index = triggered.get('index', -1)

        if 0 <= class_index < len(classes):
            selected_class = classes[class_index]
            print(f"[{prefix}] Seleccionado clase: {selected_class}")
            return selected_class

        return no_update

    # Actualizar estilo del botón "Todas"
    @callback(
        Output(f"btn-all-classes-{prefix}", "style"),
        Input(SELECTED_CLASS_STORE, "data")
    )
    def update_all_button_style(selected_class):
        is_selected = selected_class is None
        return {
            "padding": "3px 6px",
            "flex": "1",
            "borderRadius": "var(--radius-sm)",
            "border": f"1px solid {'var(--accent-1)' if is_selected else 'var(--border-weak)'}",
            "background": "var(--accent-1)" if is_selected else "var(--card-bg)",
            "color": "var(--text)",
            "cursor": "pointer",
            "fontSize": "10px",
            "fontWeight": "600" if is_selected else "500",
            "opacity": "1" if is_selected else "0.7",
            "whiteSpace": "nowrap"
        }

    # Actualizar estilos de botones de clase
    @callback(
        Output({'type': f'btn-filter-class-{prefix}', 'index': ALL}, 'style'),
        Input(SELECTED_CLASS_STORE, "data"),
        State("selected-dataset", "data")
    )
    def update_class_buttons_style(selected_class, selected_dataset):
        try:
            meta = get_dataset_metadata(selected_dataset)
            classes = meta.get("classes", [])
        except:
            classes = []

        styles = []
        for cls in classes:
            is_selected = selected_class == cls
            styles.append({
                "padding": "3px 6px",
                "flex": "1",
                "borderRadius": "var(--radius-sm)",
                "border": f"1px solid {'var(--accent-1)' if is_selected else 'var(--border-weak)'}",
                "background": "var(--accent-1)" if is_selected else "var(--card-bg)",
                "color": "var(--text)",
                "cursor": "pointer",
                "fontSize": "10px",
                "fontWeight": "600" if is_selected else "500",
                "opacity": "1" if is_selected else "0.7",
                "whiteSpace": "nowrap"
            })

        return styles

    # ===== CALLBACKS DE MANEJO DE SELECCIÓN DE CANALES (botones helper) =====
    @callback(
        Output(CHECKLIST_CHANNELS, 'value', allow_duplicate=True),
        [
            Input(f"btn-select-all-channels-{prefix}", 'n_clicks'),
            Input(f"btn-clear-channels-{prefix}", 'n_clicks'),
            Input(f"btn-only-eeg-channels-{prefix}", 'n_clicks')
        ],
        [
            State(CHECKLIST_CHANNELS, 'options'),
            State(CHECKLIST_CHANNELS, 'value')
        ],
        prevent_initial_call=True
    )
    def handle_channel_buttons(n_all, n_clear, n_eeg, options, current_value):
        if not ctx.triggered:
            return no_update

        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if button_id == f"btn-select-all-channels-{prefix}":
            all_channels = [opt['value'] for opt in options]
            print(f"[{prefix}] Seleccionados todos ({len(all_channels)} canales)")
            return all_channels

        elif button_id == f"btn-clear-channels-{prefix}":
            print(f"[{prefix}] Limpiados todos los canales")
            return []

        elif button_id == f"btn-only-eeg-channels-{prefix}":
            eeg_channels = [opt['value'] for opt in options if opt['value'] != 'Status' and not opt['value'].startswith('EXG')]
            print(f"[{prefix}] Seleccionados solo EEG ({len(eeg_channels)} canales)")
            return eeg_channels

        return no_update

    # ===== CLIENTSIDE CALLBACKS DE NAVEGACIÓN DE CANALES =====

    BTN_PREV_CHANNELS = f"btn-prev-channels-{prefix}"
    BTN_NEXT_CHANNELS = f"btn-next-channels-{prefix}"
    CHANNEL_NAV_INFO = f"channel-nav-info-{prefix}"

    # Navegación anterior
    clientside_callback(
        f"""
        function(n_clicks, currentRange) {{
          if (!n_clicks || n_clicks === 0) return window.dash_clientside.no_update;
                    const CHANNELS_PER_PAGE = (currentRange && currentRange.count) ? currentRange.count : 8;
          const currentStart = (currentRange && currentRange.start) || 0;
          const newStart = Math.max(0, currentStart - CHANNELS_PER_PAGE);
          return {{start: newStart, count: CHANNELS_PER_PAGE}};
        }}
        """,
        Output(CHANNEL_RANGE_STORE, 'data', allow_duplicate=True),
        Input(BTN_PREV_CHANNELS, 'n_clicks'),
        State(CHANNEL_RANGE_STORE, 'data'),
        prevent_initial_call=True
    )

    # Navegación siguiente
    clientside_callback(
        f"""
        function(n_clicks, currentRange, signalData) {{
          if (!n_clicks || n_clicks === 0) return window.dash_clientside.no_update;
          if (!(signalData && Array.isArray(signalData.matrix))) return window.dash_clientside.no_update;

                    const CHANNELS_PER_PAGE = (currentRange && currentRange.count) ? currentRange.count : 8;
          const total = signalData.matrix.length;
          const currentStart = (currentRange && currentRange.start) || 0;
          const newStart = Math.min(total - CHANNELS_PER_PAGE, currentStart + CHANNELS_PER_PAGE);
          return {{start: newStart, count: CHANNELS_PER_PAGE}};
        }}
        """,
        Output(CHANNEL_RANGE_STORE, 'data', allow_duplicate=True),
        Input(BTN_NEXT_CHANNELS, 'n_clicks'),
        [State(CHANNEL_RANGE_STORE, 'data'), State(DATA_STORE, 'data')],
        prevent_initial_call=True
    )

    # Actualizar texto de información de canales
    clientside_callback(
        f"""
        function(channelRange, signalData, selectedChannels) {{
          if (!(signalData && Array.isArray(signalData.matrix))) {{
            return "Canales 0 - 0 de 0";
          }}

          if (selectedChannels && Array.isArray(selectedChannels) && selectedChannels.length > 0) {{
            return `${{selectedChannels.length}} canales seleccionados`;
          }}

          const CHANNELS_PER_PAGE = (channelRange && channelRange.count) ? channelRange.count : 8;
          const total = signalData.matrix.length;
          const start = (channelRange && channelRange.start) || 0;
          const count = Math.min(CHANNELS_PER_PAGE, total - start);
          const end = start + count - 1;

          return `Canales ${{start}} - ${{end}} de ${{total}}`;
        }}
        """,
        Output(CHANNEL_NAV_INFO, 'children'),
        [Input(CHANNEL_RANGE_STORE, 'data'), Input(DATA_STORE, 'data'), Input(SELECTED_CHANNELS_STORE, 'data')]
    )

    # Actualizar estilo botón anterior
    clientside_callback(
        f"""
        function(channelRange, signalData) {{
          if (!(signalData && Array.isArray(signalData.matrix))) {{
            return {{
              padding: '3px 8px',
              borderRadius: 'var(--radius-sm)',
              border: 'none',
              background: 'var(--card-bg)',
              color: 'var(--text)',
              cursor: 'not-allowed',
              fontSize: '10px',
              fontWeight: '500',
              opacity: '0.5',
              flex: '1'
            }};
          }}

          const start = (channelRange && channelRange.start) || 0;
          const isDisabled = start === 0;

          return {{
            padding: '3px 8px',
            borderRadius: 'var(--radius-sm)',
            border: 'none',
            background: isDisabled ? 'var(--card-bg)' : 'var(--accent-1)',
            color: 'var(--text)',
            cursor: isDisabled ? 'not-allowed' : 'pointer',
            fontSize: '10px',
            fontWeight: '500',
            opacity: isDisabled ? '0.5' : '1',
            flex: '1'
          }};
        }}
        """,
        Output(BTN_PREV_CHANNELS, 'style'),
        [Input(CHANNEL_RANGE_STORE, 'data'), Input(DATA_STORE, 'data')]
    )

    # Actualizar estilo botón siguiente
    clientside_callback(
        f"""
        function(channelRange, signalData) {{
          if (!(signalData && Array.isArray(signalData.matrix))) {{
            return {{
              padding: '3px 8px',
              borderRadius: 'var(--radius-sm)',
              border: 'none',
              background: 'var(--card-bg)',
              color: 'var(--text)',
              cursor: 'not-allowed',
              fontSize: '10px',
              fontWeight: '500',
              opacity: '0.5',
              flex: '1'
            }};
          }}

                    const CHANNELS_PER_PAGE = (channelRange && channelRange.count) ? channelRange.count : 8;
          const total = signalData.matrix.length;
          const start = (channelRange && channelRange.start) || 0;
          const isDisabled = start + CHANNELS_PER_PAGE >= total;

          return {{
            padding: '3px 8px',
            borderRadius: 'var(--radius-sm)',
            border: 'none',
            background: isDisabled ? 'var(--card-bg)' : 'var(--accent-1)',
            color: 'var(--text)',
            cursor: isDisabled ? 'not-allowed' : 'pointer',
            fontSize: '10px',
            fontWeight: '500',
            opacity: isDisabled ? '0.5' : '1',
            flex: '1'
          }};
        }}
        """,
        Output(BTN_NEXT_CHANNELS, 'style'),
        [Input(CHANNEL_RANGE_STORE, 'data'), Input(DATA_STORE, 'data')]
    )

    # Actualizar disabled botón anterior
    clientside_callback(
        f"""
        function(channelRange, signalData, selectedChannels) {{
          if (!(signalData && Array.isArray(signalData.matrix))) return true;

          if (selectedChannels && Array.isArray(selectedChannels) && selectedChannels.length > 0) {{
            return true;
          }}

          const start = (channelRange && channelRange.start) || 0;
          return start === 0;
        }}
        """,
        Output(BTN_PREV_CHANNELS, 'disabled'),
        [Input(CHANNEL_RANGE_STORE, 'data'), Input(DATA_STORE, 'data'), Input(SELECTED_CHANNELS_STORE, 'data')]
    )

    # Actualizar disabled botón siguiente
    clientside_callback(
        f"""
        function(channelRange, signalData, selectedChannels) {{
          if (!(signalData && Array.isArray(signalData.matrix))) return true;

          if (selectedChannels && Array.isArray(selectedChannels) && selectedChannels.length > 0) {{
            return true;
          }}

                    const CHANNELS_PER_PAGE = (channelRange && channelRange.count) ? channelRange.count : 8;
          const total = signalData.matrix.length;
          const start = (channelRange && channelRange.start) || 0;
          return start + CHANNELS_PER_PAGE >= total;
        }}
        """,
        Output(BTN_NEXT_CHANNELS, 'disabled'),
        [Input(CHANNEL_RANGE_STORE, 'data'), Input(DATA_STORE, 'data'), Input(SELECTED_CHANNELS_STORE, 'data')]
    )

    # ===== CLIENTSIDE CALLBACK PRINCIPAL: VISUALIZACIÓN DE DOS COLUMNAS =====
    clientside_callback(
        """
        function(signalData, transformedData, channelRange, selectedChannels) {
          console.log('[TransformViewer] Callback ejecutado - versión con heatmap');
          try {
            const USE_WEBGL = false;
            const USE_DOWNSAMPLING = false;
            const DS_FACTOR = 2;
            const MAX_POINTS = 15000;
                        const CHANNELS_PER_PAGE = (channelRange && channelRange.count) ? channelRange.count : 8;

            function darkenHSL(hslColor, amount = 20) {
              const match = hslColor.match(/hsl\\((\\d+),\\s*(\\d+)%,\\s*(\\d+)%\\)/);
              if (!match) return hslColor;
              const h = parseInt(match[1]);
              const s = parseInt(match[2]);
              const l = parseInt(match[3]);
              const newL = Math.max(0, l - amount);
              return `hsl(${h}, ${s}%, ${newL}%)`;
            }

            function downsampling(xArr, yArr, opts) {
              if (!Array.isArray(yArr) || yArr.length === 0) return { x: xArr, y: yArr };
              const factor = Math.max(1, (opts && opts.factor) ? opts.factor : 1);
              const maxPts = Math.max(0, (opts && opts.maxPoints) ? opts.maxPoints : 0);
              let eff = factor;
              if (maxPts > 0 && yArr.length > maxPts) eff = Math.max(eff, Math.ceil(yArr.length / maxPts));
              if (eff <= 1) return { x: xArr, y: yArr };
              const xd = [], yd = [];
              for (let i = 0; i < yArr.length; i += eff) {
                yd.push(yArr[i]);
                xd.push(xArr ? xArr[i] : i);
              }
              return { x: xd, y: yd };
            }

            if (!(signalData && Array.isArray(signalData.matrix) && Array.isArray(signalData.matrix[0]))) {
              return [];
            }

            const total = signalData.matrix.length;
            const cols = signalData.matrix[0].length;
            const xFull = Array.from({length: cols}, (_, i) => i);

            const channelNames = signalData.channel_names || [];
            const hasChannelNames = channelNames.length > 0;

            let channelIndices = [];
            if (selectedChannels && Array.isArray(selectedChannels) && selectedChannels.length > 0) {
              channelIndices = selectedChannels.map(chName => {
                const idx = channelNames.indexOf(chName);
                return idx >= 0 ? idx : -1;
              }).filter(idx => idx >= 0);

              if (channelIndices.length === 0) {
                return [];
              }
            } else {
              const channelStart = (channelRange && channelRange.start) || 0;
              const channelCount = Math.min(CHANNELS_PER_PAGE, total - channelStart);
              channelIndices = Array.from({length: channelCount}, (_, i) => channelStart + i);
            }

            const graphsOriginal = [];
            const graphsTransformed = [];

            const fileName = signalData.file_name || 'Sin archivo';
            const sessionInfo = signalData.session || '';
            const classNameMatch = fileName.match(/^([^\\[]+)/);
            const className = classNameMatch ? classNameMatch[1] : fileName;
            const classColors = signalData.class_colors || {};
            const classColor = classColors[className] || '#3b82f6';

            for (let i = 0; i < channelIndices.length; i++) {
              const ch = channelIndices[i];
              const yRaw = signalData.matrix[ch];
              if (!Array.isArray(yRaw)) continue;

              const channelLabel = hasChannelNames && ch < channelNames.length
                ? channelNames[ch]
                : 'Ch ' + ch;

              const xy = USE_DOWNSAMPLING
                ? downsampling(xFull, yRaw, { factor: DS_FACTOR, maxPoints: MAX_POINTS })
                : { x: xFull, y: yRaw };

              const figOriginal = {
                data: [{
                  type: USE_WEBGL ? 'scattergl' : 'scatter',
                  mode: 'lines',
                  x: xy.x,
                  y: xy.y,
                  line: { width: 1, color: classColor },
                  hoverinfo: 'skip',
                  name: channelLabel
                }],
                layout: {
                  margin: { l: 50, r: 10, t: 24, b: 24 },
                  paper_bgcolor: 'rgba(0,0,0,0)',
                  plot_bgcolor: 'rgba(0,0,0,0)',
                  showlegend: false,
                  xaxis: { showgrid: false, zeroline: false, fixedrange: true, title: 'muestras' },
                  yaxis: {
                    showgrid: true,
                    gridcolor: 'rgba(128,128,128,0.25)',
                    zeroline: false,
                    fixedrange: true,
                    title: channelLabel,
                    titlefont: { size: 14, weight: 'bold' }
                  },
                  height: 320,
                  autosize: true,
                  annotations: [{
                    text: channelLabel,
                    xref: 'paper',
                    yref: 'paper',
                    x: 0.02,
                    y: 0.98,
                    xanchor: 'left',
                    yanchor: 'top',
                    showarrow: false,
                    font: {
                      size: 18,
                      color: classColor,
                      weight: 'bold'
                    },
                    bgcolor: 'rgba(0,0,0,0.7)',
                    borderpad: 6
                  }]
                }
              };

              graphsOriginal.push({
                props: {
                  id: `pg-multi-orig-${ch}`,
                  figure: figOriginal,
                  responsive: true,
                  className: 'plot-item',
                  style: { height: '320px', width: '100%', minHeight: 0, marginBottom: '12px' },
                  config: {
                    displaylogo: false,
                    responsive: true,
                    modeBarButtonsToRemove: [
                      'zoom','pan','select','lasso2d','zoomIn2d','zoomOut2d',
                      'autoScale2d','resetScale2d','toImage'
                    ]
                  }
                },
                type: 'Graph',
                namespace: 'dash_core_components'
              });

              let figTransformed;
              {
                const hasTransformedData = transformedData && Array.isArray(transformedData.matrix) && Array.isArray(transformedData.matrix[ch]);
                const yTransformed = hasTransformedData ? transformedData.matrix[ch] : Array(cols).fill(0);

                const xyTransformed = USE_DOWNSAMPLING && hasTransformedData
                  ? downsampling(xFull, yTransformed, { factor: DS_FACTOR, maxPoints: MAX_POINTS })
                  : { x: xFull, y: yTransformed };

                figTransformed = {
                  data: [{
                    type: USE_WEBGL ? 'scattergl' : 'scatter',
                    mode: 'lines',
                    x: xyTransformed.x,
                    y: xyTransformed.y,
                    line: { width: 1, color: hasTransformedData ? darkenHSL(classColor, 20) : '#555' },
                    hoverinfo: 'skip',
                    name: 'Transform ' + channelLabel
                  }],
                  layout: {
                    margin: { l: 50, r: 10, t: 24, b: 24 },
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    showlegend: false,
                    xaxis: { showgrid: false, zeroline: false, fixedrange: true, title: 'frecuencia/nivel' },
                    yaxis: {
                      showgrid: true,
                      gridcolor: 'rgba(128,128,128,0.25)',
                      zeroline: false,
                      fixedrange: true,
                      title: channelLabel,
                      titlefont: { size: 14, weight: 'bold' }
                    },
                    height: 320,
                    autosize: true,
                    annotations: hasTransformedData ? [
                      {
                        text: channelLabel,
                        xref: 'paper',
                        yref: 'paper',
                        x: 0.02,
                        y: 0.98,
                        xanchor: 'left',
                        yanchor: 'top',
                        showarrow: false,
                        font: {
                          size: 18,
                          color: darkenHSL(classColor, 20),
                          weight: 'bold'
                        },
                        bgcolor: 'rgba(0,0,0,0.7)',
                        borderpad: 6
                      }
                    ] : [
                      {
                        text: channelLabel,
                        xref: 'paper',
                        yref: 'paper',
                        x: 0.02,
                        y: 0.98,
                        xanchor: 'left',
                        yanchor: 'top',
                        showarrow: false,
                        font: {
                          size: 18,
                          color: '#888',
                          weight: 'bold'
                        },
                        bgcolor: 'rgba(0,0,0,0.7)',
                        borderpad: 6
                      },
                      {
                        text: 'Sin transformada aplicada',
                        xref: 'paper',
                        yref: 'paper',
                        x: 0.5,
                        y: 0.5,
                        showarrow: false,
                        font: { size: 12, color: 'rgba(255,255,255,0.3)' }
                      }
                    ]
                  }
                };
              }

              graphsTransformed.push({
                props: {
                  id: `pg-multi-trans-${ch}`,
                  figure: figTransformed,
                  responsive: true,
                  className: 'plot-item',
                  style: { height: '320px', width: '100%', minHeight: 0, marginBottom: '12px' },
                  config: {
                    displaylogo: false,
                    responsive: true,
                    modeBarButtonsToRemove: [
                      'zoom','pan','select','lasso2d','zoomIn2d','zoomOut2d',
                      'autoScale2d','resetScale2d','toImage'
                    ]
                  }
                },
                type: 'Graph',
                namespace: 'dash_core_components'
              });
            }

            function createStyledTitle(session, eventClass, type, color) {
              const parts = [];

              if (session) {
                parts.push({
                  props: {
                    children: 'Sesión: ',
                    style: {
                      fontSize: '11px',
                      fontWeight: '600',
                      color: 'rgba(255,255,255,0.5)',
                      marginRight: '4px'
                    }
                  },
                  type: 'Span',
                  namespace: 'dash_html_components'
                });
                parts.push({
                  props: {
                    children: session,
                    style: {
                      fontSize: '11px',
                      fontWeight: '500',
                      color: 'rgba(255,255,255,0.7)',
                      marginRight: '12px'
                    }
                  },
                  type: 'Span',
                  namespace: 'dash_html_components'
                });
              }

              parts.push({
                props: {
                  children: 'Clase: ',
                  style: {
                    fontSize: '11px',
                    fontWeight: '600',
                    color: 'rgba(255,255,255,0.5)',
                    marginRight: '4px'
                  }
                },
                type: 'Span',
                namespace: 'dash_html_components'
              });
              parts.push({
                props: {
                  children: eventClass,
                  style: {
                    fontSize: '13px',
                    fontWeight: '700',
                    color: color,
                    marginRight: '12px'
                  }
                },
                type: 'Span',
                namespace: 'dash_html_components'
              });

              parts.push({
                props: {
                  children: 'Tipo: ',
                  style: {
                    fontSize: '11px',
                    fontWeight: '600',
                    color: 'rgba(255,255,255,0.5)',
                    marginRight: '4px'
                  }
                },
                type: 'Span',
                namespace: 'dash_html_components'
              });
              parts.push({
                props: {
                  children: type,
                  style: {
                    fontSize: '11px',
                    fontWeight: '600',
                    color: 'rgba(255,255,255,0.8)',
                    textTransform: 'uppercase',
                    letterSpacing: '0.5px'
                  }
                },
                type: 'Span',
                namespace: 'dash_html_components'
              });

              return parts;
            }

            return {
              props: {
                children: [
                  {
                    props: {
                      children: [
                        {
                          props: {
                            children: createStyledTitle(sessionInfo, className, 'Original', classColor),
                            style: {
                              marginBottom: '12px',
                              paddingBottom: '8px',
                              borderBottom: '2px solid ' + classColor,
                              display: 'flex',
                              alignItems: 'center',
                              overflow: 'hidden'
                            }
                          },
                          type: 'Div',
                          namespace: 'dash_html_components'
                        },
                        ...graphsOriginal
                      ],
                      style: {
                        flex: 1,
                        paddingRight: '8px',
                        minWidth: 0
                      }
                    },
                    type: 'Div',
                    namespace: 'dash_html_components'
                  },
                  {
                    props: {
                      children: [
                        {
                          props: {
                            children: createStyledTitle(sessionInfo, className, 'Transformada', classColor),
                            style: {
                              marginBottom: '12px',
                              paddingBottom: '8px',
                              borderBottom: '2px solid ' + classColor,
                              display: 'flex',
                              alignItems: 'center',
                              overflow: 'hidden'
                            }
                          },
                          type: 'Div',
                          namespace: 'dash_html_components'
                        },
                        ...graphsTransformed
                      ],
                      style: {
                        flex: 1,
                        paddingLeft: '8px',
                        minWidth: 0
                      }
                    },
                    type: 'Div',
                    namespace: 'dash_html_components'
                  }
                ],
                style: {
                  display: 'flex',
                  gap: '16px',
                  width: '100%'
                }
              },
              type: 'Div',
              namespace: 'dash_html_components'
            };

          } catch (e) {
            console.error('[clientside:visualization] ERROR:', e);
            return window.dash_clientside.no_update;
          }
        }
        """,
        Output(PLOTS_CONTAINER, 'children'),
        [
            Input(DATA_STORE, 'data'),
            Input(TRANSFORMED_DATA_STORE, 'data'),
            Input(CHANNEL_RANGE_STORE, 'data'),
            Input(SELECTED_CHANNELS_STORE, 'data')
        ],
        prevent_initial_call=True
    )

    print(f"Callbacks registrados para TransformationViewer (prefix={prefix}, model_type={model_type})")


def create_navigation_controls(
    prefix: str,
    CHANNEL_NAV_INFO: str,
    BTN_PREV_CHANNELS: str,
    BTN_NEXT_CHANNELS: str,
    BTN_ALL_CLASSES: str,
    BTN_FILTER_CLASS: str,
    PIPELINE_HISTORY_VIEWER: str,
    TOGGLE_AUTO_APPLY: str,
    CHECKLIST_CHANNELS: str,
    BTN_SELECT_ALL: str,
    BTN_CLEAR: str,
    BTN_ONLY_EEG: str,
    CHANNEL_COUNT_DISPLAY: str
):
    """Crea los controles de navegación de canales y filtrado por clase"""

    return html.Div([
        # Navegación de canales
        html.Div([
            html.Div(
                id=CHANNEL_NAV_INFO,
                children="Canales 0 - 7 de 0",
                style={
                    "fontSize": "11px",
                    "fontWeight": "600",
                    "color": "var(--text)",
                    "marginBottom": "6px",
                    "textAlign": "center"
                }
            ),
            html.Div([
                html.Button(
                    '← Anteriores',
                    id=BTN_PREV_CHANNELS,
                    n_clicks=0,
                    disabled=True,
                    style={
                        "padding": "3px 8px",
                        "borderRadius": "var(--radius-sm)",
                        "border": "none",
                        "background": "var(--card-bg)",
                        "color": "var(--text)",
                        "cursor": "not-allowed",
                        "fontSize": "10px",
                        "fontWeight": "500",
                        "opacity": "0.5",
                        "flex": "1"
                    }
                ),
                html.Button(
                    'Siguientes →',
                    id=BTN_NEXT_CHANNELS,
                    n_clicks=0,
                    disabled=True,
                    style={
                        "padding": "3px 8px",
                        "borderRadius": "var(--radius-sm)",
                        "border": "none",
                        "background": "var(--card-bg)",
                        "color": "var(--text)",
                        "cursor": "not-allowed",
                        "fontSize": "10px",
                        "fontWeight": "500",
                        "opacity": "0.5",
                        "flex": "1"
                    }
                ),
            ], style={
                "display": "flex",
                "gap": "4px",
                "marginBottom": "12px"
            })
        ]),

        # Divisor
        html.Hr(style={
            "border": "none",
            "borderTop": "1px solid var(--border-weak)",
            "margin": "8px 0",
            "opacity": "0.4"
        }),

        # Filtro por clase (se llenará dinámicamente)
        html.Div(id=f"class-filter-container-{prefix}", children=[
            html.Div("Cargando clases...", style={"fontSize": "9px", "color": "#888"})
        ]),

        # Divisor
        html.Hr(style={
            "border": "none",
            "borderTop": "1px solid var(--border-weak)",
            "margin": "8px 0",
            "opacity": "0.4"
        }),

        # Historial del Pipeline
        html.Div(id=PIPELINE_HISTORY_VIEWER, children=[
            html.Div("Cargando historial...", style={
                "fontSize": "9px",
                "color": "var(--text-muted)",
                "textAlign": "center",
                "padding": "8px"
            })
        ]),

        # Divisor
        html.Hr(style={
            "border": "none",
            "borderTop": "1px solid var(--border-weak)",
            "margin": "8px 0",
            "opacity": "0.4"
        }),

        # Toggle Auto-aplicar Pipeline
        html.Div([
            html.Div([
                html.Span("Auto-aplicar pipeline", style={
                    "fontSize": "10px",
                    "fontWeight": "600",
                    "color": "#ddd",
                    "flex": "1"
                }),
                html.Button(
                    "ON",
                    id=TOGGLE_AUTO_APPLY,
                    n_clicks=0,
                    style={
                        "padding": "2px 8px",
                        "fontSize": "9px",
                        "borderRadius": "3px",
                        "border": "1px solid #4CAF50",
                        "background": "#4CAF50",
                        "color": "white",
                        "cursor": "pointer",
                        "fontWeight": "600"
                    }
                )
            ], style={
                "display": "flex",
                "alignItems": "center",
                "gap": "8px",
                "padding": "6px",
                "background": "#1a1a1a",
                "borderRadius": "4px",
                "border": "1px solid #333"
            })
        ], style={"marginBottom": "12px"}),

        # Divisor
        html.Hr(style={
            "border": "none",
            "borderTop": "1px solid var(--border-weak)",
            "margin": "8px 0",
            "opacity": "0.4"
        }),

        # Selector de canales específicos
        html.Div([
            # Header
            html.Div([
                html.Div("Canales específicos", style={
                    "fontSize": "10px",
                    "fontWeight": "600",
                    "color": "var(--text)",
                    "flex": "1"
                }),
                html.Div([
                    html.Button("Todos", id=BTN_SELECT_ALL, n_clicks=0, style={
                        "padding": "2px 6px",
                        "fontSize": "8px",
                        "borderRadius": "3px",
                        "border": "1px solid var(--border-weak)",
                        "background": "var(--card-bg)",
                        "color": "var(--text)",
                        "cursor": "pointer",
                        "marginRight": "4px"
                    }),
                    html.Button("Limpiar", id=BTN_CLEAR, n_clicks=0, style={
                        "padding": "2px 6px",
                        "fontSize": "8px",
                        "borderRadius": "3px",
                        "border": "1px solid var(--border-weak)",
                        "background": "var(--card-bg)",
                        "color": "var(--text)",
                        "cursor": "pointer",
                        "marginRight": "4px"
                    }),
                    html.Button("Solo EEG", id=BTN_ONLY_EEG, n_clicks=0, style={
                        "padding": "2px 6px",
                        "fontSize": "8px",
                        "borderRadius": "3px",
                        "border": "1px solid var(--border-weak)",
                        "background": "var(--card-bg)",
                        "color": "var(--text)",
                        "cursor": "pointer"
                    })
                ], style={"display": "flex"})
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "6px"}),

            # Checklist scrollable
            html.Div([
                dcc.Checklist(
                    id=CHECKLIST_CHANNELS,
                    options=[],
                    value=[],
                    labelStyle={
                        "display": "block",
                        "padding": "2px 4px",
                        "fontSize": "9px",
                        "cursor": "pointer"
                    },
                    inputStyle={
                        "marginRight": "6px",
                        "cursor": "pointer"
                    },
                    style={
                        "color": "var(--text)",
                        "lineHeight": "1.4"
                    }
                )
            ], style={
                "maxHeight": "150px",
                "overflowY": "auto",
                "overflowX": "hidden",
                "padding": "4px",
                "border": "1px solid var(--border-weak)",
                "borderRadius": "var(--radius-sm)",
                "background": "var(--card-bg)"
            }),

            # Contador
            html.Div(id=CHANNEL_COUNT_DISPLAY, children="0 canales seleccionados", style={
                "fontSize": "8px",
                "color": "var(--text-muted)",
                "marginTop": "4px",
                "textAlign": "right"
            })
        ])
    ])
