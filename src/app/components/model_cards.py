# app/components/model_cards.py
from dash import html, dcc

# IDs comunes
BTN_BACK_ID = "btn-back-from-card"

def _card_shell(title: str, body_children):
    return html.Div(
        [
            html.Div(
                [html.H2(title), html.Button("← Volver", id=BTN_BACK_ID, className="btn")],
                className="card-header"
            ),
            html.Div(body_children, className="card-body"),
        ],
        className="train-card"
    )

def card_lstm(schema: dict):
    # TODO: aquí pones el formulario específico de LSTM
    return _card_shell("Configurar LSTM", [
        html.P("Formulario LSTM aquí…"),
        dcc.Markdown("**Schema preview:**"),
        dcc.Markdown(f"```json\n{schema}\n```", style={"whiteSpace":"pre"})
    ])

def card_gru(schema: dict):
    return _card_shell("Configurar GRU", [
        html.P("Formulario GRU aquí…"),
        dcc.Markdown("**Schema preview:**"),
        dcc.Markdown(f"```json\n{schema}\n```", style={"whiteSpace":"pre"})
    ])

def card_svnn(schema: dict):
    return _card_shell("Configurar SVNN (MLP)", [
        html.P("Formulario SVNN aquí…"),
        dcc.Markdown("**Schema preview:**"),
        dcc.Markdown(f"```json\n{schema}\n```", style={"whiteSpace":"pre"})
    ])

def card_svm(schema: dict):
    return _card_shell("Configurar SVM", [
        html.P("Formulario SVM aquí…"),
        dcc.Markdown("**Schema preview:**"),
        dcc.Markdown(f"```json\n{schema}\n```", style={"whiteSpace":"pre"})
    ])

def card_randomforest(schema: dict):
    return _card_shell("Configurar RandomForest", [
        html.P("Formulario RandomForest aquí…"),
        dcc.Markdown("**Schema preview:**"),
        dcc.Markdown(f"```json\n{schema}\n```", style={"whiteSpace":"pre"})
    ])

def card_cnn(schema: dict):
    return _card_shell("Configurar CNN", [
        html.P("Formulario CNN aquí…"),
        dcc.Markdown("**Schema preview:**"),
        dcc.Markdown(f"```json\n{schema}\n```", style={"whiteSpace":"pre"})
    ])

# Registry para resolver por nombre de modelo
CARD_REGISTRY = {
    "LSTM": card_lstm,
    "GRU": card_gru,
    "SVNN": card_svnn,
    "SVM": card_svm,
    "RandomForest": card_randomforest,
    "CNN": card_cnn,
}
