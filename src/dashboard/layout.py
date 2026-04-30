"""
Dashboard layout: four tabs covering EDA, model performance, live demo,
and AI methodology documentation required by the project rubric.

All figures are built with plotly.graph_objects only — no plotly.express,
no polars, no pandas — to avoid native thread-pool deadlocks on Windows
with PyTorch CUDA installed.
"""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html

from .data_loader import (
    load_eda_data,
    load_model_results,
    GAME_COLORS,
    MODEL_COLORS,
    EXP_LEVELS,
    EXP_COLORS,
)

# ── Lazy plotly import ────────────────────────────────────────────────────────

go = None  # populated on first call to _ensure_plotting()


def _ensure_plotting() -> None:
    """Import plotly.graph_objects on first use (it's already fast, but kept lazy)."""
    import sys
    mod = sys.modules[__name__]
    if mod.go is None:
        import plotly.graph_objects as _go
        mod.go = _go


# ── Shared chart style ────────────────────────────────────────────────────────

PLOT_CONFIG = {"displayModeBar": False}

PLOT_LAYOUT: dict = dict(
    paper_bgcolor="#2c3034",
    plot_bgcolor="#2c3034",
    font=dict(color="#dee2e6", size=12),
    margin=dict(l=50, r=20, t=50, b=50),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)

# ── EDA figure builders ───────────────────────────────────────────────────────


def records_per_game_fig(data: dict) -> "go.Figure":
    _ensure_plotting()
    games  = data["games"]
    counts = [data["records_per_game"].get(g, 0) for g in games]
    colors = [GAME_COLORS.get(g, "#adb5bd") for g in games]
    fig = go.Figure(go.Bar(
        x=games, y=counts, marker_color=colors,
        text=counts, textposition="outside",
    ))
    fig.add_hline(y=2000, line_dash="dash", line_color="#adb5bd",
                  annotation_text="Target (2,000)", annotation_position="top right")
    fig.update_layout(
        **PLOT_LAYOUT, showlegend=False,
        title="Records Collected per Game",
        xaxis_title="Game",
        yaxis=dict(title="Record Count", gridcolor="#373b3e"),
    )
    return fig


def source_type_fig(data: dict) -> "go.Figure":
    _ensure_plotting()
    games = data["games"]
    stb   = data["source_type_breakdown"]
    SOURCE_COLORS = {"google_images": "#2196F3", "youtube": "#F44336"}
    SOURCE_LABELS = {"google_images": "Google Images", "youtube": "YouTube"}
    fig = go.Figure()
    for src, game_counts in stb.items():
        fig.add_trace(go.Bar(
            name=SOURCE_LABELS.get(src, src),
            x=games,
            y=[game_counts.get(g, 0) for g in games],
            marker_color=SOURCE_COLORS.get(src, "#adb5bd"),
        ))
    fig.update_layout(
        **PLOT_LAYOUT, barmode="group",
        title="Source Type Breakdown per Game",
        xaxis_title="Game",
        yaxis=dict(title="Records", gridcolor="#373b3e"),
        legend_title_text="Source",
    )
    return fig


def experience_dist_fig(data: dict) -> "go.Figure":
    _ensure_plotting()
    games    = data["games"]
    exp_dist = data["experience_dist"]
    fig = go.Figure()
    for level in EXP_LEVELS:
        if level not in exp_dist:
            continue
        game_counts = exp_dist[level]
        fig.add_trace(go.Bar(
            name=level,
            x=games,
            y=[game_counts.get(g, 0) for g in games],
            marker_color=EXP_COLORS.get(level, "#adb5bd"),
        ))
    fig.update_layout(
        **PLOT_LAYOUT, barmode="stack",
        title="Experience Level Distribution per Game",
        xaxis_title="Game",
        yaxis=dict(title="Records", gridcolor="#373b3e"),
        legend_title_text="Experience",
    )
    return fig


def null_heatmap_fig(data: dict) -> "go.Figure":
    _ensure_plotting()
    games  = data["games"]
    fields = data["null_fields"]
    npct   = data["null_pct"]
    z, text = [], []
    for field in fields:
        row_pct = [npct.get(field, {}).get(g, 0) for g in games]
        z.append(row_pct)
        text.append([f"{v:.1f}%" for v in row_pct])
    fig = go.Figure(go.Heatmap(
        z=z, x=games, y=fields,
        colorscale="RdYlGn_r", zmin=0, zmax=100,
        text=text, texttemplate="%{text}",
        colorbar=dict(title="NULL %", ticksuffix="%"),
    ))
    fig.update_layout(
        **PLOT_LAYOUT,
        title="NULL Density by Field and Game (%)",
        xaxis_title="Game", yaxis_title="Field",
        height=360,
    )
    return fig


def narration_box_fig(data: dict) -> "go.Figure":
    _ensure_plotting()
    games = data["games"]
    narr  = data["narration_lengths"]
    fig = go.Figure()
    for game in games:
        lengths = narr.get(game, [])
        if lengths:
            fig.add_trace(go.Box(
                y=lengths, name=game,
                marker_color=GAME_COLORS.get(game, "#adb5bd"),
                boxpoints=False,
            ))
    fig.update_layout(
        **PLOT_LAYOUT, showlegend=False,
        title="Gameplay Narration Length",
        yaxis=dict(title="Characters", gridcolor="#373b3e"),
    )
    return fig


# ── Model performance figure builders ─────────────────────────────────────────

_MODEL_DISPLAY = {
    "cnn": "CNN", "nn": "NN",
    "transformer": "Transformer", "ensemble": "Ensemble",
}


def accuracy_bar_fig(results: dict) -> "go.Figure":
    _ensure_plotting()
    names, accs, colors = [], [], []
    for key in ("cnn", "nn", "transformer", "ensemble"):
        if key in results:
            label = _MODEL_DISPLAY[key]
            names.append(label)
            accs.append(results[key]["accuracy"] * 100)
            colors.append(MODEL_COLORS[label])
    fig = go.Figure(go.Bar(
        x=names, y=accs, marker_color=colors,
        text=[f"{a:.1f}%" for a in accs], textposition="outside",
        width=0.5,
    ))
    fig.update_layout(
        **PLOT_LAYOUT,
        title="Test-Set Accuracy by Model",
        yaxis=dict(title="Accuracy (%)", range=[0, 115], gridcolor="#373b3e"),
        xaxis_title="Model",
    )
    return fig


def confusion_matrix_fig(results: dict, model_key: str) -> "go.Figure":
    _ensure_plotting()
    if model_key not in results:
        return go.Figure().update_layout(**PLOT_LAYOUT, title="No data")
    data    = results[model_key]
    cm      = data["cm"]
    lnames  = data["label_names"]
    row_sum = cm.sum(axis=1, keepdims=True).clip(1)
    cm_norm = cm.astype(float) / row_sum
    label   = _MODEL_DISPLAY[model_key]
    fig = go.Figure(go.Heatmap(
        z=cm_norm.tolist(),
        x=lnames, y=lnames,
        colorscale="Blues",
        zmin=0, zmax=1,
        text=[[str(v) for v in row] for row in cm.tolist()],
        texttemplate="%{text}",
        colorbar=dict(title="Row %"),
    ))
    fig.update_layout(
        **PLOT_LAYOUT,
        title=f"{label} — Confusion Matrix  (counts, row-normalized color)",
        xaxis_title="Predicted", yaxis_title="True",
        height=420,
    )
    return fig


def perclass_fig(results: dict, model_key: str) -> "go.Figure":
    _ensure_plotting()
    if model_key not in results:
        return go.Figure().update_layout(**PLOT_LAYOUT, title="No data")
    pc     = results[model_key]["per_class"]
    games  = list(pc.keys())
    accs   = [pc[g] * 100 for g in games]
    colors = [GAME_COLORS.get(g, "#adb5bd") for g in games]
    label  = _MODEL_DISPLAY[model_key]
    fig = go.Figure(go.Bar(
        x=games, y=accs, marker_color=colors,
        text=[f"{a:.1f}%" for a in accs], textposition="outside",
    ))
    fig.update_layout(
        **PLOT_LAYOUT,
        title=f"{label} — Per-Class Accuracy",
        yaxis=dict(title="Accuracy (%)", range=[0, 115], gridcolor="#373b3e"),
        xaxis_title="Game",
    )
    return fig


# ── Tab builders ──────────────────────────────────────────────────────────────


def build_eda_tab() -> dbc.Container:
    _ensure_plotting()
    data   = load_eda_data()
    total  = data["total"]
    n_gg   = data["n_gg"]
    n_yt   = data["n_yt"]
    exp_ok = data["exp_ok"]

    return dbc.Container([
        dbc.Row([
            dbc.Col(_stat_card(f"{total:,}",  "Total Records",        "#2196F3"), width=3),
            dbc.Col(_stat_card("5",            "Games Tracked",        "#9C27B0"), width=3),
            dbc.Col(_stat_card(f"{n_gg:,}",   "Google Image Records", "#F44336"), width=3),
            dbc.Col(_stat_card(f"{n_yt:,}",   "YouTube Records",      "#FF9800"), width=3),
        ], className="mb-4 mt-3 g-3"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=records_per_game_fig(data), config=PLOT_CONFIG), width=6),
            dbc.Col(dcc.Graph(figure=source_type_fig(data),      config=PLOT_CONFIG), width=6),
        ], className="mb-3"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=experience_dist_fig(data), config=PLOT_CONFIG), width=12),
        ], className="mb-3"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=null_heatmap_fig(data),  config=PLOT_CONFIG), width=7),
            dbc.Col(dcc.Graph(figure=narration_box_fig(data), config=PLOT_CONFIG), width=5),
        ], className="mb-3"),
    ], fluid=True)


def build_performance_tab() -> dbc.Container:
    _ensure_plotting()
    results = load_model_results()
    options = [
        {"label": "CNN  (EfficientNet-B0 image classifier)",       "value": "cnn"},
        {"label": "NN  (Fully-connected on narration + metadata)",  "value": "nn"},
        {"label": "Transformer  (MiniLM-L6 text embeddings)",       "value": "transformer"},
        {"label": "Ensemble  (averaged softmax probabilities)",      "value": "ensemble"},
    ]
    available = [o for o in options if o["value"] in results]
    default   = available[0]["value"] if available else None

    return dbc.Container([
        dbc.Row([
            dbc.Col(dcc.Graph(figure=accuracy_bar_fig(results), config=PLOT_CONFIG),
                    width=12),
        ], className="mb-3 mt-3"),
        dbc.Row([
            dbc.Col([
                html.P("Select a model to inspect its confusion matrix and per-class accuracy:",
                       className="text-muted mb-2"),
                dbc.RadioItems(
                    id="model-select",
                    options=available,
                    value=default,
                    inline=True,
                    className="mb-3",
                ),
            ], width=12),
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id="confusion-matrix-fig", config=PLOT_CONFIG), width=7),
            dbc.Col(dcc.Graph(id="perclass-fig",         config=PLOT_CONFIG), width=5),
        ]),
    ], fluid=True)


def build_live_demo_tab() -> dbc.Container:
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H5("Upload a gameplay screenshot", className="mt-3 mb-1"),
                html.P(
                    "The EfficientNet-B0 CNN will predict which of the five games "
                    "the screenshot is from and show its confidence for each class.",
                    className="text-muted small",
                ),
                dcc.Upload(
                    id="upload-image",
                    children=html.Div([
                        "Drag & Drop or ", html.A("click to browse"),
                        html.Br(),
                        html.Span("PNG, JPG, WEBP accepted", className="small text-muted"),
                    ]),
                    style={
                        "width": "100%", "height": "130px",
                        "lineHeight": "60px", "borderWidth": "2px",
                        "borderStyle": "dashed", "borderRadius": "8px",
                        "borderColor": "#6c757d", "textAlign": "center",
                        "cursor": "pointer", "paddingTop": "10px",
                    },
                    multiple=False,
                    accept="image/*",
                ),
                html.Div(id="upload-status", className="mt-2 text-muted small"),
            ], width=4),
            dbc.Col(html.Div(id="preview-img-container"), width=3),
            dbc.Col(dcc.Graph(id="prediction-bar", config=PLOT_CONFIG,
                              style={"height": "300px"}), width=5),
        ], className="mt-3"),
    ], fluid=True)


_CARD_1_BODY = (
    "An agentic pipeline searched Google/Bing Images and YouTube for gameplay "
    "screenshots across five games. The primary collection run used gemma4:26b "
    "(Gemma 4, 26B parameters, served locally via Ollama) for three tasks:\n"
    "(a) image validation -- rejecting ads, menus, and off-topic content;\n"
    "(b) narration generation -- a text description of the game state written "
    "independently of the image so it functions as a text feature for the Transformer;\n"
    "(c) experience-level classification -- rating visible player skill as "
    "Poor / Fair / Good / Excellent / Superior.\n\n"
    "A supplementary collection pass used ministral3:14b (Mistral 14B, fits fully "
    "in 12 GB VRAM) to scale up record volume on additional hardware, with the same "
    "prompt templates and validation logic."
)

_CARD_2_BODY = (
    "A second offline enrichment pass used ministral3:14b to fill fields left empty "
    "during collection: player_name, gameplay_timestamp, channel_description, "
    "player_experience_narration, and identifying_quotes. Records were loaded "
    "in batches of five and each field updated only when the model returned a "
    "non-sentinel value, preserving any data already collected.\n\n"
    "Earlier enrichment runs also used gemma4:26b; the final dataset reflects "
    "contributions from both models across multiple enrichment passes."
)

_CARD_3_BODY = (
    "Correctness was validated at multiple checkpoints:\n"
    "- Narration independence check: a regex flags any narration referencing\n"
    "  the image directly ('screenshot', 'visible', 'depicted') and retries.\n"
    "- Fallback pattern detection: records containing the phrase\n"
    "  'working through game mechanics and pursuing progression goals'\n"
    "  are re-queued for regeneration in the next enrichment pass.\n"
    "- NULL density audit logged after each enrichment run (target < 10%).\n"
    "- Image integrity scan: every path in the DB verified on disk;\n"
    "  orphaned records deleted.\n"
    "- pHash deduplication (Hamming <= 5) and Laplacian blur filtering\n"
    "  (variance < 100) remove unusable or duplicate images before training."
)

_CARD_4_BODY = (
    "Three independent classifiers trained on the same 70/15/15 stratified split:\n"
    "- CNN: EfficientNet-B0 fine-tuned on 224x224 screenshots\n"
    "  (Phase 1: frozen backbone warm-up; Phase 2: full fine-tune + early stopping).\n"
    "- NN: Fully-connected network on numerical gameplay features joined from\n"
    "  gameplay_records (experience level, source type) and Steam API aggregate\n"
    "  statistics (achievement count, mean completion rate, player count).\n"
    "- Transformer: SentenceTransformer (all-MiniLM-L6-v2) embeds sanitized\n"
    "  narration text into 384-dim vectors fed to the same FC classifier architecture.\n"
    "Final predictions combine all three via a learned weighting strategy that\n"
    "minimises cross-entropy on the validation set (L-BFGS-B optimisation)."
)


def build_methodology_tab() -> dbc.Container:
    _ensure_plotting()
    data        = load_eda_data()
    sample_rows = data["sample_rows"]
    headers     = ["Game", "Source", "Experience", "AI-Generated Narration"]
    table = dbc.Table(
        [
            html.Thead(html.Tr([html.Th(h) for h in headers])),
            html.Tbody([
                html.Tr([html.Td(str(cell or "")) for cell in row])
                for row in sample_rows
            ]),
        ],
        striped=True, bordered=True, hover=True,
        className="small",
        style={"fontSize": "0.82rem"},
    )

    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H4("How AI Was Used in This Project", className="mt-3 mb-3"),
                _method_card("1  Data Collection - ministral-3:14b via Ollama", _CARD_1_BODY),
                _method_card("2  Enrichment Pass - gemma4:26b",                 _CARD_2_BODY),
                _method_card("3  Validating AI Correctness",                    _CARD_3_BODY),
                _method_card("4  Model Architectures",                          _CARD_4_BODY),
                html.H5("Sample AI-Generated Narrations from the Dataset",
                        className="mt-4 mb-3"),
                table,
            ], width=12),
        ]),
    ], fluid=True)


# ── Shared component helpers ──────────────────────────────────────────────────


def _stat_card(value: str, label: str, color: str) -> dbc.Card:
    return dbc.Card(
        dbc.CardBody([
            html.H3(value, style={"color": color, "fontWeight": "700", "marginBottom": "4px"}),
            html.P(label, className="text-muted mb-0 small"),
        ]),
        className="text-center h-100",
    )


def _method_card(title: str, body: str) -> dbc.Card:
    return dbc.Card([
        dbc.CardHeader(html.Strong(title)),
        dbc.CardBody(
            html.Pre(body, style={
                "whiteSpace": "pre-wrap", "fontFamily": "inherit",
                "fontSize": "0.85rem", "marginBottom": 0,
            })
        ),
    ], className="mb-3")


# ── Root layout ───────────────────────────────────────────────────────────────


def build_tabs() -> dbc.Tabs:
    """Build all four tab panels. Called lazily via callback after server starts."""
    return dbc.Tabs(
        [
            dbc.Tab(build_eda_tab(),         label="Data Overview",     tab_id="eda"),
            dbc.Tab(build_performance_tab(), label="Model Performance", tab_id="perf"),
            dbc.Tab(build_live_demo_tab(),   label="Live Demo",         tab_id="demo"),
            dbc.Tab(build_methodology_tab(), label="AI Methodology",    tab_id="method"),
        ],
        id="tabs",
        active_tab="eda",
        className="mt-2 mx-3",
    )


def build_shell() -> html.Div:
    """
    Minimal static layout — no DB access, returns instantly.
    The real content is injected by the serve_main_content callback.
    """
    return html.Div([
        dcc.Location(id="_url", refresh=False),
        dbc.NavbarSimple(
            brand="Game Predictor Dashboard",
            color="dark",
            children=[
                html.Span("CIS 2450  ·  Spring 2026",
                          className="navbar-text text-muted small"),
            ],
        ),
        dbc.Container(
            dcc.Loading(
                html.Div(id="_main-content", style={"minHeight": "300px"}),
                type="circle",
                color="#2196F3",
            ),
            fluid=True,
            className="mt-3",
        ),
    ])
