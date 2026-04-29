"""
Dashboard layout: four tabs covering EDA, model performance, live demo,
and AI methodology documentation required by the project rubric.

Figure-building functions are defined here so callbacks.py can import them
without any circular dependencies.
"""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html

from .data_loader import (
    load_gameplay_df,
    load_model_results,
    GAME_COLORS,
    MODEL_COLORS,
    EXP_LEVELS,
    EXP_COLORS,
)

# ── Lazy plotting imports (pandas + plotly are slow to import on Windows) ─────

go = px = pl = None  # populated on first call to _ensure_plotting()


def _ensure_plotting() -> None:
    """Import plotting libraries on first use and inject them as module globals."""
    import sys
    mod = sys.modules[__name__]
    if mod.go is None:
        import plotly.graph_objects as _go
        import plotly.express as _px
        import polars as _pl
        mod.go = _go
        mod.px = _px
        mod.pl = _pl


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


def records_per_game_fig(df) -> "go.Figure":
    _ensure_plotting()
    counts = df["video_game_name"].value_counts(sort=True)
    fig = px.bar(
        counts, x="video_game_name", y="count",
        color="video_game_name", color_discrete_map=GAME_COLORS,
        title="Records Collected per Game",
        labels={"count": "Record Count", "video_game_name": "Game"},
        text="count",
    )
    fig.update_traces(textposition="outside")
    fig.add_hline(y=2000, line_dash="dash", line_color="#adb5bd",
                  annotation_text="Target (2,000)", annotation_position="top right")
    fig.update_layout(**PLOT_LAYOUT, showlegend=False,
                      yaxis=dict(title="Record Count", gridcolor="#373b3e"))
    return fig


def source_type_fig(df) -> "go.Figure":
    _ensure_plotting()
    counts = (
        df.group_by(["video_game_name", "source_type"])
        .len()
        .rename({"len": "Count"})
    )
    fig = px.bar(
        counts, x="video_game_name", y="Count",
        color="source_type", barmode="group",
        title="Source Type Breakdown per Game",
        color_discrete_map={"google_images": "#2196F3", "youtube": "#F44336"},
        labels={"video_game_name": "Game", "source_type": "Source", "Count": "Records"},
    )
    fig.update_layout(**PLOT_LAYOUT, yaxis=dict(gridcolor="#373b3e"),
                      legend_title_text="Source")
    return fig


def experience_dist_fig(df) -> "go.Figure":
    _ensure_plotting()
    counts = (
        df.filter(pl.col("experience_level").is_not_null())
        .group_by(["video_game_name", "experience_level"])
        .len()
        .rename({"len": "Count"})
    )
    fig = px.bar(
        counts, x="video_game_name", y="Count",
        color="experience_level", barmode="stack",
        title="Experience Level Distribution per Game",
        color_discrete_map=EXP_COLORS,
        category_orders={"experience_level": EXP_LEVELS},
        labels={"video_game_name": "Game", "experience_level": "Level", "Count": "Records"},
    )
    fig.update_layout(**PLOT_LAYOUT, yaxis=dict(gridcolor="#373b3e"),
                      legend_title_text="Experience")
    return fig


def null_heatmap_fig(df) -> "go.Figure":
    _ensure_plotting()
    fields = [
        "player_name", "gameplay_timestamp", "experience_level",
        "gameplay_level", "total_playtime",
        "channel_description", "player_experience_narration",
    ]
    games = sorted(df["video_game_name"].unique().to_list())
    z, text = [], []
    for field in fields:
        row_z, row_t = [], []
        for game in games:
            sub = df.filter(pl.col("video_game_name") == game)[field]
            pct = sub.is_null().mean() * 100
            row_z.append(pct)
            row_t.append(f"{pct:.1f}%")
        z.append(row_z)
        text.append(row_t)

    fig = go.Figure(go.Heatmap(
        z=z, x=games, y=fields,
        colorscale="RdYlGn_r",
        zmin=0, zmax=100,
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


def narration_box_fig(df) -> "go.Figure":
    _ensure_plotting()
    fig = px.box(
        df.filter(pl.col("narration_len") > 0),
        x="video_game_name", y="narration_len",
        color="video_game_name", color_discrete_map=GAME_COLORS,
        title="Gameplay Narration Length",
        labels={"video_game_name": "Game", "narration_len": "Characters"},
        points=False,
    )
    fig.update_layout(**PLOT_LAYOUT, showlegend=False,
                      yaxis=dict(gridcolor="#373b3e"))
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
    # Row-normalize for color scale; show raw counts as annotation text
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
    df    = load_gameplay_df()
    total = len(df)
    n_gg  = df.filter(pl.col("source_type") == "google_images").height
    n_yt  = df.filter(pl.col("source_type") == "youtube").height
    exp_ok = int(df["experience_level"].is_not_null().sum())

    return dbc.Container([
        # ── Summary cards ──────────────────────────────────────────────────
        dbc.Row([
            dbc.Col(_stat_card(f"{total:,}",  "Total Records",         "#2196F3"), width=3),
            dbc.Col(_stat_card("5",            "Games Tracked",         "#9C27B0"), width=3),
            dbc.Col(_stat_card(f"{n_gg:,}",   "Google Image Records",  "#F44336"), width=3),
            dbc.Col(_stat_card(f"{n_yt:,}",   "YouTube Records",       "#FF9800"), width=3),
        ], className="mb-4 mt-3 g-3"),

        # ── Row 1: records per game + source breakdown ──────────────────────
        dbc.Row([
            dbc.Col(dcc.Graph(figure=records_per_game_fig(df), config=PLOT_CONFIG), width=6),
            dbc.Col(dcc.Graph(figure=source_type_fig(df),      config=PLOT_CONFIG), width=6),
        ], className="mb-3"),

        # ── Row 2: experience level distribution ───────────────────────────
        dbc.Row([
            dbc.Col(dcc.Graph(figure=experience_dist_fig(df), config=PLOT_CONFIG), width=12),
        ], className="mb-3"),

        # ── Row 3: NULL heatmap + narration length ─────────────────────────
        dbc.Row([
            dbc.Col(dcc.Graph(figure=null_heatmap_fig(df),   config=PLOT_CONFIG), width=7),
            dbc.Col(dcc.Graph(figure=narration_box_fig(df),  config=PLOT_CONFIG), width=5),
        ], className="mb-3"),
    ], fluid=True)


def build_performance_tab() -> dbc.Container:
    results = load_model_results()
    options = [
        {"label": "CNN  (EfficientNet-B0 image classifier)",  "value": "cnn"},
        {"label": "NN  (Fully-connected on narration + metadata)", "value": "nn"},
        {"label": "Transformer  (MiniLM-L6 text embeddings)",  "value": "transformer"},
        {"label": "Ensemble  (averaged softmax probabilities)", "value": "ensemble"},
    ]
    available = [o for o in options if o["value"] in results]
    default   = available[0]["value"] if available else None

    return dbc.Container([
        # ── Overall accuracy comparison ────────────────────────────────────
        dbc.Row([
            dbc.Col(dcc.Graph(figure=accuracy_bar_fig(results), config=PLOT_CONFIG),
                    width=12),
        ], className="mb-3 mt-3"),

        # ── Model selector ─────────────────────────────────────────────────
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

        # ── Confusion matrix + per-class accuracy ──────────────────────────
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
    "screenshots across five games. Every candidate image was processed by "
    "ministral-3:14b for three tasks: "
    "(a) image validation -- rejecting ads, menus, and off-topic content; "
    "(b) narration generation -- a text description of the game state written "
    "independently of the image so it functions as a text feature for the NN/Transformer; "
    "(c) experience-level classification -- rating visible player skill as "
    "Poor / Fair / Good / Excellent / Superior."
)

_CARD_2_BODY = (
    "A second offline pass used gemma4:26b to fill fields left empty during "
    "collection: player_name, gameplay_timestamp, channel_description, "
    "player_experience_narration, and identifying_quotes. Records were loaded "
    "in batches of five and each field updated only when the model returned a "
    "non-sentinel value, preserving any data already collected."
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
    "- NN: Fully-connected network on TF-IDF narration features (top-100 terms,\n"
    "  fitted on training split only) plus six metadata signals.\n"
    "- Transformer: SentenceTransformer (all-MiniLM-L6-v2) embeds narration text\n"
    "  into 384-dim vectors fed to the same FC classifier architecture.\n"
    "Final predictions combine all three via averaged softmax probabilities (ensemble)."
)


def build_methodology_tab() -> dbc.Container:
    _ensure_plotting()
    df = load_gameplay_df()

    # Sample two records per game to illustrate AI-generated content
    df_narr = df.filter(pl.col("gameplay_narration").is_not_null())
    samples = []
    for game in sorted(df_narr["video_game_name"].unique().to_list()):
        group = df_narr.filter(pl.col("video_game_name") == game)
        samples.append(group.sample(n=min(2, group.height), seed=42))
    sample_df = pl.concat(samples).select(
        ["video_game_name", "source_type", "experience_level", "gameplay_narration"]
    ).with_columns(
        (
            pl.col("gameplay_narration").str.slice(0, 220)
            + pl.when(pl.col("gameplay_narration").str.len_chars() > 220)
            .then(pl.lit("..."))
            .otherwise(pl.lit(""))
        ).alias("gameplay_narration")
    ).rename({
        "video_game_name": "Game",
        "source_type": "Source",
        "experience_level": "Experience",
        "gameplay_narration": "AI-Generated Narration",
    })

    col_names = sample_df.columns
    rows_data = sample_df.to_dicts()
    table = dbc.Table(
        [
            html.Thead(html.Tr([html.Th(c) for c in col_names])),
            html.Tbody([
                html.Tr([html.Td(str(r.get(c, ""))) for c in col_names])
                for r in rows_data
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
