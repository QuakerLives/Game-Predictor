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
    """Import plotly.graph_objects on first use.

    Kept lazy so the module can be imported by the app entry point without
    triggering plotly's slow startup — only the tab that actually needs charts
    pays the import cost.
    """
    import sys
    mod = sys.modules[__name__]
    if mod.go is None:
        import plotly.graph_objects as _go
        mod.go = _go


# ── Shared chart style ────────────────────────────────────────────────────────

# Hide the plotly toolbar — it clutters the UI and we don't need export controls
PLOT_CONFIG = {"displayModeBar": False}

# Dark theme colors to match the DARKLY bootstrap theme used in app.py
PLOT_LAYOUT: dict = dict(
    paper_bgcolor="#2c3034",
    plot_bgcolor="#2c3034",
    font=dict(color="#dee2e6", size=12),
    margin=dict(l=50, r=20, t=50, b=50),
    legend=dict(bgcolor="rgba(0,0,0,0)"),  # transparent legend background
)


def _empty_pred_fig(msg: str = "") -> "go.Figure":
    """Styled placeholder that matches the post-prediction bar chart exactly.

    Pre-populates all five game rows at 0% so the container height is fixed
    before any inference runs. Without this, the chart section collapses to
    nothing on page load and jumps in size when the first prediction comes back.
    """
    _ensure_plotting()
    games  = list(GAME_COLORS.keys())
    colors = [GAME_COLORS[g] for g in games]
    # Horizontal bar at 0 for each game — just a placeholder, no real data yet
    fig = go.Figure(go.Bar(
        x=[0] * len(games), y=games,
        orientation="h",
        marker_color=colors,
        text=[""] * len(games),
        textposition="outside",
    ))
    fig.update_layout(**PLOT_LAYOUT)
    fig.update_layout(
        title=msg or "",
        # Range goes to 118 so the outside text labels have room and don't clip
        xaxis=dict(title="Confidence (%)", range=[0, 118], gridcolor="#373b3e"),
        height=280,
        margin=dict(l=110, r=30, t=60, b=40),
    )
    return fig


# ── EDA figure builders ───────────────────────────────────────────────────────
# Each function takes the pre-aggregated `data` dict from load_eda_data()
# and returns a standalone Plotly figure. Keeping them as separate functions
# makes it easy to swap or reorder charts in build_eda_tab.


def records_per_game_fig(data: dict) -> "go.Figure":
    """Bar chart: how many records were collected per game."""
    _ensure_plotting()
    games  = data["games"]
    counts = [data["records_per_game"].get(g, 0) for g in games]
    colors = [GAME_COLORS.get(g, "#adb5bd") for g in games]
    fig = go.Figure(go.Bar(
        x=games, y=counts, marker_color=colors,
        text=counts, textposition="outside",
    ))
    # Reference line so we can see how far each game is from the 2,000 target
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
    """Grouped bar chart: YouTube vs. Google Images record counts per game.

    Source type is one of the NN's input features, so knowing the breakdown
    matters — a model that only saw YouTube records wouldn't generalize well
    to Google Images inputs.
    """
    _ensure_plotting()
    games = data["games"]
    stb   = data["source_type_breakdown"]  # {source: {game: count}}
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
    """Stacked bar: distribution of AI-rated player experience levels per game.

    Stacked rather than grouped to show total records while still showing
    how the level mix varies by game.
    """
    _ensure_plotting()
    games    = data["games"]
    exp_dist = data["experience_dist"]  # {level: {game: count}}
    fig = go.Figure()
    # Iterate in the defined order (Poor → Superior) so the legend is sorted
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
    """Heatmap: what percentage of records have NULL for each metadata field.

    Red = high NULL rate (field is sparse), green = low (field is well-populated.
    This informed which fields we could use as raw features vs. binary flags.
    """
    _ensure_plotting()
    games  = data["games"]
    fields = data["null_fields"]
    npct   = data["null_pct"]  # {field: {game: pct}}
    z, text = [], []
    # Build the z matrix row-by-row (one row per field)
    for field in fields:
        row_pct = [npct.get(field, {}).get(g, 0) for g in games]
        z.append(row_pct)
        text.append([f"{v:.1f}%" for v in row_pct])
    fig = go.Figure(go.Heatmap(
        z=z, x=games, y=fields,
        colorscale="RdYlGn_r",  # red = bad (high NULL), green = good (low NULL)
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


def narration_box_fig(data: dict) -> "go.Figure":
    """Box plot: distribution of narration character lengths per game.

    We expected that more complex games (Stellaris, Skyrim) would produce
    longer narrations than simpler ones (Stardew Valley) — this chart checks that.
    Narration length is also one of the NN's numerical features.
    """
    _ensure_plotting()
    games = data["games"]
    narr  = data["narration_lengths"]  # {game: [int, ...]}
    fig = go.Figure()
    for game in games:
        lengths = narr.get(game, [])
        if lengths:
            fig.add_trace(go.Box(
                y=lengths, name=game,
                marker_color=GAME_COLORS.get(game, "#adb5bd"),
                boxpoints=False,  # don't show individual points, too noisy with 1k+ records
            ))
    fig.update_layout(
        **PLOT_LAYOUT, showlegend=False,
        title="Gameplay Narration Length",
        yaxis=dict(title="Characters", gridcolor="#373b3e"),
    )
    return fig


# ── Model performance figure builders ─────────────────────────────────────────

# Maps internal keys ("cnn", "nn", ...) to display labels
_MODEL_DISPLAY = {
    "cnn": "CNN", "nn": "NN",
    "transformer": "Transformer", "ensemble": "Ensemble",
}


def accuracy_bar_fig(results: dict) -> "go.Figure":
    """Bar chart comparing test-set accuracy across all available models."""
    _ensure_plotting()
    names, accs, colors = [], [], []
    # Only include models whose .npz files were found — missing models are skipped
    for key in ("cnn", "nn", "transformer", "ensemble"):
        if key in results:
            label = _MODEL_DISPLAY[key]
            names.append(label)
            accs.append(results[key]["accuracy"] * 100)
            colors.append(MODEL_COLORS[label])
    fig = go.Figure(go.Bar(
        x=names, y=accs, marker_color=colors,
        text=[f"{a:.1f}%" for a in accs], textposition="outside",
        width=0.5,  # narrower bars look cleaner with only 4 models
    ))
    fig.update_layout(
        **PLOT_LAYOUT,
        title="Test-Set Accuracy by Model",
        yaxis=dict(title="Accuracy (%)", range=[0, 115], gridcolor="#373b3e"),
        xaxis_title="Model",
    )
    return fig


def confusion_matrix_fig(results: dict, model_key: str) -> "go.Figure":
    """Heatmap of the confusion matrix for the selected model.

    Color is row-normalized (what fraction of true-class X was predicted as Y),
    but the cell text shows raw counts so you can judge absolute errors too.
    """
    _ensure_plotting()
    if model_key not in results:
        return go.Figure().update_layout(**PLOT_LAYOUT, title="No data")
    data    = results[model_key]
    cm      = data["cm"]
    lnames  = data["label_names"]
    # Normalize each row by its true-class total so colors show error rates, not counts
    row_sum = cm.sum(axis=1, keepdims=True).clip(1)  # clip(1) avoids divide-by-zero
    cm_norm = cm.astype(float) / row_sum
    label   = _MODEL_DISPLAY[model_key]
    fig = go.Figure(go.Heatmap(
        z=cm_norm.tolist(),
        x=lnames, y=lnames,
        colorscale="Blues",
        zmin=0, zmax=1,
        text=[[str(v) for v in row] for row in cm.tolist()],  # raw counts as labels
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
    """Bar chart: per-game accuracy for the selected model.

    Useful for spotting which games are harder to classify — the overall
    accuracy number can hide a game with a 70% rate dragging down the average.
    """
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
    """EDA tab: summary stats + five charts showing dataset characteristics."""
    _ensure_plotting()
    data   = load_eda_data()
    total  = data["total"]
    n_gg   = data["n_gg"]
    n_yt   = data["n_yt"]
    exp_ok = data["exp_ok"]

    return dbc.Container([
        # Row 1: quick summary cards at the top
        dbc.Row([
            dbc.Col(_stat_card(f"{total:,}",  "Total Records",        "#2196F3"), width=3),
            dbc.Col(_stat_card("5",            "Games Tracked",        "#9C27B0"), width=3),
            dbc.Col(_stat_card(f"{n_gg:,}",   "Google Image Records", "#F44336"), width=3),
            dbc.Col(_stat_card(f"{n_yt:,}",   "YouTube Records",      "#FF9800"), width=3),
        ], className="mb-4 mt-3 g-3"),

        # Row 2: records per game + source type breakdown side by side
        dbc.Row([
            dbc.Col(dcc.Graph(figure=records_per_game_fig(data), config=PLOT_CONFIG), width=6),
            dbc.Col(dcc.Graph(figure=source_type_fig(data),      config=PLOT_CONFIG), width=6),
        ], className="mb-3"),

        # Row 3: experience level distribution (full width — 5 stacked levels need space)
        dbc.Row([
            dbc.Col(dcc.Graph(figure=experience_dist_fig(data), config=PLOT_CONFIG), width=12),
        ], className="mb-3"),

        # Row 4: NULL density heatmap + narration length box plot
        # The heatmap is wider (7/12) since it has more fields to show
        dbc.Row([
            dbc.Col(dcc.Graph(figure=null_heatmap_fig(data),  config=PLOT_CONFIG), width=7),
            dbc.Col(dcc.Graph(figure=narration_box_fig(data), config=PLOT_CONFIG), width=5),
        ], className="mb-3"),
    ], fluid=True)


def build_performance_tab() -> dbc.Container:
    """Performance tab: overall accuracy bar + interactive confusion matrix / per-class chart.

    The confusion matrix and per-class chart update dynamically when the user
    selects a different model from the radio buttons — handled by the
    update_model_charts callback in callbacks.py.
    """
    _ensure_plotting()
    results = load_model_results()

    # Build the radio button options only for models that have saved .npz files
    options = [
        {"label": "CNN  (EfficientNet-B0 image classifier)",       "value": "cnn"},
        {"label": "NN  (Fully-connected on narration + metadata)",  "value": "nn"},
        {"label": "Transformer  (MiniLM-L6 text embeddings)",       "value": "transformer"},
        {"label": "Ensemble  (averaged softmax probabilities)",      "value": "ensemble"},
    ]
    available = [o for o in options if o["value"] in results]
    default   = available[0]["value"] if available else None

    return dbc.Container([
        # Full-width accuracy comparison bar chart at the top
        dbc.Row([
            dbc.Col(dcc.Graph(figure=accuracy_bar_fig(results), config=PLOT_CONFIG),
                    width=12),
        ], className="mb-3 mt-3"),

        # Radio buttons to pick which model's detail charts to show
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

        # The two detail charts — IDs are wired up to the callback in callbacks.py
        dbc.Row([
            dbc.Col(dcc.Graph(id="confusion-matrix-fig", config=PLOT_CONFIG), width=7),
            dbc.Col(dcc.Graph(id="perclass-fig",         config=PLOT_CONFIG), width=5),
        ]),
    ], fluid=True)


def build_live_demo_tab() -> dbc.Container:
    """Live demo tab: run each model on user-provided input and see predictions.

    Laid out as four stacked sections (CNN → Transformer → NN → Ensemble),
    separated by horizontal rules. Each section has its own input controls
    and output chart. The ensemble section reads from whatever is already
    filled in above it.
    """
    # Shared input styling to keep the dark theme consistent across text inputs
    _input_style = {
        "backgroundColor": "#343a40", "color": "#dee2e6",
        "border": "1px solid #6c757d", "borderRadius": "6px",
    }
    _textarea_style = {**_input_style, "width": "100%", "resize": "vertical",
                       "padding": "8px", "fontFamily": "inherit", "fontSize": "0.85rem"}

    return dbc.Container([

        # ── CNN ───────────────────────────────────────────────────────────────
        # User uploads an image; the callback runs the CNN and updates prediction-bar
        dbc.Row([
            dbc.Col(html.H5("CNN  ·  EfficientNet-B0 (screenshot)", className="mt-3 mb-0"), width=12),
            dbc.Col(html.P("Upload a gameplay screenshot and the CNN will classify it.",
                           className="text-muted small mb-2"), width=12),
        ]),
        dbc.Row([
            dbc.Col([
                dcc.Upload(
                    id="upload-image",
                    children=html.Div([
                        "Drag & Drop or ", html.A("click to browse"),
                        html.Br(),
                        html.Span("PNG, JPG, WEBP accepted", className="small text-muted"),
                    ]),
                    style={
                        "width": "100%", "height": "120px", "lineHeight": "55px",
                        "borderWidth": "2px", "borderStyle": "dashed", "borderRadius": "8px",
                        "borderColor": "#6c757d", "textAlign": "center",
                        "cursor": "pointer", "paddingTop": "10px",
                    },
                    multiple=False, accept="image/*",
                ),
                # Status line shows filename or error after upload
                html.Div(id="upload-status", className="mt-2 text-muted small"),
            ], width=4),
            # Preview column shows a thumbnail of the uploaded image
            dbc.Col(html.Div(id="preview-img-container"), width=3),
            dbc.Col(dcc.Graph(id="prediction-bar", config=PLOT_CONFIG,
                              figure=_empty_pred_fig("Upload a screenshot to see the prediction"),
                              style={"height": "280px"}), width=5),
        ], className="mb-4"),

        html.Hr(style={"borderColor": "#495057"}),

        # ── Transformer ───────────────────────────────────────────────────────
        # User types a narration; the callback scrubs game names and runs the Transformer
        dbc.Row([
            dbc.Col(html.H5("Transformer  ·  all-MiniLM-L6-v2 (narration text)",
                            className="mt-3 mb-0"), width=12),
            dbc.Col(html.P("Type or paste a gameplay narration and the Transformer will classify it.",
                           className="text-muted small mb-2"), width=12),
        ]),
        dbc.Row([
            dbc.Col([
                dcc.Textarea(
                    id="narration-input",
                    placeholder="Describe the gameplay scene in your own words…",
                    style={**_textarea_style, "height": "120px"},
                ),
                dbc.Button("Predict", id="transformer-predict-btn",
                           color="warning", size="sm", className="mt-2"),
                html.Div(id="transformer-status", className="mt-2 text-muted small"),
            ], width=5),
            dbc.Col(dcc.Graph(id="transformer-prediction-bar", config=PLOT_CONFIG,
                              figure=_empty_pred_fig("Enter a narration and click Predict"),
                              style={"height": "280px"}), width=7),
        ], className="mb-4"),

        html.Hr(style={"borderColor": "#495057"}),

        # ── NN ─────────────────────────────────────────────────────────────────
        # The NN takes both a narration (for the embedding) and structured metadata fields.
        # The flags (has_player_name, etc.) mirror the binary features used during training.
        dbc.Row([
            dbc.Col(html.H5("NN  ·  Gameplay features + narration embedding",
                            className="mt-3 mb-0"), width=12),
            dbc.Col(html.P("Fill in the gameplay metadata and narration; the NN combines "
                           "structured fields with a text embedding.",
                           className="text-muted small mb-2"), width=12),
        ]),
        dbc.Row([
            dbc.Col([
                dcc.Textarea(
                    id="nn-narration-input",
                    placeholder="Describe the gameplay scene…",
                    style={**_textarea_style, "height": "100px"},
                ),
                # Two dropdowns side by side: experience level and source type
                dbc.Row([
                    dbc.Col([
                        html.Label("Experience level", className="small text-muted mt-2 mb-1"),
                        dcc.Dropdown(
                            id="nn-exp-level",
                            options=[{"label": l, "value": l}
                                     for l in ["Poor", "Fair", "Good", "Excellent", "Superior"]],
                            value="Good",
                            clearable=False,
                            style={"color": "#000000"},  # dropdown text is always dark-on-white
                        ),
                    ], width=6),
                    dbc.Col([
                        html.Label("Source type", className="small text-muted mt-2 mb-1"),
                        dcc.Dropdown(
                            id="nn-source-type",
                            options=[{"label": "YouTube", "value": "youtube"},
                                     {"label": "Google Images", "value": "google_images"}],
                            value="youtube",
                            clearable=False,
                            style={"color": "#000000"},
                        ),
                    ], width=6),
                ], className="g-2"),
                # Binary flags — these map directly to the has_* features used during training
                dbc.Checklist(
                    id="nn-flags",
                    options=[
                        {"label": "Has player name",        "value": "has_player_name"},
                        {"label": "Has timestamp",          "value": "has_timestamp"},
                        {"label": "Has channel description","value": "has_channel_desc"},
                    ],
                    value=["has_player_name", "has_timestamp"],  # sensible defaults
                    inline=True,
                    className="mt-2 small",
                ),
                dbc.Button("Predict", id="nn-predict-btn",
                           color="success", size="sm", className="mt-2"),
                html.Div(id="nn-status", className="mt-2 text-muted small"),
            ], width=5),
            dbc.Col(dcc.Graph(id="nn-prediction-bar", config=PLOT_CONFIG,
                              figure=_empty_pred_fig("Fill in the form and click Predict"),
                              style={"height": "280px"}), width=7),
        ], className="mb-4"),

        html.Hr(style={"borderColor": "#495057"}),

        # ── Ensemble ──────────────────────────────────────────────────────────
        # Reads whatever inputs are already filled in above and combines them.
        # Each model is optional — if its input is missing the callback skips it
        # and reports what was skipped in the status line.
        dbc.Row([
            dbc.Col(html.H5("Ensemble  ·  Combined prediction",
                            className="mt-3 mb-0"), width=12),
            dbc.Col(html.P(
                "Select which models to combine. Each model uses the input "
                "already filled in above — fill in whichever sections you want "
                "to include before clicking Run.",
                className="text-muted small mb-2"), width=12),
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Checklist(
                    id="ensemble-models",
                    options=[
                        {"label": "CNN  (requires screenshot above)",        "value": "cnn"},
                        {"label": "Transformer  (requires narration above)",  "value": "transformer"},
                        {"label": "NN  (requires narration + form above)",    "value": "nn"},
                    ],
                    value=["cnn", "transformer", "nn"],  # all three selected by default
                    className="mb-3",
                ),
                dbc.Button("Run Ensemble", id="ensemble-run-btn",
                           color="primary", size="sm"),
                html.Div(id="ensemble-status", className="mt-2 text-muted small"),
            ], width=4),
            dbc.Col(dcc.Graph(id="ensemble-prediction-bar", config=PLOT_CONFIG,
                              figure=_empty_pred_fig("Select models and click Run Ensemble"),
                              style={"height": "280px"}), width=8),
        ], className="mb-4"),
    ], fluid=True)


# ── Methodology tab text constants ────────────────────────────────────────────
# Kept as module-level strings so they're easy to update without wading through
# the layout code. html.Pre renders them with whitespace preserved.

_CARD_1_BODY = (
    "An agentic pipeline searched Google/Bing Images and YouTube for gameplay "
    "screenshots across five games. Every candidate image was processed by "
    "the serving LLM for three tasks: "
    "(a) image validation -- rejecting ads, menus, and off-topic content; "
    "(b) narration generation -- a text description of the game state written "
    "independently of the image so it functions as a text feature for the Transformer; "
    "(c) experience-level classification -- rating visible player skill as "
    "Poor / Fair / Good / Excellent / Superior.\n\n"
    "Pass 1 used gemma4:26b (Gemma 4, 26B parameters, served locally via Ollama) "
    "for both data collection and enrichment.\n"
    "Pass 2 used ministral3:14b (Mistral 14B, served locally via Ollama) "
    "for both data collection and enrichment, with the same prompt templates "
    "and validation logic."
)

_CARD_2_BODY = (
    "Each pass included an offline enrichment phase to fill fields left empty "
    "during collection: player_name, gameplay_timestamp, channel_description, "
    "player_experience_narration, and identifying_quotes. Records were loaded "
    "in batches of five and each field updated only when the model returned a "
    "non-sentinel value, preserving any data already collected.\n\n"
    "The final dataset reflects contributions from both models across their "
    "respective collection and enrichment passes."
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
    "minimizes cross-entropy on the validation set (L-BFGS-B optimization)."
)


def build_methodology_tab() -> dbc.Container:
    """Methodology tab: explains how AI was used in data collection and modeling.

    Required by the project rubric. Shows four cards (collection, enrichment,
    validation, model architectures) plus a table of sample AI-generated narrations.
    """
    _ensure_plotting()
    data        = load_eda_data()
    sample_rows = data["sample_rows"]  # 2 narration samples per game, pulled from DB

    # Table of sample narrations so graders can see what the AI actually produced
    headers = ["Game", "Source", "Experience", "AI-Generated Narration"]
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
                _method_card("1  Data Collection - gemma4:26b & ministral3:14b via Ollama", _CARD_1_BODY),
                _method_card("2  Enrichment Passes - gemma4:26b & ministral3:14b",         _CARD_2_BODY),
                _method_card("3  Validating AI Correctness",                               _CARD_3_BODY),
                _method_card("4  Model Architectures",                                     _CARD_4_BODY),
                html.H5("Sample AI-Generated Narrations from the Dataset",
                        className="mt-4 mb-3"),
                table,
            ], width=12),
        ]),
    ], fluid=True)


# ── Shared component helpers ──────────────────────────────────────────────────


def _stat_card(value: str, label: str, color: str) -> dbc.Card:
    """Small metric card: big colored number on top, label below."""
    return dbc.Card(
        dbc.CardBody([
            html.H3(value, style={"color": color, "fontWeight": "700", "marginBottom": "4px"}),
            html.P(label, className="text-muted mb-0 small"),
        ]),
        className="text-center h-100",
    )


def _method_card(title: str, body: str) -> dbc.Card:
    """Expandable card with a bold header and pre-formatted text body."""
    return dbc.Card([
        dbc.CardHeader(html.Strong(title)),
        dbc.CardBody(
            # html.Pre preserves newlines and indentation in the body strings above
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
    Minimal static shell returned immediately on server start — no DB calls here.

    The actual tabs (EDA charts, model results, live demo) are built by
    build_tabs() and injected into _main-content by the serve_main_content
    callback once the browser connects. This keeps server startup under ~1 second
    even when DuckDB takes a moment to open the database.
    """
    return html.Div([
        # dcc.Location lets callbacks react to the URL — used to trigger the
        # initial content load without the user having to click anything
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
            # Loading spinner shown while build_tabs() runs on first page visit
            dcc.Loading(
                html.Div(id="_main-content", style={"minHeight": "300px"}),
                type="circle",
                color="#2196F3",
            ),
            fluid=True,
            className="mt-3",
        ),
    ])
