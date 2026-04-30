"""
Dash callbacks:
  - _url location     → build and inject the full tab layout on first page load
  - model-select radio → update confusion matrix and per-class accuracy charts
  - image upload       → CNN inference, image preview, prediction bar chart
"""
from __future__ import annotations

import plotly.graph_objects as go
from dash import Input, Output, State, callback, html

from .data_loader import load_model_results, predict_from_b64, GAME_COLORS
from .layout import confusion_matrix_fig, perclass_fig, PLOT_LAYOUT, PLOT_CONFIG


@callback(
    Output("_main-content", "children"),
    Input("_url", "pathname"),
)
def serve_main_content(_):
    """
    Triggered on first page load. Builds all tab panels (hits DuckDB, loads
    model results) and injects them into the page. Subsequent renders use the
    cached lru_cache results so they are instant.
    """
    from .layout import build_tabs
    return build_tabs()


@callback(
    Output("confusion-matrix-fig", "figure"),
    Output("perclass-fig", "figure"),
    Input("model-select", "value"),
)
def update_model_charts(model_key: str):
    """Swap the confusion matrix and per-class accuracy when the user picks a model."""
    results = load_model_results()
    return confusion_matrix_fig(results, model_key), perclass_fig(results, model_key)


@callback(
    Output("preview-img-container", "children"),
    Output("prediction-bar", "figure"),
    Output("upload-status", "children"),
    Input("upload-image", "contents"),
    State("upload-image", "filename"),
)
def update_prediction(contents, filename):
    """Run CNN inference on the uploaded image and render the prediction bar chart."""
    empty_fig = (
        go.Figure()
        .update_layout(**PLOT_LAYOUT, title="Upload an image to see the prediction")
    )

    if contents is None:
        return html.Div(), empty_fig, ""

    # Image preview
    preview = html.Img(
        src=contents,
        style={"width": "100%", "borderRadius": "6px", "marginTop": "8px"},
    )

    try:
        probs = predict_from_b64(contents)
    except Exception as exc:
        err_fig = go.Figure().update_layout(**PLOT_LAYOUT, title=f"Inference error: {exc}")
        return preview, err_fig, f"Error processing {filename}: {exc}"
    if not probs:
        no_model = go.Figure().update_layout(**PLOT_LAYOUT, title="CNN model not found — run game-cnn-train first")
        return preview, no_model, f"Uploaded: {filename}"

    # Sort ascending so the highest bar appears at the top of a horizontal chart
    sorted_items = sorted(probs.items(), key=lambda kv: kv[1])
    games  = [g for g, _ in sorted_items]
    scores = [p * 100 for _, p in sorted_items]
    colors = [GAME_COLORS.get(g, "#adb5bd") for g in games]
    top    = max(probs, key=probs.get)

    fig = go.Figure(go.Bar(
        x=scores, y=games,
        orientation="h",
        marker_color=colors,
        text=[f"{s:.1f}%" for s in scores],
        textposition="outside",
    ))
    fig.update_layout(**PLOT_LAYOUT)
    fig.update_layout(
        title=f"Predicted: <b>{top}</b>  ({probs[top]*100:.1f}% confidence)",
        xaxis=dict(title="Confidence (%)", range=[0, 118], gridcolor="#373b3e"),
        height=300,
        margin=dict(l=110, r=30, t=60, b=40),
    )

    return preview, fig, f"Uploaded: {filename}"
