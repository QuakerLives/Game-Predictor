"""
Dash callbacks:
  - _url location         → build and inject the full tab layout on first page load
  - model-select radio    → update confusion matrix and per-class accuracy charts
  - image upload          → CNN inference, image preview, prediction bar chart
  - narration text        → Transformer inference, prediction bar chart
  - gameplay form         → NN inference, prediction bar chart
  - ensemble-run-btn      → combine whichever models have inputs and show averaged result
"""
from __future__ import annotations

import plotly.graph_objects as go
from dash import Input, Output, State, callback, html

from .data_loader import (
    load_model_results, predict_from_b64,
    predict_from_narration, predict_from_gameplay_features,
    GAME_COLORS,
)
from .layout import confusion_matrix_fig, perclass_fig, PLOT_LAYOUT, PLOT_CONFIG, _empty_pred_fig


def _prob_bar(probs: dict, title: str) -> go.Figure:
    """Build a horizontal confidence bar chart from a {game: probability} dict.

    Sorted ascending so the highest-confidence game appears at the top.
    Used by all three individual model callbacks and the ensemble callback.
    """
    sorted_items = sorted(probs.items(), key=lambda kv: kv[1])
    games  = [g for g, _ in sorted_items]
    scores = [p * 100 for _, p in sorted_items]
    colors = [GAME_COLORS.get(g, "#adb5bd") for g in games]
    fig = go.Figure(go.Bar(
        x=scores, y=games, orientation="h",
        marker_color=colors,
        text=[f"{s:.1f}%" for s in scores],
        textposition="outside",
    ))
    fig.update_layout(**PLOT_LAYOUT)
    fig.update_layout(
        title=title,
        # Range goes to 118 so the outside text labels have room
        xaxis=dict(title="Confidence (%)", range=[0, 118], gridcolor="#373b3e"),
        height=280,
        margin=dict(l=110, r=30, t=60, b=40),
    )
    return fig


@callback(
    Output("_main-content", "children"),
    Input("_url", "pathname"),
)
def serve_main_content(_):
    """Triggered on first page load — builds all tab panels and injects them.

    Subsequent renders use the lru_cache results in data_loader.py so they
    are instant. Doing this lazily (on first browser connection rather than
    at import time) keeps server startup fast.
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
    if contents is None:
        # Nothing uploaded yet — show placeholder chart and empty status
        return html.Div(), _empty_pred_fig("Upload a screenshot to see the prediction"), ""

    # Show a thumbnail preview of whatever was uploaded
    preview = html.Img(
        src=contents,
        style={"width": "100%", "borderRadius": "6px", "marginTop": "8px"},
    )

    try:
        probs = predict_from_b64(contents)
    except Exception as exc:
        return preview, _empty_pred_fig(f"Inference error: {exc}"), f"Error processing {filename}: {exc}"

    if not probs:
        # Model file not found — user needs to train first
        return preview, _empty_pred_fig("CNN model not found — run game-cnn-train first"), f"Uploaded: {filename}"

    # Build the confidence bar chart with the top prediction in the title
    top = max(probs, key=probs.get)
    fig = _prob_bar(probs, f"Predicted: <b>{top}</b>  ({probs[top]*100:.1f}% confidence)")
    return preview, fig, f"Uploaded: {filename}"


@callback(
    Output("transformer-prediction-bar", "figure"),
    Output("transformer-status", "children"),
    Input("transformer-predict-btn", "n_clicks"),
    State("narration-input", "value"),
    prevent_initial_call=True,  # don't fire on page load — only when the button is clicked
)
def update_transformer_prediction(_, narration):
    """Run Transformer inference on the typed narration."""
    empty = _empty_pred_fig("Enter a narration and click Predict")
    if not narration or not narration.strip():
        return empty, ""
    try:
        probs = predict_from_narration(narration)
    except Exception as exc:
        return _empty_pred_fig(f"Error: {exc}"), str(exc)
    if not probs:
        return _empty_pred_fig("Transformer model not found — run game-transformer-train first"), ""
    top = max(probs, key=probs.get)
    return _prob_bar(probs, f"Predicted: <b>{top}</b>  ({probs[top]*100:.1f}% confidence)"), ""


@callback(
    Output("nn-prediction-bar", "figure"),
    Output("nn-status", "children"),
    Input("nn-predict-btn", "n_clicks"),
    State("nn-narration-input", "value"),
    State("nn-exp-level", "value"),
    State("nn-source-type", "value"),
    State("nn-flags", "value"),
    prevent_initial_call=True,
)
def update_nn_prediction(_, narration, exp_level, source_type, flags):
    """Run NN gameplay inference on the filled form."""
    empty = _empty_pred_fig("Fill in the fields and click Predict")
    if not narration or not narration.strip():
        return empty, ""
    # flags is None if no checkboxes are ticked — default to empty list
    flags = flags or []
    try:
        probs = predict_from_gameplay_features(
            narration=narration,
            experience_level=exp_level or "Good",
            source_type=source_type or "youtube",
            has_player_name="has_player_name" in flags,
            has_timestamp="has_timestamp" in flags,
            has_channel_desc="has_channel_desc" in flags,
        )
    except Exception as exc:
        return _empty_pred_fig(f"Error: {exc}"), str(exc)
    if not probs:
        return _empty_pred_fig("NN model not found — run 'game-nn --gameplay' and retrain to save scaler"), ""
    top = max(probs, key=probs.get)
    return _prob_bar(probs, f"Predicted: <b>{top}</b>  ({probs[top]*100:.1f}% confidence)"), ""


@callback(
    Output("ensemble-prediction-bar", "figure"),
    Output("ensemble-status", "children"),
    Input("ensemble-run-btn", "n_clicks"),
    State("ensemble-models", "value"),
    State("upload-image", "contents"),
    State("narration-input", "value"),
    State("nn-narration-input", "value"),
    State("nn-exp-level", "value"),
    State("nn-source-type", "value"),
    State("nn-flags", "value"),
    prevent_initial_call=True,
)
def update_ensemble_prediction(_, selected, img_contents, trans_narration,
                                nn_narration, exp_level, source_type, flags):
    """Combine predictions from whichever models are selected and have inputs.

    Any model without its required input (no image for CNN, no narration for
    Transformer/NN) is skipped and noted in the status line rather than failing.
    """
    import numpy as np

    empty = _empty_pred_fig("Select models and click Run Ensemble")
    selected = selected or []
    if not selected:
        return empty, "Select at least one model."

    all_probs: list[dict] = []
    skipped: list[str] = []
    flags = flags or []

    if "cnn" in selected:
        if img_contents:
            try:
                p = predict_from_b64(img_contents)
                if p:
                    all_probs.append(p)
                else:
                    skipped.append("CNN (model not found)")
            except Exception as exc:
                skipped.append(f"CNN (error: {exc})")
        else:
            skipped.append("CNN (no image uploaded)")

    if "transformer" in selected:
        if trans_narration and trans_narration.strip():
            try:
                p = predict_from_narration(trans_narration)
                if p:
                    all_probs.append(p)
                else:
                    skipped.append("Transformer (model not found)")
            except Exception as exc:
                skipped.append(f"Transformer (error: {exc})")
        else:
            skipped.append("Transformer (no narration entered)")

    if "nn" in selected:
        # Fall back to the Transformer's narration field if the NN box was left empty
        narr = nn_narration if (nn_narration and nn_narration.strip()) else trans_narration
        if narr and narr.strip():
            try:
                p = predict_from_gameplay_features(
                    narration=narr,
                    experience_level=exp_level or "Good",
                    source_type=source_type or "youtube",
                    has_player_name="has_player_name" in flags,
                    has_timestamp="has_timestamp" in flags,
                    has_channel_desc="has_channel_desc" in flags,
                )
                if p:
                    all_probs.append(p)
                else:
                    skipped.append("NN (model not found — retrain to save scaler)")
            except Exception as exc:
                skipped.append(f"NN (error: {exc})")
        else:
            skipped.append("NN (no narration entered)")

    if not all_probs:
        status = "No models could run.  " + "  |  ".join(skipped)
        return _empty_pred_fig("No predictions available"), status

    # Average the probability dicts across however many models ran successfully
    games = list(all_probs[0].keys())
    avg = {g: float(np.mean([p.get(g, 0.0) for p in all_probs])) for g in games}
    top = max(avg, key=avg.get)
    n = len(all_probs)
    title = f"Ensemble ({n} model{'s' if n > 1 else ''}): <b>{top}</b>  ({avg[top]*100:.1f}%)"
    status = ("Skipped: " + "  |  ".join(skipped)) if skipped else f"{n} model(s) combined."
    return _prob_bar(avg, title), status
