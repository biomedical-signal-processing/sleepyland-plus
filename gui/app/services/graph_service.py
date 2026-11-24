"""Convert saved probability arrays into Plotly hypnograms and hypnodensity plots."""

import numpy as np                  # For numerical operations and loading .npy arrays
import os                           # For filesystem navigation and directory checks
import plotly.graph_objects as go   # Plotly Figure/Scatter objects for building plots
import json                         # For saving Plotly figures as JSON strings

STAGE_COLORS = {
    0: '#505C65',  # Wake
    1: '#427D83',  # N1
    2: '#355D88',  # N2
    3: '#3D7053',  # N3
    4: '#7F4147',  # REM
}
PLOT_BACKGROUND = '#0C1729'
PLOT_BORDER = '#364965'
PLOT_LABEL = '#A9BAD3'
PLOT_HIGHLIGHT = '#72E1C1'
PRED_LINE = '#F2F6FC'   # near-white: highest contrast on the navy background
HYPNOGRAM_STAGE_ORDER = ['NREM3', 'NREM2', 'NREM1', 'REM', 'Wake']
HYPNOGRAM_STAGE_INDEX = {'NREM3': 3, 'NREM2': 2, 'NREM1': 1, 'REM': 4, 'Wake': 0}


def _style_plot(fig, hypnogram=False):
    """Apply the shared high-contrast palette to both graph types."""
    fig.update_layout(
        plot_bgcolor=PLOT_BACKGROUND,
        paper_bgcolor=PLOT_BACKGROUND,
        font=dict(color=PLOT_LABEL),
        hoverlabel=dict(bgcolor=PLOT_BACKGROUND, bordercolor=PLOT_HIGHLIGHT,
                        font=dict(color=PLOT_LABEL)),
        xaxis=dict(showline=True, mirror=True, linecolor=PLOT_BORDER,
                   gridcolor='rgba(54,73,101,0.45)', zeroline=False,
                   tickfont=dict(color=PLOT_LABEL), title_font=dict(color=PLOT_LABEL), ticksuffix='m'),
        yaxis=dict(showline=True, mirror=True, linecolor=PLOT_BORDER,
                   gridcolor='rgba(54,73,101,0.45)', zeroline=False,
                   tickfont=dict(color=PLOT_LABEL), title_font=dict(color=PLOT_LABEL)),
    )
    if hypnogram:
        for position, stage in enumerate(HYPNOGRAM_STAGE_ORDER):
            fig.add_shape(type='rect', xref='paper', yref='y', x0=0, x1=1,
                          y0=position - 0.5, y1=position + 0.5,
                          fillcolor=STAGE_COLORS[HYPNOGRAM_STAGE_INDEX[stage]],
                          opacity=0.30, line_width=0, layer='below')


def _stage_trace(x, y, name, color):
    """One thin step line; no markers, which crowd a full night into a blur."""
    return go.Scatter(
        x=x, y=y, mode='lines', name=name,
        line=dict(color=color, width=1, shape='hv'),
        hovertemplate='%{x:.1f} min<br>%{y}<extra>' + name + '</extra>',
    )


# (Commented-out softmax implementation: kept here for reference)
"""def softmax(logits):
    # Compute the numerically stable softmax function for a 2D array of shape (N, C)
    logits_stable = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(logits_stable)
    sum_exp = np.sum(exp_values, axis=1, keepdims=True)
probabilities = exp_values / sum_exp
    return probabilities"""


def _ensemble_prediction_folder(folder_name, resolution, lead=None):
    """Return the ensemble probabilities directory for soft-vote runs."""
    ensemble_dir = f"/app/output/{folder_name}/ensemble/{resolution}s"
    lead_dirs = (f"_{lead}",) if lead else ("_all", "_eeg", "_eog", "_chin")
    for lead_dir in lead_dirs:
        for output_name in ("soft-voting", "majority"):
            prediction_dir = os.path.join(ensemble_dir, lead_dir, output_name)
            if os.path.isdir(prediction_dir):
                return prediction_dir
    # Preserve the legacy path in warnings when neither output was produced.
    return os.path.join(ensemble_dir, "_all", "soft-voting")


def create_hypnogram(folder_name, models_selected, log, resolution="30"):
    """
    Create a predicted-hypnogram plot for each selected model and save it as JSON.

    This function:
      - Reads model outputs from /app/output/<folder_name>/<model>/... (see below).
      - Builds Plotly figures of predicted sleep stages over time (hypnograms).
      - Saves each figure as a JSON file under /app/static/<model>/.
      - Writes a total_slides.json file with the number of "slides" (records) available.

    The models_selected list contains:
      - "ensemble" / "ensemble_<lead>" for ensemble outputs
      - "<base_model>_<lead_type>" for per-modality models (e.g. "usleep_eeg", "usleep_eog")

    For non-ensemble tags, majority predictions are expected under:
      /app/output/<folder_name>/<base_model>/<resolution>s/_<lead_type>/majority

    For ensemble tags, predictions are read from soft voting or the legacy majority directory.
    """
    total_slides = 0  # Will store number of .npy files (slides) for the last processed model
    sleep_stage_labels = ['Wake', 'NREM1', 'NREM2', 'NREM3', 'REM']

    for model_tag in models_selected:
        # Resolve the prediction folder for an ensemble tag or a base_model + lead_type tag
        if model_tag == "ensemble" or model_tag.startswith("ensemble_"):
            lead = model_tag.split("_", 1)[1] if "_" in model_tag else None
            majority_folder = _ensemble_prediction_folder(folder_name, resolution, lead)
        else:
            try:
                base_model, lead_type = model_tag.split("_", 1)
            except ValueError:
                log.warning(f"Model tag '{model_tag}' has no lead_type suffix; skipping.")
                continue
            majority_folder = f"/app/output/{folder_name}/{base_model}/{resolution}s/_{lead_type}/majority"
        model = model_tag  # full tag is used for static output naming (e.g. usleep_eeg)

        if not os.path.exists(majority_folder):
            log.warning(f"Majority folder not found for {model_tag}: {majority_folder}")
            continue

        majority_files = sorted([file for file in os.listdir(majority_folder) if file.endswith('_PRED.npy')])
        if not majority_files:
            log.warning(f"No majority .npy files found in {majority_folder} for {model_tag}")
            continue

        # Number of available slides (e.g., number of nights or subjects); same for all models.
        total_slides = len(majority_files)

        for i, maj_file in enumerate(majority_files):
            raw = np.load(os.path.join(majority_folder, maj_file))
            # Shape can be (N, C) at 30 s or (N, K, C) at finer resolutions.
            # Flatten sub-epochs so downstream code always sees (N*K, C).
            if raw.ndim == 3:
                raw = raw.reshape(-1, raw.shape[-1])
            sleep_stages = raw.argmax(-1).astype(int)

            time = np.arange(len(sleep_stages)) * int(resolution) / 60
            stage_names = [sleep_stage_labels[stage] for stage in sleep_stages]

            fig = go.Figure()
            fig.add_trace(_stage_trace(time, stage_names, 'Pred', PRED_LINE))
            fig.update_layout(
                title=f"{maj_file.split('.')[0].split('_')[0]}",
                xaxis=dict(title='Time (min)'),
                yaxis=dict(
                    title='Sleep Stage',
                    categoryorder='array',
                    categoryarray=HYPNOGRAM_STAGE_ORDER,
                ),
                yaxis_range=[-0.5, 4.5],
            )
            _style_plot(fig, hypnogram=True)

            os.makedirs(f"/app/static/{model}", exist_ok=True)
            # Filenames follow pattern: hypnogram_combined_<model>_<index>.json
            with open(f"/app/static/{model}/hypnogram_combined_{model}_{i}.json", "w") as f:
                json.dump(fig.to_json(), f)

    # Save the total number of slides so the frontend knows how many records it can navigate.
    with open("/app/static/total_slides.json", "w") as f:
        json.dump({"total_slides": total_slides}, f)


def create_hypnodensity_graph(folder_name, models_selected, log, is_logits=False, resolution="30"):
    """
    Create hypnodensity (cumulative probability) plots for each model and save them as JSON.

    Hypnodensity graphs show the cumulative probability distribution over sleep stages
    across time, often visualized as stacked area plots.

    Notes with run.py integration
    -----------------------------
    The models_selected list contains:
      - "ensemble"
      - "<base_model>_<lead_type>" (e.g. "usleep_eeg", "deepresnet_eog")

    For non-ensemble tags, majority predictions are expected under:
      /app/output/<folder_name>/<base_model>/30s/_<lead_type>/majority

    For the ensemble tag, predictions are read from soft voting or the legacy majority directory.
    """
    for model_tag in models_selected:

        # Resolve majority_folder as in create_hypnogram
        if model_tag == "ensemble" or model_tag.startswith("ensemble_"):
            lead = model_tag.split("_", 1)[1] if "_" in model_tag else None
            majority_folder = _ensemble_prediction_folder(folder_name, resolution, lead)
            model = model_tag
        else:
            try:
                base_model, lead_type = model_tag.split("_", 1)
            except ValueError:
                log.warning(f"Model tag '{model_tag}' has no lead_type suffix; skipping.")
                continue

            majority_folder = f"/app/output/{folder_name}/{base_model}/{resolution}s/_{lead_type}/majority"
            model = model_tag

        if not os.path.exists(majority_folder):
            log.warning(f"Majority folder not found for {model_tag}: {majority_folder}")
            continue

        majority_files = sorted([file for file in os.listdir(majority_folder) if file.endswith('_PRED.npy')])

        if not majority_files:
            log.warning(f"No majority .npy files found in {majority_folder} for {model_tag}")
            continue

        # Iterate over each majority file (one per slide)
        for i, maj_file in enumerate(majority_files):
            sleep_probabilities_majority = np.load(
                os.path.join(majority_folder, maj_file)
            )
            # Flatten (N, K, C) → (N*K, C) for finer-than-30s resolutions
            if sleep_probabilities_majority.ndim == 3:
                sleep_probabilities_majority = sleep_probabilities_majority.reshape(
                    -1, sleep_probabilities_majority.shape[-1]
                )

            # Compute cumulative probabilities along class axis for stacked-plot visualization:
            cumulative_probs = np.cumsum(sleep_probabilities_majority, axis=1)

            # Human-readable names for each stage (order must match columns in probabilities)
            stage_names = ['Wake', 'N1', 'N2', 'N3', 'REM']

            # Initialize a Plotly figure
            fig = go.Figure()

            # For each class index j, add a cumulative probability curve
            for j in range(cumulative_probs.shape[1]):
                fig.add_trace(
                    go.Scatter(
                        x=np.arange(0, len(cumulative_probs)) * int(resolution) / 60,
                        y=cumulative_probs[:, j],               # Cumulative probabilities on y-axis
                        mode='lines',
                        name=stage_names[j],
                        line=dict(width=0.5, color=PLOT_BORDER),
                        fill='tonexty',                          # Fill to next trace to create stacked effect
                        fillcolor=STAGE_COLORS[j],
                        hoverinfo='none'
                    )
                )

            fig.update_layout(
                title=f"{maj_file.split('.')[0].split('_')[0]}",
                xaxis_title='Time (min)',
                yaxis_title='Cumulative Probability',
                yaxis_range=[0, 1],
                showlegend=True,
            )
            _style_plot(fig)

            fig_json = fig.to_json()

            if not os.path.exists(f"/app/static/{model}"):
                os.makedirs(f"/app/static/{model}")

            with open(f"/app/static/{model}/hypnodensity_{model}_{i}.json", "w") as f:
                json.dump(fig_json, f)
