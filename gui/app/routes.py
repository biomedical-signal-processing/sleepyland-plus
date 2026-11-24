"""HTTP routes for EDF upload, prediction requests, plots, and saved runs."""

# Core Flask imports: Blueprint groups routes, request gives access to HTTP data,
# render_template renders HTML files, send_file is used for file downloads.
from flask import Blueprint, request, render_template, send_file

# Service that triggers the actual prediction workflow in the backend stack
# (e.g., manager-api / usleepyland).
from app.services.prediction_service import sent_to_prediction_service

# Services that read prediction outputs and generate Plotly-ready JSON
# for hypnograms and hypnodensity plots.
from app.services.graph_service import create_hypnogram, create_hypnodensity_graph

# Utility to validate uploaded files (not used in the routes shown, but imported for possible future checks).
from app.utils.file_validation import validate_files

# Utilities to handle directories and uploaded files:
# - clear_directory: remove files from a directory.
# - save_uploaded_files / save_edf_files: persist incoming files to disk.
from app.utils.file_handling import (
    clear_directory,
    save_uploaded_files,
    save_edf_files,
)

# Centralized logging configuration for this app.
from app.utils.logging_config import configure_logging

# jsonify converts Python objects (dicts, lists) to JSON HTTP responses.
from flask import jsonify

# Standard libraries:
import pandas as pd  # Currently unused here; may be used in other parts of the app.
import os            # For filesystem operations (list directories, check files).
import json          # For reading/writing JSON files (e.g., status.json).

# Initialize a logger for this module, using the shared logging configuration.
log = configure_logging()

# Create a Blueprint named 'main'. This allows you to register these routes
# in your main Flask app under a common prefix if desired.
bp = Blueprint('main', __name__)
def _scan_output_folder(folder_name: str):
    """
    Reconstruct expanded_models and per-tag resolution by inspecting the output folder.

    Output structure written by run.py:
      <folder>/<base_model>/<Xs>/_<lead>/majority/*.npy   (e.g. usleep/5s/_eeg/majority/)
      <folder>/ensemble/<Xs>/_<lead>/soft-voting/*.npy
      <folder>/ensemble/<Xs>/_all/soft-voting/*.npy  (multiple leads)
      <folder>/ensemble/<Xs>/_all/majority/*.npy  (legacy soft-vote output)

    Returns
    -------
    expanded_models : list[str]
        e.g. ['usleep_eeg', 'usleep_eog', 'deepresnet_eeg', 'ensemble']
    resolution_map : dict[str, str]
        tag → resolution string without 's'  (e.g. {'usleep_eeg': '5', 'deepresnet_eeg': '30'})
    """
    base_dir = f"/app/output/{folder_name}"
    expanded_models = []
    resolution_map = {}

    for base_model in ('usleep', 'deepresnet', 'sleeptransformer'):
        model_dir = os.path.join(base_dir, base_model)
        if not os.path.isdir(model_dir):
            continue

        # Pick the first *s folder (e.g. '30s', '5s')
        res_folder = next(
            (e for e in sorted(os.listdir(model_dir))
             if e.endswith('s') and os.path.isdir(os.path.join(model_dir, e))),
            None,
        )
        if not res_folder:
            continue
        resolution = res_folder[:-1]   # '5s' → '5'

        res_dir = os.path.join(model_dir, res_folder)

        for entry in sorted(os.listdir(res_dir)):
            if not entry.startswith('_'):
                continue
            lead_type = entry[1:]       # '_eeg' → 'eeg'
            majority_dir = os.path.join(res_dir, entry, 'majority')
            if os.path.isdir(majority_dir) and any(
                f.endswith('_PRED.npy') for f in os.listdir(majority_dir)
            ):
                tag = f"{base_model}_{lead_type}"
                expanded_models.append(tag)
                resolution_map[tag] = resolution

    # Ensemble
    ensemble_dir = os.path.join(base_dir, 'ensemble')
    if os.path.isdir(ensemble_dir):
        res_folder = next(
            (e for e in sorted(os.listdir(ensemble_dir))
             if e.endswith('s') and os.path.isdir(os.path.join(ensemble_dir, e))),
            None,
        )
        if res_folder:
            ensemble_prediction_dir = next(
                (
                    os.path.join(ensemble_dir, res_folder, lead_dir, output_name)
                    for lead_dir in ('_all', '_eeg', '_eog', '_chin')
                    for output_name in ('soft-voting', 'majority')
                    if os.path.isdir(os.path.join(ensemble_dir, res_folder, lead_dir, output_name))
                    and any(
                        f.endswith('_PRED.npy')
                        for f in os.listdir(os.path.join(ensemble_dir, res_folder, lead_dir, output_name))
                    )
                ),
                None,
            )
            if ensemble_prediction_dir:
                expanded_models.append('ensemble')
                resolution_map['ensemble'] = res_folder[:-1]
            combined_dir = os.path.join(ensemble_dir, res_folder, '_all', 'soft-voting')
            has_combined = os.path.isdir(combined_dir) and any(
                f.endswith('_PRED.npy') for f in os.listdir(combined_dir)
            )
            for lead in ('eeg', 'eog', 'chin') if has_combined else ():
                prediction_dir = os.path.join(ensemble_dir, res_folder, f'_{lead}', 'soft-voting')
                if os.path.isdir(prediction_dir) and any(
                    f.endswith('_PRED.npy') for f in os.listdir(prediction_dir)
                ):
                    tag = f'ensemble_{lead}'
                    expanded_models.append(tag)
                    resolution_map[tag] = res_folder[:-1]

    return expanded_models, resolution_map


def _write_status(folder_name: str, percent: int, message: str) -> None:
    """Write a progress status file readable by the /progress/<folder_name> endpoint."""
    status_path = f"/app/output/{folder_name}/status.json"
    os.makedirs(os.path.dirname(status_path), exist_ok=True)
    with open(status_path, "w") as f:
        json.dump({"percent": percent, "message": message}, f)


@bp.route('/')
def index():
    """
    Render the main HTML page of the GUI.
    When the user accesses the root URL, this returns index.html,
    which contains the Bootstrap layout and your JavaScript.
    """
    return render_template('index.html')


@bp.route('/get_studies', methods=['GET'])
def get_studies():
    """
    Return a JSON list of all completed study folders in /app/output.

    Each subdirectory under /app/output is treated as one "study" (i.e., one run),
    so this endpoint can be used by the frontend to show a list of previous runs.
    """
    output_folder = '/app/output'

    # Collect all names in /app/output that are directories (ignore plain files).
    studies = [
        name
        for name in os.listdir(output_folder)
        if os.path.isdir(os.path.join(output_folder, name))
    ]

    # Return the list as JSON with HTTP 200.
    return jsonify(studies), 200


@bp.post('/load/<folder_name>')
async def load_study(folder_name):
    """
    Regenerate hypnogram and hypnodensity JSON plots for a previously computed study.

    Scans /app/output/<folder_name>/ to reconstruct which models were run and at what
    resolution, then calls the graph service to re-create the static JSON files that
    the frontend fetches for plotting.
    """
    try:
        output_dir = f"/app/output/{folder_name}"
        if not os.path.isdir(output_dir):
            return jsonify({'error': f'Study not found: {folder_name}'}), 404

        expanded_models, resolution_map = _scan_output_folder(folder_name)

        if not expanded_models:
            return jsonify({'error': 'No model outputs found in this study.'}), 404

        # Group tags by resolution so we call graph functions once per resolution bucket.
        # In practice all tags in a single run share the same resolution, but we handle
        # the mixed case correctly just in case.
        from collections import defaultdict
        by_resolution = defaultdict(list)
        for tag in expanded_models:
            by_resolution[resolution_map.get(tag, '30')].append(tag)

        for resolution, models in by_resolution.items():
            create_hypnogram(folder_name, models, log, resolution=resolution)
            create_hypnodensity_graph(folder_name, models, log, resolution=resolution)

        return jsonify({'expanded_models': expanded_models, 'folder_name': folder_name}), 200

    except Exception as e:
        log.error(f"Error loading study {folder_name}: {e}")
        return jsonify({'error': str(e)}), 500


@bp.post('/delete/<folder_name>')
def delete_study(folder_name):
    """Remove a study's output folder from /app/output/<folder_name>."""
    import shutil
    folder_path = f"/app/output/{folder_name}"
    if not os.path.isdir(folder_path):
        return jsonify({'error': f'Study not found: {folder_name}'}), 404
    try:
        shutil.rmtree(folder_path)
        return jsonify({'message': f'{folder_name} deleted.'}), 200
    except Exception as e:
        log.error(f"Error deleting study {folder_name}: {e}")
        return jsonify({'error': str(e)}), 500


@bp.route('/output/<path:filename>', methods=['GET'])
def download_csv(filename):
    """
    Allow the user to download any file located under /app/output.

    Example: /output/folder_name/results.csv will serve /app/output/folder_name/results.csv
    as a downloadable attachment in the browser.
    """
    # send_file streams the file to the client. as_attachment=True tells the browser
    # to treat it as a file download instead of trying to render it inline.
    return send_file(f'/app/output/{filename}', as_attachment=True)


@bp.post("/reset-directories")
async def reset_directories():
    """
    Reset the input directory used by the GUI for EDF uploads.

    This is called:
      - Directly by the frontend when the user hits the Reset button.
      - Internally by /process before starting a new run.

    It ensures no leftover files from previous runs remain in ./input.
    """
    try:
        input_directory = "./input"

        # Remove all files (and possibly subfolders) inside ./input.
        clear_directory(input_directory)

        # Return a simple JSON confirmation message.
        return {"message": "Directories reset successfully"}
    except Exception as e:
        # If something goes wrong (e.g., permission issues), return an error JSON.
        return {"error": f"An error occurred: {e}"}


@bp.post("/quit")
def quit_system():
    """
    Stop all three SLEEPYLAND containers.

    The containers have no Docker CLI, so each service exposes /shutdown and
    exits its own process, which stops its container. The GUI stops last so it
    can still answer the browser. Containers stay on disk; the launcher (or
    `docker compose up`) starts them again.
    """
    import threading
    import requests

    stopped, failed = [], []
    for name, url in (("usleepyland", "http://usleepyland:7777/shutdown"),
                      ("manager-api", "http://manager-api:8989/shutdown")):
        try:
            requests.post(url, timeout=5).raise_for_status()
            stopped.append(name)
        except requests.RequestException as e:
            # A service that is already down cannot answer; keep stopping the rest.
            log.warning(f"Could not stop {name}: {e}")
            failed.append(name)

    log.info("Shutdown requested from the UI; stopping the GUI container.")
    threading.Timer(1.0, os._exit, args=(0,)).start()
    return jsonify({"message": "SLEEPYLAND is stopping", "stopped": stopped, "unreachable": failed})


@bp.route('/progress/<folder_name>')
def get_progress(folder_name):
    """
    Report progress for a given study/folder.

    The progress is read from /app/output/<folder_name>/status.json, which is expected
    to be written periodically by the prediction pipeline or manager service.

    This endpoint is polled by the frontend (script.js) every second while a job is running.
    """
    # This path must point to the shared output volume that all containers see.
    status_path = f"/app/output/{folder_name}/status.json"

    # If a status.json exists, read it and return its JSON content.
    if os.path.exists(status_path):
        with open(status_path, 'r') as f:
            return jsonify(json.load(f))

    # If no status.json yet, return a default "starting" state.
    return jsonify({"percent": 0, "message": "Starting process..."})


@bp.route('/process', methods=['POST'])
async def process():
    """
    Main entry point for running sleep staging on one batch of EDF files.

    This route is called by the GUI when the user clicks the Predict button.
    It orchestrates the following steps:
      1. Read the uploaded EDF files and selected options from the request.
      2. Reset the input directory to avoid mixing runs.
      3. Save the new EDF files into an input subfolder named after folderName.
      4. Call the prediction service (manager-api/usleepyland) to run the models.
      5. After successful prediction, generate hypnogram and hypnodensity JSON files.
      6. Return a JSON message indicating completion or error.
    """
    # Read form fields sent by the frontend via FormData.
    folder_name = request.form.get('folderName')          # Folder chosen by the user (for input & output).
    uploaded_files = request.files.getlist('edf-files')   # List of uploaded EDF files.
    models_selected = request.form.get('models')          # Comma-separated model IDs.
    channels_selected = request.form.get('channels')      # Comma-separated channel names.
    model_set = request.form.get('model_set', 'adult')    # Subject type: 'adult' or 'child'.
    resolution = request.form.get('resolution', '30')     # Prediction resolution in seconds (USleep only).
    ensemble_method = request.form.get('ensemble_method', 'soft_vote')
    review_software = request.form.get('review_software', '')
    preprocessing = {key: request.form.get(key) for key in (
        'apply_notch', 'notch_freq', 'apply_bandpass', 'bandpass_low',
        'bandpass_high', 'apply_quality_control', 'apply_channel_qc',
        'max_bad_epoch_fraction')}
    if review_software not in {'somnomedics', 'compumedics', 'none'}:
        return jsonify({'error': 'Select SOMNOmedics, Compumedics, or None.'}), 400

    # Convert comma-separated strings into Python lists.
    channels_selected = channels_selected.split(",")
    models_selected = models_selected.split(",")

    # Reset the input directory before processing this new batch.
    # This clears ./input so we start from a clean state.
    await reset_directories()

    # Normalize channel names to uppercase (e.g., "eeg" -> "EEG") so downstream
    # services can rely on a standard format.
    channels_selected = [x.upper() for x in channels_selected]

    # ------------------------------------------------------------------
    # Derive lead types for plotting from the selected channels.
    #
    # run.py writes majority predictions per lead_type (eeg, eog, chin),
    # so we build a parallel list of "model_lead" tags like "usleep_eeg"
    # for visualization purposes. Manager-api still receives the original
    # models_selected list (without the lead suffix).
    # ------------------------------------------------------------------
    lead_types = []
    if "EEG" in channels_selected:
        lead_types.append("eeg")
    if "EOG" in channels_selected:
        lead_types.append("eog")
    # run.py writes EMG predictions under the internal "chin" lead type.
    if "CHIN" in channels_selected or "EMG" in channels_selected:
        lead_types.append("chin")

    # Build expanded model tags for plotting: "<base_model>_<lead_type>".
    # Ensemble is kept as a special tag "ensemble".
    expanded_models = []
    for m in models_selected:
        if m == "ensemble":
            expanded_models.append("ensemble")
        else:
            for lt in lead_types:
                expanded_models.append(f"{m}_{lt}")

    try:
        # Save the uploaded EDF files into /app/input/<folder_name>, logging operations along the way.
        # Example: if folder_name="patient_batch_01", files go to /app/input/patient_batch_01.
        save_edf_files(uploaded_files, folder_name, log)

        # Call the prediction service to run the selected models on the EDF files.
        # With the simplified signature, folder_name is used for BOTH:
        #   - folder_root_name (input): manager-api will look in /app/input/<folder_name>.
        #   - folder_name (output): manager-api will write to /app/output/<folder_name>.
        #
        # channels_selected: list of channels (e.g. ["EEG", "EOG"]).
        # models_selected  : list of base model IDs (e.g. ["usleep", "deepresnet", "ensemble"]).
        response = sent_to_prediction_service(
            folder_name,
            channels_selected,
            models_selected,
            model_set,
            resolution,
            ensemble_method,
            review_software,
            preprocessing,
        )

        # Propagate any manager or inference-service failure to the frontend.
        # In particular, manager-api returns 502 when the downstream
        # uSLEEPYLAND service rejects or fails a run.py invocation.
        if response.status_code >= 400:
            return (
                jsonify(
                    {'error': f'Prediction failed with status code: {response.status_code}'}
                ),
                response.status_code
            )

        # If prediction completed successfully, generate hypnogram JSONs.
        # We now use expanded_models so that each <model, lead_type> pair
        # (e.g. "usleep_eeg", "usleep_eog") is treated as a distinct curve,
        # and "ensemble" remains a special global model.
        expanded_models, resolution_map = _scan_output_folder(folder_name)
        from collections import defaultdict
        by_resolution = defaultdict(list)
        for tag in expanded_models:
            by_resolution[resolution_map.get(tag, resolution)].append(tag)

        _write_status(folder_name, 90, "Generating hypnograms…")
        for plot_resolution, tags in by_resolution.items():
            create_hypnogram(folder_name, tags, log, resolution=plot_resolution)

        _write_status(folder_name, 95, "Generating hypnodensity…")
        for plot_resolution, tags in by_resolution.items():
            create_hypnodensity_graph(folder_name, tags, log, resolution=plot_resolution)

        # Return a success message to the frontend.
        # The frontend will now:
        #   - Stop polling progress,
        #   - Build result cards,
        #   - Fetch the generated JSONs,
        #   - Render the interactive plots.
        return jsonify({
            'message': 'Processing complete',
            'expanded_models': expanded_models,
        }), 200

    except Exception as e:
        # Log any unexpected exception and return a 500 error to the client.
        log.error(f"An error occurred: {str(e)}")
        return jsonify({'error': f'An error occurred app.py: {e}'}), 500
