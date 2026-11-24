"""Forward work and status requests to the inference service."""

# utils.py

# HTTP client used to talk to downstream services inside the Docker network
import requests

# Standard filesystem utilities
import os
import json

# Logging
import logging


# Configure logging for this module so debug and error output is visible
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# URL of the downstream uSLEEPYLAND-PLUS service used by manager-api
PREDICT_URL = "http://usleepyland:7777/predict"


def update_job_status(folder_name: str, percent: int, message: str) -> None:
    """
    Write a progress status JSON file for the current job.

    The GUI polls this file via /progress/<folder_name> so it can show
    a live progress bar and status message to the user.

    Parameters
    ----------
    folder_name : str
        Logical job identifier used as the top-level folder under /app/output.

    percent : int
        Progress percentage between 0 and 100.

    message : str
        Human-readable status message describing the current step.
    """
    try:
        # Build the status file path inside the shared output volume
        status_path = f"/app/output/{folder_name}/status.json"

        # Ensure the parent folder exists
        os.makedirs(os.path.dirname(status_path), exist_ok=True)

        # Write the progress information as JSON
        with open(status_path, "w") as f:
            json.dump({"percent": percent, "message": message}, f)

    except Exception as e:
        # Log any failure to update the status file
        logger.error(f"Could not update status file: {e}")


def predict_service(
    folder_name: str,
    channels: list[str],
    models: list[str],
    log: logging.Logger,
    resolution: str | None,
    run_ensemble_flag: bool,
    model_set: str = "adult",
    ensemble_method: str = "soft_vote",
    review_software: str = "somnomedics",
    preprocessing: dict | None = None,
) -> dict:
    """
    Run predictions for all EDF files in /app/input/<folder_name>.

    Behaviour
    ---------
    - Uses `folder_name` as the single key for:
        input:  /app/input/<folder_name>
        output: /app/output/<folder_name>
    - Reads all .edf files from the input folder.
    - For each EDF, calls the uSLEEPYLAND-PLUS service once, passing:
        * all selected models in a single request
        * selected channels
        * optional resolution (pred_sec)
        * ensemble flag (mapped to --run_ensemble)
    - Updates job progress by EDF file (not by model).
    - Returns a simple success or error summary.

    Ensemble handling
    -----------------
    - The GUI may either:
        * include "ensemble" in the `models` list, or
        * send an explicit `run_ensemble_flag`.
    - Here we treat "ensemble" as a UI signal only:
        * If "ensemble" is present in `models`, we set run_ensemble_flag=True
          and remove "ensemble" from the model list.
        * The actual ensemble computation is done inside run.py when
          run_ensemble_flag is True.

    Parameters
    ----------
    folder_name : str
        Job/site identifier, used as folder name under /app/input and /app/output.

    channels : list[str]
        Channel selection from the GUI (e.g. ["EEG"], ["EEG", "EOG"]).

    models : list[str]
        Selected models and possibly the string "ensemble" if the user
        toggled the ensemble option.

    log : logging.Logger
        Logger instance to be used for error/debug output.

    resolution : str | None
        Optional temporal resolution; forwarded as-is to usleepyland, where it
        is interpreted as pred_sec by run.py.

    run_ensemble_flag : bool
        Initial ensemble flag (may also be set based on "ensemble" appearing
        in `models`).

    Returns
    -------
    dict
        A simple response describing success or error for the whole job.
    """
    # Initialize progress tracking for this job
    update_job_status(folder_name, 0, "Initializing process...")

    response = None

    # Build the input folder path from the folder name provided by the GUI
    edf_folder = f"/app/input/{folder_name}"

    # Read all EDF files from the selected folder
    try:
        edf_files = [f for f in os.listdir(edf_folder) if f.endswith(".edf")]
    except FileNotFoundError:
        log.error(f"Input folder not found: {edf_folder}")
        return {"error": f"Input folder not found: {edf_folder}"}

    # Stop early if the folder does not contain EDF files
    if not edf_files:
        log.error(f"No EDF files found in {edf_folder}")
        return {"error": "No EDF files found"}

    # Interpret 'ensemble' as a UI flag, not as an actual model name.
    # This preserves the old GUI logic where "ensemble" was in the model list.
    if "ensemble" in models:
        run_ensemble_flag = True
        models = [m for m in models if m != "ensemble"]

    # After removing "ensemble", ensure there is at least one real model to run
    if not models:
        log.error("No models provided to predict_service after removing 'ensemble'.")
        return {"error": "No models provided"}

    # Number of steps: one step per EDF file
    num_files = len(edf_files)
    total_steps = num_files
    current_step = 0

    # Progress is split into 0-85% for inference; 85-100% is reserved for
    # graph generation in the GUI (routes.py writes those updates).
    for f_idx, edf_file in enumerate(edf_files):
        current_step += 1

        # Start of this file: at least 5% so the user sees activity immediately.
        # With 1 file: always 5%. With N files: evenly distributed in [5, 85).
        start_pct = max(5, int(f_idx / num_files * 85))
        end_pct   = int((f_idx + 1) / num_files * 85)

        models_str = ", ".join(models)
        update_job_status(
            folder_name, start_pct,
            f"Running {models_str} on file {f_idx + 1}/{num_files}…"
        )

        logger.debug(f"folder_name: {folder_name}")
        logger.debug(f"input_folder: {edf_folder}")
        logger.debug(f"channels: {channels}")
        logger.debug(f"models: {models}")
        logger.debug(f"file_name: {edf_file}")
        logger.debug(f"resolution: {resolution}")
        logger.debug(f"run_ensemble_flag: {run_ensemble_flag}")

        response = requests.post(
            PREDICT_URL,
            data={
                "folder_name": folder_name,
                "channels": channels,
                "models": models,
                "file_name": edf_file,
                "resolution": resolution,
                "model_set": model_set,
                "ensemble_method": ensemble_method,
                "review_software": review_software,
                **(preprocessing or {}),
                "run_ensemble_flag": str(run_ensemble_flag).lower(),
            },
        )

        if response.status_code != 200:
            error_message = (
                f"Prediction failed for {edf_file} with models={models}. "
                f"Status: {response.status_code}, body: {response.text}"
            )
            log.error(error_message)
            update_job_status(folder_name, start_pct, error_message)
            return {"error": error_message}
        else:
            update_job_status(
                folder_name, end_pct,
                f"File {f_idx + 1}/{num_files} complete."
            )

    return {"message": "Prediction completed successfully."}
