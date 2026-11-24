"""Expose the bundled EDF inference engine through the Docker HTTP service."""

# FastAPI core imports for defining routes and parsing form data
from fastapi import FastAPI, Response, Form

# Response types used by the API routes
from fastapi.responses import JSONResponse

# Type hints used throughout the service
from typing import List, Optional

# Used to run the external uSLEEPYLAND-PLUS run.py command
import subprocess

# Filesystem utilities
import os

# Logging for debugging and traceability
import logging


# FastAPI application instance for the usleepyland-plus service
app = FastAPI()

# Configure logging globally for this service
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _normalize_channel_types(channels: List[str]) -> str:
    """
    Map GUI channel selections (EEG/EOG/EMG) to the `lead_types` string
    expected by uSLEEPYLAND-PLUS run.py.

    - GUI "EEG" -> "eeg"
    - GUI "EOG" -> "eog"
    - GUI "EMG" -> "chin" (internal EMG/CHIN lead type)

    If no channels are selected, return "all" so run.py decides what to use.
    """
    lead_types = set()

    for ch in channels:
        ch_u = ch.upper()
        if ch_u == "EEG":
            lead_types.add("eeg")
        elif ch_u == "EOG":
            lead_types.add("eog")
        elif ch_u == "EMG":
            # README: internal lead name is `chin`
            lead_types.add("chin")

    if not lead_types:
        # Let run.py pick leads based on EDF contents
        return "all"

    # Stable ordering for nicer logs
    return ",".join(sorted(lead_types))


async def run_prediction(
    models: List[str],
    folder_name: str,
    channels: List[str],
    file_name: str,
    resolution: Optional[str],
    model_set: str = "adult",
    run_ensemble: bool = True,
    ensemble_method: str = "soft_vote",
    review_software: str = "somnomedics",
    preprocessing: Optional[dict] = None,
) -> JSONResponse:
    """
    Run prediction for a single EDF file using uSLEEPYLAND-PLUS run.py.

    We delegate to run.py for:
      - loading the requested models (usleep, deepresnet, sleeptransformer, ...)
      - resolving weights under ./weights/<family>/<model>-nsrr-2024[_lead]
      - optional final ensemble across models via --run_ensemble

    Docker layout assumptions:
      - WORKDIR = /app/uSLEEPYLAND
      - run.py is at /app/uSLEEPYLAND/run.py
      - weights are at /app/uSLEEPYLAND/weights/...

    Paths:
      - EDF input:  /app/input/<folder_name>/<file_name>
      - Outputs:    ./<site_id>/... relative to /app/uSLEEPYLAND
                    (c == folder_name)
    """
    try:
        # Full EDF path for this run
        edf_path = os.path.join("/app/input", folder_name, file_name)

        if not os.path.exists(edf_path):
            return Response(content=f"EDF file not found: {edf_path}", status_code=400)

        # Map GUI channel selection to run.py lead_types
        lead_types_arg = _normalize_channel_types(channels)

        # uSLEEPYLAND-PLUS expects comma-separated model names in one string
        # e.g. "usleep,deepresnet,sleeptransformer"
        model_names_str = ",".join(models)

        # pred_sec: use GUI resolution if valid, else default 30
        try:
            pred_sec = int(resolution) if resolution not in (None, "") else 30
        except ValueError:
            pred_sec = 30

        # Fixed number of classes in your pipeline
        n_classes = 5

        # Build the command for run.py in single-file mode
        command = [
            "python",
            "run.py",
            "--edf_file",
            edf_path,
            "--site_id",
            folder_name,
            "--output_root",
            "/app/output",
            "--model_name",
            model_names_str,
            "--lead_types",
            lead_types_arg,
            "--pred_sec",
            str(pred_sec),
            "--n_classes",
            str(n_classes),
            "--model_set",
            model_set,
            "--review_software",
            review_software,
        ]

        # Let run.py perform its own cross-model ensemble if requested
        preprocessing = preprocessing or {}
        if preprocessing.get('apply_notch'):
            command += ['--apply_notch', '--notch_freq', str(preprocessing['notch_freq'])]
        if preprocessing.get('apply_bandpass'):
            command += ['--apply_bandpass', '--bandpass_low', str(preprocessing['bandpass_low']),
                        '--bandpass_high', str(preprocessing['bandpass_high'])]
        if not preprocessing.get('apply_quality_control', True):
            command.append('--no_quality_control')
        if preprocessing.get('apply_channel_qc'):
            command += ['--apply_channel_qc', '--max_bad_epoch_fraction',
                        str(preprocessing['max_bad_epoch_fraction'])]

        if run_ensemble:
            command.append("--run_ensemble")
            command.append("--ensemble_method")
            command.append(ensemble_method)

        logger.debug(f"Running uSLEEPYLAND-PLUS command: {command}")

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            logger.error(
                "run.py failed (exit %s): %s\nstderr: %s",
                result.returncode,
                command,
                result.stderr.strip(),
            )
            return Response(
                content=f"run.py error (exit {result.returncode}): {result.stderr}",
                status_code=500,
            )

        logger.info(
            "run.py completed successfully for site_id=%s, models=%s, file=%s",
            folder_name,
            model_names_str,
            file_name,
        )

        return JSONResponse(
            content={
                "message": "Prediction completed successfully.",
                "site_id": folder_name,
                "models": models,
                "lead_types": lead_types_arg,
                "pred_sec": pred_sec,
                "n_classes": n_classes,
                "model_set": model_set,
                "ensemble_run": run_ensemble,
            },
            status_code=200,
        )

    except Exception as e:
        error_message = f"An error occurred in prediction: {e}"
        logger.error(error_message, exc_info=True)
        return Response(content=error_message, status_code=500)


@app.post("/predict")
async def prediction(
    folder_name: str = Form(...),
    channels: List[str] = Form(...),
    model: Optional[str] = Form(None),
    models: List[str] = Form([]),
    file_name: str = Form(...),
    resolution: Optional[str] = Form(None),
    model_set: str = Form("adult"),
    run_ensemble_flag: bool = Form(True),
    ensemble_method: str = Form("soft_vote"),
    review_software: str = Form("somnomedics"),
    apply_notch: bool = Form(False),
    notch_freq: float = Form(50.0),
    apply_bandpass: bool = Form(False),
    bandpass_low: float = Form(0.3),
    bandpass_high: float = Form(35.0),
    apply_quality_control: bool = Form(True),
    apply_channel_qc: bool = Form(False),
    max_bad_epoch_fraction: float = Form(0.2),
):
    """
    Main API endpoint: run prediction for a single EDF file.

    Accepts both the old pattern (single `model`) and the new pattern (`models`):

    - New calls:
        models=["usleep","deepresnet"]

    The folder_name is used as site_id in run.py and also as the key
    under /app/input and the output tree.
    """
    # Backward compatibility: if `model` is provided but `models` is empty, wrap it
    if model and not models:
        models = [model]

    if not models:
        return Response(content="No model(s) specified.", status_code=400)

    if ensemble_method not in {"soft_vote"}:
        return Response(content="Invalid ensemble method.", status_code=400)
    if review_software not in {"somnomedics", "compumedics", "none"}:
        return Response(content="Invalid review software.", status_code=400)
    if not 0 <= max_bad_epoch_fraction <= 1:
        return Response(content="max_bad_epoch_fraction must be between 0 and 1.", status_code=400)

    return await run_prediction(
        models=models,
        folder_name=folder_name,
        channels=channels,
        file_name=file_name,
        resolution=resolution,
        model_set=model_set,
        run_ensemble=run_ensemble_flag,
        ensemble_method=ensemble_method,
        review_software=review_software,
        preprocessing=dict(apply_notch=apply_notch, notch_freq=notch_freq,
                           apply_bandpass=apply_bandpass, bandpass_low=bandpass_low,
                           bandpass_high=bandpass_high, apply_quality_control=apply_quality_control,
                           apply_channel_qc=apply_channel_qc, max_bad_epoch_fraction=max_bad_epoch_fraction),
    )


@app.post("/shutdown")
async def shutdown():
    """Stop this container when the user clicks Stop in the GUI."""
    import threading

    logger.info("Shutdown requested from the GUI; stopping usleepyland.")
    # Exit shortly after replying; any running run.py child ends with the container.
    threading.Timer(0.5, os._exit, args=(0,)).start()
    return {"message": "usleepyland stopping"}
