"""Validate UI requests and coordinate model inference in the Docker stack."""

# api.py

# FastAPI core components used to define the web API and receive form data
from fastapi import FastAPI, Form

# JSONResponse is used to return structured JSON success/error messages
from fastapi.responses import JSONResponse

# Internal service function that contains the core prediction workflow
from utils import predict_service

# Standard logging module
import logging


# Create the FastAPI application instance
app = FastAPI()

# Configure the logging system so debug/info/error messages are visible
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s: %(message)s")

# Module-level logger used throughout this file
logger = logging.getLogger(__name__)


async def perform_prediction(
    folder_name: str,
    channels: list[str],
    models: list[str],
    resolution: str | None,
    run_ensemble_flag: bool,
    model_set: str = "adult",
    ensemble_method: str = "soft_vote",
    review_software: str = "somnomedics",
    preprocessing: dict | None = None,
):
    """
    Wrapper around predict_service().

    This helper keeps the route handler small and centralizes exception handling.
    It passes the folder name, selected channels, selected models, optional
    resolution, and the ensemble flag down into the core prediction function.
    """
    try:
        # Delegate the actual workflow to utils.predict_service()
        response = predict_service(
            folder_name=folder_name,
            channels=channels,
            models=models,
            log=logger,
            resolution=resolution,
            run_ensemble_flag=run_ensemble_flag,
            model_set=model_set,
            ensemble_method=ensemble_method,
            review_software=review_software,
            preprocessing=preprocessing,
        )
        if "error" in response:
            return JSONResponse(content=response, status_code=502)
        return response

    except Exception as e:
        # Log the error and convert it into a JSON HTTP error response
        logger.error(f"An error occurred: {str(e)}")
        return JSONResponse(
            content={"error": f"An error occurred in api.py: {e}"},
            status_code=500,
        )


@app.post("/predict")
async def predict(
    folder_name: str = Form(...),
    channels: list[str] = Form(...),
    models: list[str] = Form(...),
    resolution: str | None = Form(None),
    run_ensemble_flag: bool = Form(False),
    model_set: str = Form("adult"),
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
    Main prediction endpoint.

    This route receives form data from the GUI and passes it to the internal
    prediction service.

    Required form fields:
      - folder_name: used for both input and output folder naming
      - channels: list of selected modalities (e.g. ["EEG"], ["EEG", "EOG"])
      - models: list of selected models (e.g. ["usleep", "deepresnet"])

    Optional form fields:
      - resolution: temporal resolution, forwarded as pred_sec to run.py
      - run_ensemble_flag: whether to let uSLEEPYLAND-PLUS run.py perform
        the final cross-model ensemble (--run_ensemble)
    """
    # There are no children EMG (chin) weights; refuse instead of failing mid-run.
    requested = {c.strip().lower() for value in channels for c in value.split(",")}
    if model_set == "children" and requested & {"emg", "chin"}:
        return JSONResponse(
            content={"error": "EMG is not available with Children weights: there are no "
                              "children EMG models. Use Adult weights or remove EMG."},
            status_code=400,
        )
    return await perform_prediction(
        folder_name=folder_name,
        channels=channels,
        models=models,
        resolution=resolution,
        run_ensemble_flag=run_ensemble_flag,
        model_set=model_set,
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
    import os
    import threading

    logger.info("Shutdown requested from the GUI; stopping manager-api.")
    # Exit shortly after replying, so the GUI receives the response first.
    threading.Timer(0.5, os._exit, args=(0,)).start()
    return {"message": "manager-api stopping"}
