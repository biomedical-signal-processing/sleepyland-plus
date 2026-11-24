"""Send prediction requests from Flask to the manager API service."""

import requests  # Used to make HTTP requests from this service to the manager-api backend


# URL of the manager-api service endpoint that handles prediction for one batch.
# In docker-compose, the service is named "manager-api" and exposes port 8989,
# so inside the Docker network it is reachable as "http://manager-api:8989".
# The path /predict is implemented by manager-api to:
#   - Read EDF files from the shared input volume (/app/input/<folder_name>),
#   - Run the selected models,
#   - Optionally run ensemble (if "ensemble" is present in models),
#   - Write results and status.json into the shared output volume (/app/output/<folder_name>).
manager_predict_url = "http://manager-api:8989/predict"


def sent_to_prediction_service(
    folder_name,
    channels,
    models_selected,
    model_set="adult",
    resolution="30",
    ensemble_method="soft_vote",
    review_software="somnomedics",
    preprocessing=None,
):
    """
    Send a prediction request to the manager-api service for a single batch of EDF files.

    This function is called from the GUI's /process route, after:
      - The EDF files have been saved to /app/input/<folder_name>,
      - The user has chosen folderName in the GUI,
      - The user has selected available channels and models.

    With this simplified interface, `folder_name` is used for BOTH:
      - locating input at /app/input/<folder_name>,
      - writing output under /app/output/<folder_name>.

    Parameters
    ----------
    folder_name : str
        Name chosen by the user (e.g., "patient_batch_01").
        It identifies:
          - the input directory:  /app/input/<folder_name>
          - the output directory: /app/output/<folder_name>

    channels : list or str
        The list of modalities selected by the user (e.g., ["EEG", "EOG"]).
        In the GUI route, this list is uppercased before being passed here.
        When sent as POST data, FastAPI will parse it as a list[str] thanks
        to the `list[str] = Form(...)` annotation in manager-api.

    models_selected : list or str
        The list of model identifiers selected by the user
        (e.g., ["usleep", "deepresnet", "ensemble"]).
        manager-api:
          - interprets the presence of "ensemble" as a request to run the
            final ensemble in uSLEEPYLAND-PLUS (sets run_ensemble_flag=True),
          - removes "ensemble" from the list,
          - forwards only real model names (e.g. ["usleep","deepresnet"])
            to the uSLEEPYLAND-PLUS service.

    Returns
    -------
    requests.Response
        The HTTP response object returned by manager-api. The caller (in routes.py)
        inspects response.status_code to decide whether prediction succeeded or failed.
    """
    # Construct and send a POST request to manager-api.
    # The data is sent as form-encoded key-value pairs (application/x-www-form-urlencoded),
    # which manager-api reads via Form(...) parameters in its /predict endpoint.
    #
    # manager-api expects:
    #   - folder_name        : str
    #   - channels           : list[str]
    #   - models             : list[str] (may include "ensemble")
    #   - resolution         : Optional[str] (omitted here -> defaults to None)
    #   - run_ensemble_flag  : bool (manager-api derives this from "ensemble" in models)
    return requests.post(
        manager_predict_url,
        data={
            "folder_name": folder_name,   # manager-api will use /app/input/<folder_name> and /app/output/<folder_name>
            "channels": channels,         # list or string of channels (e.g. ["EEG", "EOG"])
            "models": models_selected,    # list or string of models (e.g. ["usleep","deepresnet","ensemble"])
            "model_set": model_set,       # subject type: 'adult' or 'child'
            "resolution": resolution,     # prediction resolution in seconds (forwarded as pred_sec to run.py)
            "ensemble_method": ensemble_method,
            "review_software": review_software,
            **(preprocessing or {}),
        },
    )
