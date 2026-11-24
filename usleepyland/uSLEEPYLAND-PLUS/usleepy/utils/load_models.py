"""Model registry, model loading, and prediction (one-shot and sliding window)."""

import logging
import os
import math
from pathlib import Path
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

def get_and_load_model(model_dir, hparams, weights_file_name=None, clear_previous=True):
    """
    Initializes a model in model_dir according to hparams and loads weights
    in .h5 file at path 'weights_file_name' or automatically determined from
    the 'model' sub-folder under 'model_dir' if not specified.

    Args:
        model_dir:        Path to project folder
        hparams:            A YAMLHParams object storing hyperparameters
        weights_file_name:  Optional path to .h5 parameter file
        clear_previous:     Clear previous keras session before initializing new model graph.

    Returns:
        Parameter-initialized model
    """
    if not weights_file_name:
        from usleepy.models.model_init import init_and_load_best_model
        model, _ = init_and_load_best_model(
            hparams=hparams,
            model_dir=os.path.join(model_dir, "model"),
            clear_previous=clear_previous,
            by_name=True
        )
    else:
        from usleepy.models.model_init import init_and_load_model
        weights_file_name = os.path.join(model_dir, "model", weights_file_name)
        model = init_and_load_model(hparams=hparams,
                                    weights_file=weights_file_name,
                                    clear_previous=clear_previous,
                                    by_name=True)
    return model

def get_and_load_one_shot_model(n_periods, model_dir, hparams, weights_file_name=None, clear_previous=True):
    """
    Returns a model according to 'hparams', potentially initialized from
    parameters in a .h5 file 'weights_file_name'.

    Independent of the settings in 'hparams', the returned model will be
    configured in 'one shot' mode - that is the model will predict on an entire
    PSG input in one forward pass. The 'full_hypnogram' array is used to
    determine the corresponding number of segments.

    Args:
        n_periods:          Number of epochs that the model should score in 1 forward pass
        model_dir:        Path to project directory
        hparams:            YAMLHparams object
        weights_file_name:  Optional path to .h5 parameter file
        clear_previous:     Clear previous keras session before initializing new model graph.

    Returns:
        Initialized model
    """
    # Set seguence length
    hparams["build"]["batch_shape"][1] = n_periods
    hparams["build"]["batch_shape"][0] = 1  # Should not matter
    return get_and_load_model(model_dir,
                              hparams=hparams,
                              weights_file_name=weights_file_name,
                              clear_previous=clear_previous)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

SUPPORTED_MODELS = {"usleep", "deepresnet", "sleeptransformer"}

# Models that output logits -> apply softmax after prediction
LOGIT_MODELS         = {"sleeptransformer"}
# Models that use one-shot (full-recording) prediction
ONE_SHOT_MODELS      = {"usleep"}
# Only usleep supports sub-epoch temporal resolution
SUPPORTS_TEMPORAL_RESOLUTION = {"usleep"}
# Only usleep saves intermediate layer representations
SUPPORTS_REPRESENTATIONS     = {"usleep"}



# ---------------------------------------------------------------------------
# usleepy model helpers
# ---------------------------------------------------------------------------

def load_usleepy_hparams(model_dir: str):
    from usleepy.hyperparameters import YAMLHParams
    from usleepy import Defaults

    hparams_path = Defaults.get_hparams_path(model_dir)
    hparams = YAMLHParams(hparams_path, no_version_control=True)

    datasets = hparams.get("datasets")
    if datasets:
        ds_path = list(datasets.values())[0]
        if not os.path.isabs(ds_path):
            ds_path = os.path.join(Defaults.get_hparams_dir(model_dir), ds_path)
        hparams.update(YAMLHParams(ds_path, no_version_control=True))

    return hparams


def load_usleepy_model(
    model_dir: str,
    hparams,
    one_shot: bool,
    n_periods: int,
    weights_dir: Optional[Path] = None,
    lead_type: Optional[str] = None,
):
    """
    Load a usleepy model, optionally overriding the weights location.

    Parameters
    ----------
    weights_dir : if given, weights are loaded from
                  <weights_dir>/<lead_type>/model_weights.h5
                  and passed as weights_file_name to the usleepy loader.
                  If None, usleepy resolves weights from inside model_dir.
    lead_type   : required when weights_dir is given.
    """
    weights_file_name = None
    if weights_dir is not None:
        if lead_type is None:
            raise ValueError("lead_type must be provided when weights_dir is set.")
        w_path = Path(weights_dir) / lead_type / "model_weights.h5"
        if not w_path.exists():
            raise FileNotFoundError(
                f"Weights not found at {w_path}. "
                f"Expected: <weights_dir>/<lead_type>/model_weights.h5"
            )
        weights_file_name = str(w_path)
        logger.info("Loading weights from explicit path: %s", weights_file_name)

    if one_shot:
        from usleepy.utils.load_models import get_and_load_one_shot_model
        return get_and_load_one_shot_model(
            n_periods=n_periods,
            model_dir=model_dir,
            hparams=hparams,
            weights_file_name=weights_file_name,
        )
    else:
        from usleepy.utils.load_models import get_and_load_model
        return get_and_load_model(
            model_dir=model_dir,
            hparams=hparams,
            weights_file_name=weights_file_name,
        )



# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def run_usleep_one_shot(
    sig_proc: np.ndarray,
    n_periods: int,
    data_per_period: int,
    model,
    layer_names: Optional[list] = None,
):
    """One-shot forward pass (usleep). Returns (pred_raw, repr_arr)."""
    from tensorflow.keras.models import Model as KerasModel

    if layer_names is None:
        layer_names = ["average_pool"]

    X = np.expand_dims(
        sig_proc.reshape(n_periods, data_per_period, 1).astype(np.float32), 0
    )

    matched = [l.output for l in model.layers if l.name in layer_names]
    multi   = KerasModel(inputs=model.input, outputs=matched + [model.output])
    outputs = multi.predict_on_batch(X)

    feat_outputs = outputs[:-1]
    pred_raw     = outputs[-1]
    repr_arr     = np.asarray(feat_outputs).squeeze() if feat_outputs else None

    return pred_raw, repr_arr


def run_usleepy_sliding_window(
    sig_proc: np.ndarray,
    n_periods: int,
    data_per_period: int,
    model,
    hparams,
    is_logits: bool,
) -> np.ndarray:
    """Sliding-window batched prediction. Returns pred [n_periods, n_classes]."""
    from usleepy.utils.preprocessing import compute_spectrogram
    from usleepy.utils.preprocessing import batch_scaling

    psg = sig_proc.reshape(n_periods, data_per_period, 1).astype(np.float32)

    num_epochs_per_batch = hparams.get("build").get("batch_shape")[1]
    total_batches        = math.ceil(n_periods / num_epochs_per_batch)
    last_batch_epochs    = n_periods % num_epochs_per_batch

    batched = np.zeros(
        (total_batches, num_epochs_per_batch, data_per_period, 1), dtype=np.float32
    )
    for i in range(total_batches):
        start = i * num_epochs_per_batch
        end   = start + num_epochs_per_batch
        if i == total_batches - 1 and last_batch_epochs != 0:
            batched[i] = psg[-num_epochs_per_batch:]
        else:
            batched[i] = psg[start:end]

    if hparams.get("fit", {}).get("get_spectrogram"):
        batched = compute_spectrogram(batched)
        if hparams.get("scaler"):
            batch_scaling(batched, hparams["scaler"])

    pred = model.predict_on_batch(batched)
    if is_logits:
        from usleepy.utils.nn import softmax
        logger.info("Applying softmax to predictions of shape: {}".format(pred.shape))
        pred = softmax(pred, axis=-1)

    pred = pred.reshape(-1, pred.shape[-1])
    if last_batch_epochs != 0:
        n_invalid = num_epochs_per_batch - last_batch_epochs
        pred = np.concatenate([pred[:-num_epochs_per_batch], pred[-last_batch_epochs:]])

    return pred


def normalise_pred(
    pred_raw,
    n_periods: int,
    preds_per_epoch: int,
    n_classes: int,
    ch_name: str,
) -> np.ndarray:
    """Normalise any model output shape to [n_periods, preds_per_epoch, n_classes]."""
    pred = np.asarray(pred_raw)
    if callable(getattr(pred, "numpy", None)):
        pred = pred.numpy()

    if pred.ndim == 4:
        pred = pred[0]

    if pred.ndim == 3:
        if pred.shape[0] == 1 and pred.shape[-1] == n_classes:
            pred = pred[0].reshape(n_periods, 1, n_classes)
        elif pred.shape[-1] == n_classes and pred.shape[1] == preds_per_epoch:
            pass
        elif pred.shape[1] == n_classes and preds_per_epoch == 1:
            pred = pred.reshape(n_periods, 1, n_classes)
        else:
            raise ValueError(
                f"Unexpected 3-D pred shape {pred.shape} for {ch_name} "
                f"(n_periods={n_periods}, preds_per_epoch={preds_per_epoch}, n_classes={n_classes})"
            )

    elif pred.ndim == 2:
        if pred.shape[1] != n_classes:
            raise ValueError(
                f"Unexpected 2-D pred shape {pred.shape} for {ch_name} "
                f"(expected n_classes={n_classes})"
            )
        pred = pred.reshape(n_periods, 1, n_classes)

    else:
        raise ValueError(f"Unexpected pred ndim={pred.ndim} for {ch_name}")

    if pred.shape[0] != n_periods:
        raise ValueError(
            f"n_periods mismatch for {ch_name}: "
            f"pred.shape[0]={pred.shape[0]}, expected {n_periods}"
        )
    if pred.shape[-1] != n_classes:
        raise ValueError(
            f"n_classes mismatch for {ch_name}: "
            f"pred.shape[-1]={pred.shape[-1]}, expected {n_classes}"
        )

    return pred

