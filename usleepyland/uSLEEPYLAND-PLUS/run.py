#!/usr/bin/env python
"""
run.py

Extract sleep-stage predictions from EDF files using one or more models:

    usleep       – usleepy one-shot model + configurable temporal resolution
    deepresnet   – usleepy sliding-window model
    sleeptransformer  – usleepy sliding-window model (logits -> softmax)

Model families and weights layout
---------------------------------
The script can use different pretrained weight families, for example:

    - adult (default) models under:
          ./weights/default/<model_name>-nsrr-2024[_<lead_type>]

    - children models under:
          ./weights/children/<model_name>-nsrr-2024[_<lead_type>]

You select the family with:

    --model_set adult      # uses ./weights/default/...
    --model_set children   # uses ./weights/children/...

For each requested model and lead type, the script derives:

    base model dir:
        ./weights/<family>/<model_name>-nsrr-2024
    lead-specific dir:
        ./weights/<family>/<model_name>-nsrr-2024_<lead_type>

Main workflow
-------------
This script supports multi-model batch execution in a single call.

For each requested model:
    1. derive the model directory automatically as:
           ./weights/<family>/<model_name>-nsrr-2024
    2. derive lead-specific model directories as:
           ./weights/<family>/<model_name>-nsrr-2024_<lead_type>
    3. process all requested EDF files with that model
    4. save outputs under:
           ./<site_id>/<model_name>/...

After all requested models finish, the script can optionally run a final
ensemble stage:

    - it reads selected-model channel predictions for each EDF subject
    - it soft-averages matching channels across models, then channels per lead
    - it writes ensemble outputs under:
          ./<site_id>/ensemble/<pred_sec>s/_all/soft-voting/<BidsFolder>_PRED.npy
    - it writes one subject-level HDF5 containing models and ensemble results
    - it writes review-software files beside each exported annotation

Channel handling
----------------
All models are run in a single-channel setting:
    - each recognised EEG / EOG / chin channel is processed independently
    - per-lead-type majority voting is then applied across channels
    - the optional final ensemble combines across models and lead types

Preprocessing order
-------------------
Preprocessing uses the following order:

    1. optional notch filtering        (apply_notch_filtering)
    2. optional bandpass filtering     (apply_filtering)
    3. optional quality control        (clip_noisy_values – IQR-based clip)
    4. resampling to target_fs         (set_psg_sample_rate)
    5. optional scaling                (apply_scaling)

Notes
-----
- Internal lead type names are: eeg, eog, chin
- Only usleep supports sub-epoch temporal resolution (--pred_sec < 30)
- The current final ensemble stage expects 30-s majority predictions
  and therefore is supported when --pred_sec = 30
- Subject IDs / BidsFolder values are taken from the EDF filename stem

Usage examples
--------------
# Single EDF, one model (adult weights, default family)
python run.py \\
    --edf_file /data/subject.edf \\
    --site_id output \\
    --model_name usleep \\
    --lead_types "eeg,eog" \\
    --pred_sec 30 \\
    --n_classes 5 \\
    --model_set adult

# Directory scan, multiple models, final ensemble (adult weights)
python run.py \\
    --edf_dir /data/psg \\
    --site_id output \\
    --model_name "usleep,deepresnet,sleeptransformer" \\
    --lead_types "eeg,eog,chin" \\
    --pred_sec 30 \\
    --n_classes 5 \\
    --run_ensemble \\
    --model_set adult

# Directory scan with folder-name regex filter (children weights)
python run.py \\
    --edf_dir /data/psg \\
    --folder_regex ".*nsrr.*" \\
    --site_id output \\
    --model_name "deepresnet,sleeptransformer" \\
    --lead_types "eeg" \\
    --pred_sec 30 \\
    --n_classes 5 \\
    --model_set children

# usleep with sub-epoch resolution + preprocessing (adult weights)
python run.py \\
    --edf_file /data/subject.edf \\
    --site_id output \\
    --model_name usleep \\
    --lead_types "eeg" \\
    --pred_sec 1 \\
    --apply_bandpass --bandpass_low 0.3 --bandpass_high 35 \\
    --apply_notch --notch_freq 50 \\
    --scaling robust \\
    --model_set adult
"""
import argparse
import re
from pathlib import Path
import h5py
import logging
from typing import Union, List, Optional, Dict

import mne

import numpy as np

from usleepy.utils.annotations import export_for_review_software, npyprobs_to_annot
from usleepy.utils.ensemble import write_subject_hdf5
from usleepy.utils.header_extractors import (
    build_metadata_from_edf_path,
    prepare_channels,
    raw_to_physio_dicts,
)
from usleepy.utils.preprocessing import sig_preprocessing
from usleepy.utils.load_models import (
    LOGIT_MODELS,
    ONE_SHOT_MODELS,
    SUPPORTED_MODELS,
    SUPPORTS_REPRESENTATIONS,
    SUPPORTS_TEMPORAL_RESOLUTION,
    load_usleepy_hparams,
    load_usleepy_model,
    normalise_pred,
    run_usleep_one_shot,
    run_usleepy_sliding_window,
)

from collections import defaultdict


logger = logging.getLogger(__name__)

RUN_DIR = Path(__file__).resolve().parent


def _setup_logging(log_file: Optional[Path] = None) -> None:
    """Configure root logger. Called once from main()."""
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        root.addHandler(sh)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_file), mode="a")
        fh.setFormatter(fmt)
        root.addHandler(fh)
        logger.info("Logging to file: %s", log_file)


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------

def extract_from_sleep_model(
    physiological_data: dict,
    physiological_fs: dict,
    metadata: dict,
    lead_model_dirs: Dict[str, str],
    out_dir: Union[str, Path],
    selected_lead_types: Optional[List[str]] = None,
    pred_samples: Optional[int] = None,
    model_name: str = "usleep",
    n_classes: int = 5,
    preproc_config: Optional[dict] = None,
    weights_dir: Optional[Path] = None,
    edf_path: Optional[Path] = None,
    review_software: str = "both",
) -> str:
    """
    Unified extraction entry point for all supported models.

    Parameters
    ----------
    physiological_data  : {ch_name: 1D np.ndarray}
    physiological_fs    : {ch_name: sampling_rate}
    metadata            : {"BidsFolder": str, "SiteID": str}
    lead_model_dirs   : mapping of lead_type -> absolute path to usleepy project dir.
                          e.g. {"eeg": "/models/usleep-nsrr-2024_eeg",
                                "eog": "/models/usleep-nsrr-2024_eog",
                                "chin": "/models/usleep-nsrr-2024_emg"}
                          Only lead types present in this dict will be processed.
    out_dir             : root output directory. HDF5 files are written to
                          <out_dir>/<pred_sec>s/_<lead_type>/<bids_folder>.h5
    selected_lead_types : further restrict which lead types to run
                          (must be a subset of lead_model_dirs keys); None = all keys
    pred_samples        : samples per prediction; only used by usleep
    model_name          : one of 'usleep', 'deepresnet', 'sleeptransformer'
    n_classes           : 2, 3, 4, or 5
    preproc_config      : preprocessing settings dict (see parse_args for keys)
    weights_dir         : optional path to a directory containing model weights.
                          Expected structure: <weights_dir>/<lead_type>/model_weights.h5
                          If None, usleepy resolves weights from inside each model_dir.

    Returns
    -------
    str  Path to the written HDF5 file
    """
    model_name = model_name.lower()
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"model_name must be one of {sorted(SUPPORTED_MODELS)}, got {model_name!r}")

    if preproc_config is None:
        preproc_config = {
            "apply_bandpass":           False,
            "bandpass_low":             None,
            "bandpass_high":            None,
            "apply_notch":              False,
            "notch_freq":               50.0,
            "apply_quality_control":    True,
            "apply_channel_qc":         False,
            "max_bad_epoch_fraction":   0.2,
            "min_max_times_global_iqr": 20.0,
            "warn_fraction":            0.15,
            "scaling":                  "robust",
        }

    bids_folder = metadata["BidsFolder"]
    site_id     = metadata["SiteID"]

    processed_channels, processed_fs, channel_lead_type = prepare_channels(
        physiological_data, physiological_fs
    )

    lead_type_to_channels: dict[str, list] = {}
    for ch, lt in channel_lead_type.items():
        lead_type_to_channels.setdefault(lt, []).append(ch)

    logger.info("EDF=%s | SiteID=%s | model=%s", bids_folder, site_id, model_name)
    logger.info("Recognised %d usable channels", len(channel_lead_type))
    for lt in ["eeg", "eog", "chin"]:
        logger.info("Lead type '%s': %s", lt, sorted(lead_type_to_channels.get(lt, [])))

    all_lead_types = ["eeg", "eog", "chin"]
    if selected_lead_types is None:
        lead_types = all_lead_types
    else:
        lead_types = [lt.lower() for lt in selected_lead_types if lt.lower() in all_lead_types]

    epoch_sec     = 30
    target_fs     = 128
    epoch_samples = epoch_sec * target_fs   # 3840

    if model_name not in SUPPORTS_TEMPORAL_RESOLUTION:
        if pred_samples is not None and pred_samples != epoch_samples:
            logger.warning(
                "model '%s' does not support sub-epoch temporal resolution; "
                "ignoring pred_samples=%d and using 30-s resolution.",
                model_name, pred_samples,
            )
        pred_samples = epoch_samples

    if pred_samples is None or pred_samples <= 0:
        pred_samples = epoch_samples

    if epoch_samples % pred_samples != 0:
        raise ValueError(
            f"pred_samples={pred_samples} must evenly divide {epoch_samples} "
            f"(30 s × {target_fs} Hz)."
        )

    if n_classes not in (2, 3, 4, 5):
        raise ValueError(f"n_classes must be 2, 3, 4, or 5. Got {n_classes}.")

    preds_per_epoch = epoch_samples // pred_samples
    pred_sec        = int(pred_samples / target_fs)

    all_representations: dict[str, dict] = {lt: {} for lt in lead_types}
    all_predictions:     dict[str, dict] = {lt: {} for lt in lead_types}
    majority_sums:       dict            = {lt: None for lt in lead_types}
    majority_counts:     dict            = {lt: None for lt in lead_types}
    channel_bad_epochs:  dict            = {lt: {} for lt in lead_types}
    excluded_channels:   dict            = {lt: [] for lt in lead_types}

    for lead_type in lead_types:
        if lead_type not in lead_model_dirs:
            logger.info("No model_dir supplied for lead_type='%s' – skipping.", lead_type)
            continue

        model_dir = str(Path(lead_model_dirs[lead_type]).resolve())
        logger.info("model=%s | lead_type=%s | model_dir=%s", model_name, lead_type, model_dir)

        try:
            hparams = load_usleepy_hparams(model_dir)
        except Exception as exc:
            logger.warning("Could not load hparams for %s/%s: %s – skipping.", model_name, lead_type, exc)
            continue

        if "build" not in hparams:
            hparams["build"] = {}

        if model_name in SUPPORTS_TEMPORAL_RESOLUTION:
            hparams["build"]["data_per_prediction"] = pred_samples

        one_shot = model_name in ONE_SHOT_MODELS
        is_logits = model_name in LOGIT_MODELS
        usleepy_model = None

        for ch_name, lt in channel_lead_type.items():
            if lt != lead_type:
                continue

            sig = processed_channels[ch_name]
            fs  = processed_fs[ch_name]
            if sig is None or len(sig) <= 1:
                continue

            sig_proc, n_periods, data_per_period, noisy_idx = sig_preprocessing(
                sig=sig,
                fs=fs,
                epoch_sec=epoch_sec,
                target_fs=target_fs,
                # step 2
                # The UI shares one EEG/EOG filter range; EMG needs its own passband.
                apply_bandpass=preproc_config["apply_bandpass"] and lead_type != "chin",
                bandpass_low=preproc_config["bandpass_low"],
                bandpass_high=preproc_config["bandpass_high"],
                # step 1
                apply_notch=preproc_config["apply_notch"] and lead_type != "chin",
                notch_freq=preproc_config["notch_freq"],
                # step 3
                apply_quality_control=preproc_config["apply_quality_control"],
                detect_bad_epochs=preproc_config["apply_channel_qc"] and lead_type == "eeg",
                min_max_times_global_iqr=preproc_config["min_max_times_global_iqr"],
                warn_fraction=preproc_config["warn_fraction"],
                channel_id=ch_name,
                # step 5
                scaling=preproc_config["scaling"],
            )

            if n_periods == 0:
                logger.warning("ch=%s | n_periods=0 after preprocessing – skipping.", ch_name)
                continue
            if preproc_config["apply_channel_qc"] and lead_type == "eeg":
                channel_bad_epochs[lead_type][ch_name] = noisy_idx
                if len(noisy_idx) / n_periods > preproc_config["max_bad_epoch_fraction"]:
                    excluded_channels[lead_type].append(ch_name)
                    logger.warning("ch=%s | excluded: %d/%d bad epochs exceed %.1f%%",
                                   ch_name, len(noisy_idx), n_periods,
                                   100 * preproc_config["max_bad_epoch_fraction"])
                    continue

            logger.info(
                "ch=%s | lead=%s | n_periods=%d | data_per_period=%d",
                ch_name, lead_type, n_periods, data_per_period,
            )

            if usleepy_model is None:
                try:
                    usleepy_model = load_usleepy_model(
                        model_dir=model_dir,
                        hparams=hparams,
                        one_shot=one_shot,
                        n_periods=n_periods,
                        weights_dir=weights_dir,
                        lead_type=lead_type,
                    )
                except Exception as exc:
                    logger.warning("Could not load model for %s/%s: %s – skipping.", model_name, lead_type, exc)
                    break

            repr_arr = None
            if one_shot:
                pred_raw, repr_arr = run_usleep_one_shot(
                    sig_proc, n_periods, data_per_period, usleepy_model
                )
            else:
                pred_raw = run_usleepy_sliding_window(
                    sig_proc, n_periods, data_per_period, usleepy_model, hparams, is_logits
                )

            pred_arr = normalise_pred(pred_raw, n_periods, preds_per_epoch, n_classes, ch_name)

            if repr_arr is not None:
                all_representations[lead_type][ch_name] = repr_arr
            all_predictions[lead_type][ch_name] = pred_arr

            valid = np.ones(pred_arr.shape[0], dtype=bool)
            if preproc_config["apply_channel_qc"] and lead_type == "eeg":
                valid[noisy_idx] = False
            weighted = pred_arr * valid[:, None, None]

            if majority_sums[lead_type] is None:
                majority_sums[lead_type] = weighted.copy()
                majority_counts[lead_type] = valid.astype(np.int32)
            else:
                common = min(len(majority_sums[lead_type]), len(weighted))
                majority_sums[lead_type] = majority_sums[lead_type][:common] + weighted[:common]
                majority_counts[lead_type] = majority_counts[lead_type][:common] + valid[:common]

    for lt in lead_types:
        logger.info(
            "Lead '%s' -> predictions: %d channel(s): %s",
            lt, len(all_predictions[lt]), sorted(all_predictions[lt].keys()),
        )
        if model_name in SUPPORTS_REPRESENTATIONS:
            logger.info(
                "Lead '%s' -> representations: %d channel(s): %s",
                lt, len(all_representations[lt]), sorted(all_representations[lt].keys()),
            )

    majority_probs:  dict[str, np.ndarray] = {}
    majority_labels: dict[str, np.ndarray] = {}
    for lt in lead_types:
        if majority_sums[lt] is None:
            continue
        counts = majority_counts[lt]
        avg = majority_sums[lt] / np.maximum(counts[:, None, None], 1)
        avg[counts == 0] = 1.0 / n_classes
        majority_probs[lt]  = avg
        majority_labels[lt] = np.argmax(avg, axis=-1)

    resolved_out_dir = Path(out_dir).resolve() / f"{pred_sec}s"
    if resolved_out_dir.exists() and not resolved_out_dir.is_dir():
        resolved_out_dir.unlink()  # remove conflicting file if any
    resolved_out_dir.mkdir(parents=True, exist_ok=True)

    # Save per-channel and majority predictions as .npy in subfolders ──
    # Structure:
    #   <out_dir>/<pred_sec>s/<lead_type>/<channel_name>/<bids_folder>_PRED.npy
    #   <out_dir>/<pred_sec>s/<lead_type>/majority/<bids_folder>_PRED.npy
    for lt in lead_types:
        preds = all_predictions.get(lt, {})
        if not preds:
            continue

        # Per-channel predictions
        for cand, pred_arr in preds.items():
            safe_cand = str(cand).replace("/", "-").replace(" ", "")
            chan_dir = resolved_out_dir / f"_{lt}" / safe_cand
            chan_dir.mkdir(parents=True, exist_ok=True)

            npy_path = chan_dir / f"{bids_folder}_PRED.npy"
            # Save as npy; you can add more keys if needed
            if pred_sec == 30:
                np.save(npy_path, pred_arr.mean(axis=1)) # [n_epochs, n_classes]
            else:
                np.save(npy_path, pred_arr)
            logger.info("Saved channel prediction .npy: %s", npy_path)

            # --- NEW:write .annot from majority over sub-epochs ---
            # pred_arr has shape [n_epochs, n_preds_per_epoch, n_classes]
            # collapse sub-epochs by averaging across axis=1
            probs_epoch = pred_arr.mean(axis=1)  # [n_epochs, n_classes]
            annot_path = chan_dir / f"{bids_folder}_PRED.annot"
            try:
                npyprobs_to_annot(
                    probs=probs_epoch,
                    out_path=annot_path,
                    channel=safe_cand,
                    fmt="standard",  # or "model"
                    time_encoding="elapsed_seconds",
                    start_time="00:00:00",
                    model=model_name,
                )
                logger.info("Saved channel annotation .annot: %s", annot_path)
                # Software import files from this .annot (if EDF path is available)
                if edf_path is not None:
                    export_for_review_software(annot_path, edf_path, chan_dir, review_software)

            except Exception as exc:
                logger.warning("Failed to write .annot/SOMNO for %s: %s", npy_path, exc)

        # Majority-vote probabilities per lead_type
        if lt in majority_probs:
            mj_dir = resolved_out_dir / f"_{lt}" / "majority"
            mj_dir.mkdir(parents=True, exist_ok=True)

            mj_path = mj_dir / f"{bids_folder}_PRED.npy"
            if pred_sec == 30:
                np.save(mj_path, majority_probs[lt].mean(axis=1)) # [n_epochs, n_classes]
            else:
                np.save(mj_path, majority_probs[lt])
            np.save(mj_dir / f"{bids_folder}_VALID.npy", majority_counts[lt] > 0)
            logger.info("Saved majority-vote prediction .npy: %s", mj_path)

            # --- majority .annot ---
            probs_epoch_mj = majority_probs[lt].mean(axis=1) # [n_epochs, n_classes]
            annot_mj_path = mj_dir / f"{bids_folder}_PRED.annot"
            try:
                npyprobs_to_annot(
                    probs=probs_epoch_mj,
                    out_path=annot_mj_path,
                    channel="majority",
                    fmt="standard",
                    time_encoding="elapsed_seconds",
                    start_time="00:00:00",
                    model=model_name,
                )
                logger.info("Saved majority annotation .annot: %s", annot_mj_path)
                if edf_path is not None:
                    export_for_review_software(annot_mj_path, edf_path, mj_dir, review_software)

            except Exception as exc:
                logger.warning("Failed to write majority .annot/SOMNO for %s: %s", mj_path, exc)

    hdf5_path = resolved_out_dir / f"{bids_folder}.h5"

    with h5py.File(hdf5_path, "w") as h5f:
        # ── global metadata ───────────────────────────────────────────────
        h5f.attrs["BidsFolder"]  = bids_folder
        h5f.attrs["SiteID"]      = site_id
        h5f.attrs["ModelName"]   = model_name
        h5f.attrs["EpochSeconds"] = epoch_sec
        h5f.attrs["TargetFS"]    = target_fs
        h5f.attrs["PredSamples"] = pred_samples
        h5f.attrs["PredSeconds"] = float(pred_samples) / float(target_fs)
        h5f.attrs["NumClasses"]  = int(n_classes)

        # ── preprocessing metadata ────────────────────────────────────────
        h5f.attrs["ApplyBandpass"]  = bool(preproc_config["apply_bandpass"])
        h5f.attrs["BandpassLow"]    = -1.0 if preproc_config["bandpass_low"]  is None else float(preproc_config["bandpass_low"])
        h5f.attrs["BandpassHigh"]   = -1.0 if preproc_config["bandpass_high"] is None else float(preproc_config["bandpass_high"])
        h5f.attrs["ApplyNotch"]     = bool(preproc_config["apply_notch"])
        h5f.attrs["NotchFreq"]      = float(preproc_config["notch_freq"])
        h5f.attrs["ApplyQC"]        = bool(preproc_config["apply_quality_control"])
        h5f.attrs["ApplyChannelQC"] = bool(preproc_config["apply_channel_qc"])
        h5f.attrs["MaxBadEpochFraction"] = float(preproc_config["max_bad_epoch_fraction"])
        h5f.attrs["IQRMultiplier"]  = float(preproc_config["min_max_times_global_iqr"])
        h5f.attrs["ScalingMode"]    = str(preproc_config["scaling"])

        for lt in lead_types:
            preds = all_predictions.get(lt, {})
            reprs = all_representations.get(lt, {})
            if not preds and not reprs and not excluded_channels[lt]:
                continue

            grp = h5f.create_group(lt)
            grp.attrs["excluded_channels"] = ",".join(excluded_channels[lt])
            for cand, indices in channel_bad_epochs[lt].items():
                grp.create_dataset(f"{cand}_bad_epoch_indices", data=np.asarray(indices, dtype=np.int32))
            if majority_counts[lt] is not None:
                grp.create_dataset("valid_channel_counts", data=majority_counts[lt])

            for cand, pred_arr in preds.items():
                ds = grp.create_dataset(f"{cand}_pred", data=pred_arr)
                ds.attrs["lead_type"]    = lt
                ds.attrs["channel_name"] = cand
                ds.attrs["n_classes"]    = int(n_classes)
                ds.attrs["description"]  = (
                    "Per-epoch × per-sub-epoch class probabilities. "
                    "Shape [n_epochs, n_preds_per_epoch, n_classes]. "
                    "n_preds_per_epoch = EpochSeconds * TargetFS / PredSamples."
                )

            for cand, repr_arr in reprs.items():
                ds = grp.create_dataset(f"{cand}_repr", data=repr_arr)
                ds.attrs["lead_type"]    = lt
                ds.attrs["channel_name"] = cand
                ds.attrs["description"]  = "Intermediate layer features (average_pool)."

            if lt in majority_probs:
                mv = grp.create_group("majority_vote")
                ds = mv.create_dataset("probs", data=majority_probs[lt])
                ds.attrs["n_classes"]   = int(n_classes)
                ds.attrs["description"] = (
                    "Class probabilities averaged across all channels of this lead_type. "
                    "Shape [n_epochs, n_preds_per_epoch, n_classes]."
                )
                ds = mv.create_dataset("labels", data=majority_labels[lt])
                ds.attrs["n_classes"]   = int(n_classes)
                ds.attrs["description"] = (
                    "Argmax labels from majority-vote probs. "
                    "Shape [n_epochs, n_preds_per_epoch]."
                )

        def _log_h5(name, obj):
            if isinstance(obj, h5py.Group):
                logger.info("H5 GROUP   : /%s", name)
            elif isinstance(obj, h5py.Dataset):
                logger.info("H5 DATASET : /%s | shape=%s | dtype=%s", name, obj.shape, obj.dtype)

        logger.info("HDF5 written: %s", hdf5_path)
        logger.info("HDF5 attrs  : %s", dict(h5f.attrs))
        h5f.visititems(_log_h5)

    print(f"Saved to {hdf5_path}")
    return str(hdf5_path)


# ---------------------------------------------------------------------------
# Single-EDF wrapper
# ---------------------------------------------------------------------------

def process_single_edf(
    edf_path: Path,
    site_id: str,
    lead_model_dirs: Dict[str, str],
    out_dir: Union[str, Path],
    selected_lead_types: Optional[List[str]] = None,
    pred_samples: Optional[int] = None,
    model_name: str = "usleep",
    n_classes: int = 5,
    preproc_config: Optional[dict] = None,
    weights_dir: Optional[Path] = None,
    review_software: str = "both",
) -> str:
    print(f"\n=== Processing EDF: {edf_path} ===")
    raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose="ERROR")
    physio, fs = raw_to_physio_dicts(raw)
    meta = build_metadata_from_edf_path(
        edf_path, site_id=site_id
    )

    hdf5_path = extract_from_sleep_model(
        physiological_data=physio,
        physiological_fs=fs,
        metadata=meta,
        lead_model_dirs=lead_model_dirs,
        out_dir=out_dir,
        selected_lead_types=selected_lead_types,
        pred_samples=pred_samples,
        model_name=model_name,
        n_classes=n_classes,
        preproc_config=preproc_config,
        weights_dir=weights_dir,
        edf_path=edf_path,
        review_software=review_software,
    )
    print(f"  -> {hdf5_path}")
    return hdf5_path


# ---------------------------------------------------------------------------
# Path helper (WSL / Windows cross-platform)
# ---------------------------------------------------------------------------

def _safe_path(raw: str) -> Path:
    """Resolve a path that may be a Windows-style string passed into WSL."""
    p = raw.strip().strip('"').strip("'")
    import re as _re
    win = _re.match(r'^([A-Za-z]):[/\\](.+)$', p)
    if win:
        drive = win.group(1).lower()
        rest  = win.group(2).replace('\\', '/')
        return Path(f'/mnt/{drive}/{rest}')
    path = Path(p)
    return path if path.is_absolute() else Path.cwd() / path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract sleep-stage predictions from EDF files.\n\n"
            "All paths are explicit – the script has no dependency on its own "
            "directory.\n\n"
            "Supported models\n"
            "----------------\n"
            "  usleep       One-shot. Supports --pred_sec. Saves predictions + reprs.\n"
            "  deepresnet   Sliding-window. 30-s resolution. Predictions only.\n"
            "  sleeptransformer  Sliding-window + softmax. 30-s resolution. Predictions only.\n\n"
            "Preprocessing order\n"
            "-------------------\n"
            "  1. notch filter      --apply_notch\n"
            "  2. bandpass filter   --apply_bandpass\n"
            "  3. IQR clip (QC)     --apply_quality_control  [default: ON]\n"
            "  4. resample          always applied when fs != 128 Hz\n"
            "  5. scaling           --scaling\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── EDF source (mutually exclusive) ──────────────────────────────────────
    src_grp = parser.add_mutually_exclusive_group(required=True)
    src_grp.add_argument(
        "--edf_file", type=str,
        help="Absolute path to a single EDF file.",
    )
    src_grp.add_argument(
        "--edf_dir", type=str,
        help="Absolute path to a directory; all EDF files are processed recursively.",
    )

    # ── Path arguments ────────────────────────────────────────────────────────
    parser.add_argument(
        "--log_file", type=str, default=None,
        help=(
            "Path to the log file. "
            "Default: <out_dir>/sleep_model_extraction.log"
        ),
    )
    parser.add_argument(
        "--weights_dir", type=str, default=None,
        help=(
            "Optional path to a directory containing pre-trained model weights. "
            "Expected structure: <weights_dir>/<lead_type>/model_weights.h5 "
            "(e.g. /weights/eeg/model_weights.h5). "
            "If not provided, usleepy resolves weights from inside --model_dir."
        ),
    )

    # Root directory where <site_id> subfolder will be created
    parser.add_argument(
        "--output_root",
        type=str,
        default=".",
        help=(
            "Root directory for all outputs under this site_id. "
            "Default is current working directory '.', so outputs go to ./<site_id>. "
            "If set, outputs go to <output_root>/<site_id>."
        ),
    )

    # ── Scan options ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--pattern", type=str, default="*.edf",
        help="Glob pattern for EDF files within --edf_dir (default: *.edf).",
    )
    parser.add_argument(
        "--folder_regex", type=str, default=None,
        help=(
            "Optional regex applied to each EDF file's parent folder name. "
            "Only EDFs whose parent directory name matches the pattern are processed. "
            "Example: --folder_regex '.*nsrr.*'"
        ),
    )

    # ── Model / run options ───────────────────────────────────────────────────
    parser.add_argument(
        "--site_id", type=str, default="SITE001",
        help="Site identifier stored in metadata and used in the output folder (default: SITE001).",
    )
    parser.add_argument(
        "--model_name", type=str, default="usleep",
        help=(
            "Comma-separated models to run: usleep, deepresnet, sleeptransformer "
            "(default: usleep). Example: --model_name 'usleep,deepresnet'"
        ),
    )
    parser.add_argument(
        "--lead_types", type=str, default="all",
        help=(
            "Comma-separated lead types to process: eeg, eog, chin, or 'all' (default: all). "
            "Only lead types present in --model_dir are actually run."
        ),
    )
    parser.add_argument(
        "--pred_sec", type=int, default=30,
        help=(
            "Temporal resolution in seconds (default 30). "
            "Only applies to usleep. Valid divisors of 30: 1, 2, 3, 5, 6, 10, 15, 30."
        ),
    )
    parser.add_argument(
        "--n_classes", type=int, default=5, choices=[2, 3, 4, 5],
        help="Number of output sleep-stage classes (default 5).",
    )

    # ── Preprocessing ─────────────────────────────────────────────────────────
    parser.add_argument("--apply_bandpass",  action="store_true", help="Apply bandpass filter (step 2).")
    parser.add_argument("--bandpass_low",    type=float, default=None, help="Bandpass low-cut frequency (Hz).")
    parser.add_argument("--bandpass_high",   type=float, default=None, help="Bandpass high-cut frequency (Hz).")

    parser.add_argument("--apply_notch",     action="store_true", help="Apply notch filter (step 1).")
    parser.add_argument("--notch_freq",      type=float, default=50.0, help="Notch frequency in Hz (default 50.0).")

    parser.add_argument(
        "--apply_quality_control", dest="apply_quality_control",
        action="store_true", default=True,
        help="Apply IQR-based clip QC – clip_noisy_values (step 3, default: ON).",
    )
    parser.add_argument(
        "--no_quality_control", dest="apply_quality_control", action="store_false",
        help="Disable IQR-based clip QC.",
    )
    parser.add_argument("--apply_channel_qc", action="store_true",
                        help="Exclude channels over the bad-epoch threshold and mask bad epochs in channel averaging.")
    parser.add_argument("--max_bad_epoch_fraction", type=float, default=0.2,
                        help="Exclude a channel when its fraction of IQR flagged epochs exceeds this value (0-1).")
    parser.add_argument(
        "--min_max_times_global_iqr", type=float, default=20.0,
        help="IQR multiplier for clip_noisy_values threshold (default 20.0).",
    )
    parser.add_argument(
        "--warn_fraction", type=float, default=0.15,
        help="Fraction of noisy epochs that triggers a WARNING log (default 0.15).",
    )
    parser.add_argument(
        "--scaling", type=str, default="robust", choices=["none", "robust", "usleepy"],
        help="Scaling mode applied after resampling (default: robust).",
    )

    parser.add_argument(
        "--run_ensemble",
        action="store_true",
        help=(
            "After all requested models finish for all processed EDFs, build an "
            "ensemble for each EDF using --ensemble_method."
        ),
    )
    parser.add_argument(
        "--ensemble_method",
        type=str,
        default="soft_vote",
        choices=["soft_vote"],
        help="Final ensemble method: average majority predictions.",
    )
    parser.add_argument(
        "--model_set",
        type=str,
        default="adult",
        choices=["adult", "children"],
        help=(
            "Select which pretrained model family to use. "
            "'adult' uses ./weights/adult/, "
            "'children' uses ./weights/children/ "
            "(default: adult)."
        ),
    )
    parser.add_argument(
        "--review_software",
        choices=["somnomedics", "compumedics", "both", "none"],
        default="both",
        help="Software import format, or none to skip import files (default: both).",
    )

    args = parser.parse_args()

    requested_models = [x.strip().lower() for x in args.model_name.split(",") if x.strip()]
    invalid_models = [m for m in requested_models if m not in SUPPORTED_MODELS]
    if invalid_models:
        parser.error(
            f"Unsupported model(s): {invalid_models}. "
            f"Valid models: {sorted(SUPPORTED_MODELS)}"
        )
    args.model_names = requested_models

    # ── Validation ────────────────────────────────────────────────────────────
    if args.pred_sec not in (1, 2, 3, 5, 6, 10, 15, 30):
        parser.error("--pred_sec must evenly divide 30. Valid: 1, 2, 3, 5, 6, 10, 15, 30.")

    if args.apply_bandpass and args.bandpass_low is None and args.bandpass_high is None:
        parser.error("--apply_bandpass requires at least one of --bandpass_low or --bandpass_high.")

    return args


def main():
    # ------------------------------------------------------------------
    # Parse CLI arguments once at startup.
    #
    # At this point parse_args() is expected to already:
    # - validate supported models
    # - expose args.model_names as a list, e.g. ["usleep", "deepresnet"]
    # - validate pred_sec and preprocessing constraints
    # ------------------------------------------------------------------
    args = parse_args()
    if not 0 <= args.max_bad_epoch_fraction <= 1:
        raise ValueError("max_bad_epoch_fraction must be between 0 and 1")

    # ------------------------------------------------------------------
    # Resolve requested lead types from the CLI.
    #
    # Supported internal lead types in this script are:
    #   - "eeg"
    #   - "eog"
    #   - "chin"
    #
    # If the user passes --lead_types all, we keep None here and later
    # expand it to all supported lead types.
    #
    # Examples:
    #   --lead_types "all"        -> selected_lead_types = None
    #   --lead_types "eeg,eog"    -> ["eeg", "eog"]
    # ------------------------------------------------------------------
    selected_lead_types = (
        None
        if args.lead_types.lower() == "all"
        else [x.strip().lower() for x in args.lead_types.split(",") if x.strip()]
    )

    # ------------------------------------------------------------------
    # Requested models to run, already parsed from the comma-separated
    # CLI string by parse_args().
    #
    # Example:
    #   --model_names "usleep,deepresnet"
    # becomes:
    #   ["usleep", "deepresnet"]
    # ------------------------------------------------------------------
    model_names: List[str] = args.model_names

    # ------------------------------------------------------------------
    # Convert the requested temporal resolution in seconds into samples
    # at the model target sample rate (128 Hz).
    #
    # Example:
    #   pred_sec = 30  -> pred_samples = 3840
    #   pred_sec =  1  -> pred_samples = 128
    #
    # Note: only usleep actually supports sub-epoch resolution; the lower
    # level code will force 30 s for models that do not support it.
    # ------------------------------------------------------------------
    pred_samples = int(args.pred_sec * 128)

    # ------------------------------------------------------------------
    # Collect all preprocessing options into one dict so they can be
    # passed unchanged into process_single_edf() and then down to the
    # actual extraction pipeline.
    #
    # This keeps main() responsible only for orchestration, while the
    # lower-level functions handle the actual signal processing.
    # ------------------------------------------------------------------
    preproc_config = {
        "apply_bandpass":           args.apply_bandpass,
        "bandpass_low":             args.bandpass_low,
        "bandpass_high":            args.bandpass_high,
        "apply_notch":              args.apply_notch,
        "notch_freq":               args.notch_freq,
        "apply_quality_control":    args.apply_quality_control,
        "apply_channel_qc":         args.apply_channel_qc,
        "max_bad_epoch_fraction":   args.max_bad_epoch_fraction,
        "min_max_times_global_iqr": args.min_max_times_global_iqr,
        "warn_fraction":            args.warn_fraction,
        "scaling":                  args.scaling,
    }

    # ------------------------------------------------------------------
    # Resolve the final list of lead types that this run will actually
    # attempt to process.
    #
    # If no explicit list was requested, default to all supported lead
    # types used internally by the pipeline.
    # ------------------------------------------------------------------
    all_lead_types = ["eeg", "eog", "chin"]
    active_leads = selected_lead_types if selected_lead_types else all_lead_types

    # ------------------------------------------------------------------
    # Resolve the root output folder for the whole run.
    #
    # New behaviour:
    #   - If --output_root is not provided (default "."), we keep the
    #     original behaviour: outputs under ./<site_id>.
    #   - If --output_root is provided, outputs go to
    #     <output_root>/<site_id>.
    #
    # Structure:
    #   <output_root>/<site_id>/
    #       usleep/
    #       deepresnet/
    #       sleeptransformer/
    #       ensemble/
    #
    # We ensure this exists and is a directory. If a regular file with
    # that same name exists, remove it so we can create the folder.
    # ------------------------------------------------------------------
    # Base root for all sites
    output_root = _safe_path(args.output_root)
    # Final site-level root (where all model/ensemble subfolders live)
    site_root = output_root / f"{args.site_id}"
    if site_root.exists() and not site_root.is_dir():
        site_root.unlink()
    site_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Configure logging.
    #
    # If the user provided --log_file, use it.
    # Otherwise, write a default log file under the site root:
    #   <output_root>/<site_id>/sleep_model_extraction.log
    # ------------------------------------------------------------------
    log_file = _safe_path(args.log_file) if args.log_file else site_root / "sleep_model_extraction.log"
    _setup_logging(log_file)

    # ------------------------------------------------------------------
    # Log the high-level run configuration for traceability.
    # ------------------------------------------------------------------
    logger.info("site_id      : %s", args.site_id)
    logger.info("site_root    : %s", site_root)
    logger.info("models       : %s", model_names)
    logger.info("lead_types   : %s", active_leads)
    logger.info("pred_sec     : %d", args.pred_sec)
    logger.info("ensemble_method: %s", args.ensemble_method)
    logger.info("review_software: %s", args.review_software)

    # ------------------------------------------------------------------
    # Optional external weights directory.
    #
    # When provided, lower-level model-loading code will look for weights
    # under:
    #   <weights_dir>/<lead_type>/model_weights.h5
    #
    # If not provided, each model resolves weights from inside its own
    # model directory.
    # ------------------------------------------------------------------
    weights_dir = _safe_path(args.weights_dir) if args.weights_dir else None
    if weights_dir is not None:
        logger.info("weights_dir  : %s", weights_dir)

    # ------------------------------------------------------------------
    # Optional folder-name regex used only in directory mode.
    #
    # The regex is applied to the parent folder name of each EDF file,
    # allowing the user to keep only EDFs whose parent directory matches
    # a pattern.
    # ------------------------------------------------------------------
    folder_regex = re.compile(args.folder_regex) if args.folder_regex else None

    # --------------------------------------------------------------
    # Resolve the EDF file list once, before any model is run.
    #
    # The script is model-first:
    #   1. determine the EDF list once
    #   2. for each model, iterate over all EDFs
    #
    # This avoids recomputing the EDF scan separately per model.
    # --------------------------------------------------------------
    if args.edf_file is not None:
        # ----------------------------------------------------------
        # Single-file mode:
        # normalize the given path and wrap it as a one-element list
        # so the rest of main() can always iterate over edf_files.
        # ----------------------------------------------------------
        edf_files = [_safe_path(args.edf_file)]
        if not edf_files[0].exists():
            raise FileNotFoundError(f"EDF not found: {edf_files[0]}")
    else:
        # ----------------------------------------------------------
        # Directory mode:
        # recursively scan for EDF files matching the user pattern.
        # ----------------------------------------------------------
        edf_dir = _safe_path(args.edf_dir)
        if not edf_dir.is_dir():
            raise NotADirectoryError(f"Not a directory: {edf_dir}")

        # Recursively gather matching EDF files.
        edf_files = sorted(edf_dir.rglob(args.pattern))

        # Optionally filter EDFs by parent folder name regex.
        if folder_regex is not None:
            edf_files = [f for f in edf_files if folder_regex.search(f.parent.name)]
            logger.info(
                "folder_regex=%r filtered to %d EDF file(s)",
                args.folder_regex, len(edf_files)
            )

        # If nothing matched, log and exit cleanly.
        if not edf_files:
            logger.warning(
                "No EDF files found in %s matching pattern='%s'%s",
                edf_dir, args.pattern,
                f" and folder_regex='{args.folder_regex}'" if folder_regex else "",
            )
            return

    # Log how many EDFs will be processed in total.
    logger.info("Found %d EDF file(s) to process", len(edf_files))

    # --------------------------------------------------------------
    # Build a lookup table from subject ID to EDF path.
    #
    # In this pipeline, subject identity is derived from:
    #   BidsFolder = edf_path.stem
    #
    # This mapping is therefore the safest way to recover the original
    # EDF path later during the ensemble stage, especially because the
    # ensemble now also needs the EDF path to export SOMNO-formatted
    # outputs from the final ensemble annotation.
    #
    # Example:
    #   /data/sub-001.edf  -> {"sub-001": Path("/data/sub-001.edf")}
    # --------------------------------------------------------------
    edf_by_bids_folder = {
        edf_path.stem: edf_path
        for edf_path in edf_files
    }

    # --------------------------------------------------------------
    # Prediction stage
    #
    # Model-first execution:
    #   for each requested model:
    #       derive its output directory
    #       derive its lead-specific model directories
    #       process all EDF files
    #
    # This structure is preferable because model setup/loading is the
    # expensive axis, and grouping EDFs by model keeps the run more
    # coherent from a compute and bookkeeping perspective.
    # --------------------------------------------------------------
    successful_models: Dict[str, List[str]] = defaultdict(list)
    for model_name in model_names:
        # ----------------------------------------------------------
        # Per-model output root:
        #   <site_root>/<model_name>/
        #
        # Example:
        #   output_root=/app/output
        #   site_id=iis_batch1
        #   -> /app/output/iis_batch1/usleep/
        #
        # We ensure this path exists and is a directory.
        # ----------------------------------------------------------
        out_dir = site_root / model_name
        if out_dir.exists() and not out_dir.is_dir():
            out_dir.unlink()
        out_dir.mkdir(parents=True, exist_ok=True)

        # ----------------------------------------------------------
        # Derive the base directory for this model automatically.
        #
        # The user can select which pretrained model family to use:
        #   --model_set adult   -> ./weights/adult/
        #   --model_set children -> ./weights/children/
        #
        # Convention used:
        #   adult   : ./weights/adult/<model_name>-nsrr-2024
        #   children: ./weights/children/<model_name>-nsrr-2024
        #
        # Then for each requested lead type we derive:
        #   ./weights/<family>/<model_name>-nsrr-2024_<lead_type>
        #
        # Example:
        #   model_name   = "deepresnet"
        #   model_set    = "children"
        #   active_leads = ["eeg", "eog"]
        #
        # Gives:
        #   ./weights/children/deepresnet-nsrr-2024_eeg
        #   ./weights/children/deepresnet-nsrr-2024_eog
        # ----------------------------------------------------------
        model_family_dir = "adult" if args.model_set == "adult" else "children"
        base_model_dir = _safe_path(f"./weights/{model_family_dir}/{model_name}-nsrr-2024")

        lead_model_dirs = {
            # Predictions use the "chin" lead name; bundled weight folders use "emg".
            lt: str(f"{base_model_dir}_{'emg' if lt == 'chin' else lt}")
            for lt in active_leads
        }

        logger.info("model_set    : %s", args.model_set)

        # Log the current model batch configuration.
        logger.info(
            "Starting model batch | model=%s | model_set=%s | out_dir=%s | base_model_dir=%s",
            model_name, args.model_set, out_dir, base_model_dir
        )

        # ----------------------------------------------------------
        # Build a reusable kwargs dict passed to process_single_edf()
        # for every EDF in this model batch.
        #
        # This avoids rebuilding the same parameter bundle repeatedly
        # inside the EDF loop.
        # ----------------------------------------------------------
        common_kwargs = dict(
            site_id=args.site_id,
            lead_model_dirs=lead_model_dirs,
            out_dir=out_dir,
            selected_lead_types=selected_lead_types,
            pred_samples=pred_samples,
            model_name=model_name,
            n_classes=args.n_classes,
            preproc_config=preproc_config,
            weights_dir=weights_dir,
            review_software=args.review_software,
        )

        # ----------------------------------------------------------
        # Process every EDF with the current model.
        #
        # Each EDF is isolated in a try/except block so a failure on one
        # file does not stop the full batch for that model.
        # ----------------------------------------------------------
        for edf_path in edf_files:
            try:
                process_single_edf(edf_path=edf_path, **common_kwargs)
                successful_models[edf_path.stem].append(model_name)
            except Exception as e:
                logger.exception(
                    "Failed to process %s with model=%s: %s",
                    edf_path.name, model_name, e
                )

    # Consolidate only EDFs from this invocation. The model HDF5 files are
    # temporary inputs; each subject leaves with one HDF5 at the site root.
    for bids_folder, edf_path in edf_by_bids_folder.items():
        write_subject_hdf5(
            site_root=site_root,
            bids_folder=bids_folder,
            model_names=successful_models[bids_folder],
            lead_types=active_leads,
            pred_sec=args.pred_sec,
            n_classes=args.n_classes,
            run_ensemble=args.run_ensemble,
            edf_path=edf_path,
            review_software=args.review_software,
        )

if __name__ == "__main__":
    main()
