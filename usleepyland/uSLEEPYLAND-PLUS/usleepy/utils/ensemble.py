"""Combine per-channel and per-model predictions and write per-subject outputs."""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np

from usleepy.utils.annotations import export_for_review_software, npyprobs_to_annot

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ensemble helpers
# ---------------------------------------------------------------------------




def write_ensemble_prediction(
    probabilities: np.ndarray, directory: Path, bids_folder: str,
    channel: str, edf_path: Path, review_software: str, model: str = "ensemble",
) -> Path:
    """Expose one HDF5 prediction as the usual NumPy and review files."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{bids_folder}_PRED.npy"
    np.save(path, probabilities)
    annot_path = directory / f"{bids_folder}_PRED.annot"
    npyprobs_to_annot(
        probs=probabilities.mean(axis=1) if probabilities.ndim == 3 else probabilities,
        out_path=annot_path, channel=channel,
        fmt="standard", time_encoding="elapsed_seconds",
        start_time="00:00:00", model=model,
    )
    try:
        export_for_review_software(annot_path, edf_path, directory, review_software)
    except Exception as exc:
        logger.warning("Failed review-software export for %s: %s", annot_path, exc)
    return path


def vote_probability_arrays(inputs: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    """Average valid contributors per epoch; mark epochs with no contributor."""
    lengths = {array.shape for array, _ in inputs}
    if len(lengths) != 1:
        raise ValueError(f"Ensemble inputs have incompatible shapes: {lengths}")
    arrays = np.stack([array for array, _ in inputs])
    masks = []
    for array, mask in inputs:
        mask = np.asarray(mask, dtype=bool)
        # Model validity is stored per epoch, while predictions can also
        # have a sub-epoch axis (for example [epochs, 1, classes]).
        if array.ndim == 3 and mask.shape == (array.shape[0],):
            mask = np.broadcast_to(mask[:, None], array.shape[:-1])
        if mask.shape != array.shape[:-1]:
            raise ValueError(
                f"Validity mask {mask.shape} does not match predictions {array.shape}"
            )
        masks.append(mask)
    masks = np.stack(masks)
    counts = masks.sum(axis=0)
    probs = (arrays * masks[..., None]).sum(axis=0) / np.maximum(counts[..., None], 1)
    probs[counts == 0] = 1.0 / probs.shape[1]
    return probs.astype(np.float32), counts > 0


def write_subject_hdf5(
    *, site_root: Path, bids_folder: str, model_names: List[str],
    lead_types: List[str], pred_sec: int, n_classes: int,
    run_ensemble: bool, edf_path: Path, review_software: str,
) -> Optional[Path]:
    """Consolidate selected models and channel/lead soft votes into one file."""
    model_files = {
        model: site_root / model / f"{pred_sec}s" / f"{bids_folder}.h5"
        for model in model_names
    }
    model_files = {model: path for model, path in model_files.items() if path.exists()}
    if not model_files:
        return None

    final_path = site_root / f"{bids_folder}.h5"
    temporary_path = site_root / f"{bids_folder}.h5.tmp"
    try:
        with h5py.File(temporary_path, "w") as target:
            target.attrs["BidsFolder"] = bids_folder
            target.attrs["SiteID"] = site_root.name
            target.attrs["SelectedModels"] = ",".join(model_names)
            target.attrs["NumClasses"] = n_classes
            target.attrs["PredSeconds"] = pred_sec
            models_group = target.create_group("models")
            channels: Dict[str, Dict[str, List[Tuple[str, np.ndarray, np.ndarray]]]] = defaultdict(lambda: defaultdict(list))
            for model, path in model_files.items():
                with h5py.File(path, "r") as source:
                    model_group = models_group.create_group(model)
                    model_leads = []
                    for key, value in source.attrs.items():
                        model_group.attrs[key] = value
                    for lead in lead_types:
                        if lead not in source:
                            continue
                        source.copy(lead, model_group)
                        if "majority_vote" in source[lead]:
                            lead_probs = np.asarray(source[lead]["majority_vote/probs"], dtype=np.float32)
                            valid_counts = np.asarray(source[lead]["valid_channel_counts"])
                            model_leads.append((lead_probs, valid_counts > 0))
                        if not run_ensemble or pred_sec != 30:
                            continue
                        lead_group = source[lead]
                        bad_epochs = {
                            name[:-len("_bad_epoch_indices")]: np.asarray(ds, dtype=int)
                            for name, ds in lead_group.items()
                            if name.endswith("_bad_epoch_indices") and isinstance(ds, h5py.Dataset)
                        }
                        for name, ds in lead_group.items():
                            if not name.endswith("_pred") or not isinstance(ds, h5py.Dataset):
                                continue
                            channel = str(ds.attrs.get("channel_name", name[:-5]))
                            probs = np.asarray(ds, dtype=np.float32).mean(axis=1)
                            if probs.ndim != 2 or probs.shape[1] != n_classes:
                                continue
                            valid = np.ones(probs.shape[0], dtype=bool)
                            invalid = bad_epochs.get(channel, np.empty(0, dtype=int))
                            valid[invalid[(invalid >= 0) & (invalid < len(valid))]] = False
                            channels[lead][channel].append((model, probs, valid))

                    if len(model_leads) > 1:
                        combined, combined_valid = vote_probability_arrays(model_leads)
                        group = model_group.create_group("all").create_group("majority_vote")
                        group.create_dataset("probs", data=combined, compression="gzip")
                        group.create_dataset("labels", data=np.argmax(combined, axis=-1), compression="gzip")
                        group.create_dataset("valid", data=combined_valid, compression="gzip")
                        directory = site_root / model / f"{pred_sec}s" / "_all" / "majority"
                        write_ensemble_prediction(
                            combined.mean(axis=1) if pred_sec == 30 else combined,
                            directory, bids_folder, "model_all", edf_path,
                            review_software, model=model,
                        )
                        np.save(
                            directory / f"{bids_folder}_VALID.npy",
                            combined_valid.all(axis=1) if pred_sec == 30 else combined_valid,
                        )

            if run_ensemble and pred_sec == 30:
                ensemble_group = target.create_group("ensemble")
                ensemble_group.attrs["Method"] = "soft_vote"
                lead_results = []
                for lead in lead_types:
                    if not channels[lead]:
                        continue
                    lead_group = ensemble_group.create_group(lead)
                    channel_group = lead_group.create_group("channels")
                    channel_results = []
                    for channel, sources in sorted(channels[lead].items()):
                        probs, valid = vote_probability_arrays([(p, v) for _, p, v in sources])
                        group = channel_group.create_group(channel)
                        group.attrs["SourceModels"] = ",".join(model for model, _, _ in sources)
                        group.create_dataset("probs", data=probs, compression="gzip")
                        group.create_dataset("labels", data=np.argmax(probs, axis=1), compression="gzip")
                        group.create_dataset("valid", data=valid, compression="gzip")
                        safe_channel = channel.replace("/", "-").replace(" ", "")
                        write_ensemble_prediction(
                            probs, site_root / "ensemble" / "30s" / f"_{lead}" / safe_channel,
                            bids_folder, channel, edf_path, review_software,
                        )
                        channel_results.append((probs, valid))
                    majority, majority_valid = vote_probability_arrays(channel_results)
                    group = lead_group.create_group("majority")
                    group.create_dataset("probs", data=majority, compression="gzip")
                    group.create_dataset("labels", data=np.argmax(majority, axis=1), compression="gzip")
                    group.create_dataset("valid", data=majority_valid, compression="gzip")
                    group.attrs["SourceChannels"] = ",".join(sorted(channels[lead]))
                    write_ensemble_prediction(
                        majority, site_root / "ensemble" / "30s" / f"_{lead}" / "soft-voting",
                        bids_folder, "ensemble_majority", edf_path, review_software,
                    )
                    lead_results.append((majority, majority_valid))
                if len(lead_results) > 1:
                    combined, combined_valid = vote_probability_arrays(lead_results)
                    group = ensemble_group.create_group("all")
                    group.create_dataset("probs", data=combined, compression="gzip")
                    group.create_dataset("labels", data=np.argmax(combined, axis=1), compression="gzip")
                    group.create_dataset("valid", data=combined_valid, compression="gzip")
                    write_ensemble_prediction(
                        combined, site_root / "ensemble" / "30s" / "_all" / "soft-voting",
                        bids_folder, "ensemble_all", edf_path, review_software,
                    )
        temporary_path.replace(final_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    for path in model_files.values():
        path.unlink()
    # Remove HDF5 files left by the previous per-lead output layout on reruns.
    for lead in [*lead_types, "all"]:
        old_path = site_root / "ensemble" / f"{pred_sec}s" / f"_{lead}" / "soft-voting" / f"{bids_folder}.h5"
        old_path.unlink(missing_ok=True)
    logger.info("Consolidated subject HDF5: %s", final_path)
    return final_path

