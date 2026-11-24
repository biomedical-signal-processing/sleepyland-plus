"""Convert sleep-stage predictions into annotation and review-software files."""

import os
from pathlib import Path
import numpy as np
from datetime import timedelta
from usleepy.utils.header_extractors import extract_header
import logging


def npyprobs_to_annot(
    probs: np.ndarray,
    out_path: Path,
    channel: str = ".",
    fmt: str = "standard",
    time_encoding: str = "elapsed_seconds",
    start_time: str = "00:00:00",
    model: str = ".",
):
    """
    Convert a [n_epochs, 5] probability array (W,N1,N2,N3,R)
    into a Luna-style .annot file.
    """
    # Expect [n, 5]
    if probs.ndim != 2 or probs.shape[1] != 5:
        raise ValueError(f"Expected probs with shape (n, 5), got {probs.shape}")

    # Stage orders
    npy_stages   = ["W", "N1", "N2", "N3", "R"]
    annot_stages = ["N1", "N2", "N3", "R", "W"]
    reordered = probs[:, [npy_stages.index(s) for s in annot_stages]]

    # Time handling (30 s epochs)
    def _parse_time_str(time_str: str) -> timedelta:
        h, m, s = map(int, time_str.split(":"))
        return timedelta(hours=h, minutes=m, seconds=s)

    def _format_time(start_seconds: int, encoding: str, start_td: timedelta, epoch_idx: int):
        if encoding == "elapsed_seconds":
            return start_seconds, start_seconds + 30
        elif encoding == "clock_time":
            start_clock = (start_td + timedelta(seconds=start_seconds)).strftime("%H:%M:%S")
            stop_clock = (start_td + timedelta(seconds=start_seconds + 30)).strftime("%H:%M:%S")
            return start_clock, stop_clock
        elif encoding == "elapsed_hms":
            start_hms = f"0+{str(timedelta(seconds=start_seconds))}"
            stop_hms = f"0+{str(timedelta(seconds=start_seconds + 30))}"
            return start_hms, stop_hms
        elif encoding == "epoch":
            return f"e{epoch_idx}", f"e{epoch_idx + 1}"
        else:
            raise ValueError(f"Unknown time_encoding: {encoding!r}")

    start_td = _parse_time_str(start_time)
    lines = []
    start_seconds = 0

    for i, p in enumerate(reordered):
        start_t, stop_t = _format_time(start_seconds, time_encoding, start_td, i)
        max_idx = int(np.argmax(p))
        max_stage = annot_stages[max_idx]

        if fmt == "standard":
            meta = ";".join([f"p{st}={val:.8g}" for st, val in zip(annot_stages, p)])
            lines.append(f"stage\t{max_stage}\t{channel}\t{start_t}\t{stop_t}\t{meta}")
        elif fmt == "model":
            meta = f"p{max_stage}={p[max_idx]:.2f}"
            lines.append(f"{max_stage}\t{model}\t{channel}\t{start_t}\t{stop_t}\t{meta}")
        else:
            raise ValueError(f"Unknown fmt: {fmt!r}")

        start_seconds += 30

    # Header
    lines.insert(0, "class\tinstance\tchannel\tstart\tstop\tmeta")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))

def _parse_confidence(meta_str: str, stage_label: str) -> float:
    """
    Given the 'meta' column (e.g. "pN1=0.00;...;pW=0.99")
    and a stage label (e.g. 'W'), returns the corresponding confidence.
    """
    confidence = 0.0
    search_key = f"p{stage_label}"
    for item in meta_str.split(";"):
        item = item.strip()
        if "=" in item:
            key, val = item.split("=", 1)
            if key.strip() == search_key:
                try:
                    confidence = float(val.strip())
                except ValueError:
                    confidence = 0.0
                break
    return confidence

def _confidence_code(prob: float) -> int:
    """
    Translates the probability into a confidence code:
      8: high (prob >= 0.7)
      9: medium (0.4 <= prob < 0.7)
      10: low (prob < 0.4)
    """
    if prob >= 0.7:
        return 8
    elif prob >= 0.4:
        return 9
    else:
        return 10

def annot_to_somno(
    annot_file: str,
    edf_file: str,
    out_dir: str,
    save_conf: bool = False,
    overwrite: bool = True,
    logger=None,
):
    """
    Convert a single Luna-style .annot file to SOMNOmedics .txt format.

    Parameters
    ----------
    annot_file : str
        Path to .annot file.
    edf_file : str
        Path to corresponding EDF file (for date/time info).
    out_dir : str
        Directory to save output .txt / *_conf.txt files.
    save_conf : bool
        If True, also write a *_conf.txt file with confidence codes.
    overwrite : bool
        If False, skip existing outputs.
    logger : callable or None
        Optional logging function; if None, no logging is performed.
    """
    log = logger or (lambda *a, **k: None)

    # Retrieve date/time from EDF header
    header = extract_header(edf_file)
    start_dt = header.get("date", "UnknownStart")

    if start_dt == "UnknownStart":
        date_str = "XX.XX.XXXX"
        time_str = "00:00:00"
        base_dt = None
    else:
        date_str = start_dt.strftime("%d.%m.%Y")
        time_str = start_dt.strftime("%H:%M:%S")
        base_dt = start_dt

    os.makedirs(out_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(annot_file))[0]
    output_file = os.path.join(out_dir, base_name + "_somno.txt")
    output_file_conf = os.path.join(out_dir, base_name + "_somno_conf.txt")

    if os.path.exists(output_file) and not overwrite:
        log(f"Skipping existing file: {output_file}")
        return

    if save_conf and os.path.exists(output_file_conf) and not overwrite:
        log(f"Skipping existing file: {output_file_conf}")
        return

    # Stage label -> SOMNOmedics numeric code
    stage_map = {"N3": 0, "N2": 1, "N1": 2, "R": 3, "W": 4}

    numeric_stages = []
    conf_codes = []

    try:
        with open(annot_file, "r") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line or line.startswith("class"):
                continue
            cols = line.split("\t")
            if len(cols) < 6:
                continue

            stage_label = cols[1].strip()
            meta_str = cols[5].strip()

            code = stage_map.get(stage_label, 5)
            numeric_stages.append(code)

            if save_conf:
                prob = _parse_confidence(meta_str, stage_label)
                conf_codes.append(_confidence_code(prob))
    except Exception as e:
        log(f"Error reading {annot_file}: {e}")
        return

    try:
        # Main .txt
        with open(output_file, "w") as out_f:
            out_f.write("Signal ID: SchlafProfil\\profil\n")
            out_f.write(f"Start Time: {date_str} {time_str}\n\n")
            for i, s in enumerate(numeric_stages):
                if base_dt is not None:
                    epoch_dt = base_dt + timedelta(seconds=30 * i)
                    epoch_time_str = epoch_dt.strftime("%H:%M:%S") + ",000"
                else:
                    total_seconds = 30 * i
                    h = total_seconds // 3600
                    m = (total_seconds % 3600) // 60
                    sec = total_seconds % 60
                    epoch_time_str = f"{h:02}:{m:02}:{sec:02},000"
                out_f.write(f"{epoch_time_str}; {s}\n")
        log(f"File saved: {output_file}")

        # Optional *_conf.txt
        if save_conf:
            with open(output_file_conf, "w") as out_fc:
                out_fc.write("Signal ID: SchlafProfil\\echtheit\n")
                out_fc.write(f"Start Time: {date_str} {time_str}\n\n")
                for i, (s, conf) in enumerate(zip(numeric_stages, conf_codes)):
                    if base_dt is not None:
                        epoch_dt = base_dt + timedelta(seconds=30 * i)
                        epoch_time_str = epoch_dt.strftime("%H:%M:%S") + ",000"
                    else:
                        total_seconds = 30 * i
                        h = total_seconds // 3600
                        m = (total_seconds % 3600) // 60
                        sec = total_seconds % 60
                        epoch_time_str = f"{h:02}:{m:02}:{sec:02},000"
                    out_fc.write(f"{epoch_time_str}; {s}; {conf}\n")
            log(f"File saved: {output_file_conf}")
    except Exception as e:
        log(f"Error writing SOMNO files for {annot_file}: {e}")

def annot_to_compumedics(
    annot_file: str,
    edf_file: str,
    out_dir: str,
    overwrite: bool = True,
    logger=None,
):
    """
    Convert a single Luna-style .annot file to Compumedics-style .txt format.

    Output contains only the stage label from column 2 of each valid row,
    one label per line.
    """
    log = logger or (lambda *a, **k: None)

    os.makedirs(out_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(annot_file))[0]
    output_file = os.path.join(out_dir, base_name + "_compumedics.txt")

    if os.path.exists(output_file) and not overwrite:
        log(f"Skipping existing file: {output_file}")
        return

    labels = []

    try:
        with open(annot_file, "r") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line or line.startswith("class"):
                continue
            cols = line.split("\t")
            if len(cols) < 2:
                continue

            stage_label = cols[1].strip()
            labels.append(stage_label)
    except Exception as e:
        log(f"Error reading {annot_file}: {e}")
        return

    try:
        with open(output_file, "w") as out_f:
            for label in labels:
                out_f.write(f"{label}\n")
        log(f"File saved: {output_file}")
    except Exception as e:
        log(f"Error writing Compumedics file for {annot_file}: {e}")


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Review-software export
# ---------------------------------------------------------------------------

def export_for_review_software(
    annot_path: Path,
    edf_path: Path,
    prediction_dir: Path,
    review_software: str,
) -> None:
    """Write import files beneath the selected software's directory."""
    if review_software not in {"somnomedics", "compumedics", "both", "none"}:
        raise ValueError(f"Unsupported review software: {review_software}")

    if review_software in {"somnomedics", "both"}:
        annot_to_somno(
            annot_file=str(annot_path),
            edf_file=str(edf_path),
            out_dir=str(prediction_dir / "somnomedics"),
            save_conf=True,
            overwrite=True,
            logger=logger.info,
        )

    if review_software in {"compumedics", "both"}:
        annot_to_compumedics(
            annot_file=str(annot_path),
            edf_file=str(edf_path),
            out_dir=str(prediction_dir / "compumedics"),
            overwrite=True,
            logger=logger.info,
        )

