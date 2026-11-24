"""
EDF header extraction and channel recognition (EEG / EOG / chin EMG).

A collection of functions for extracting a header of the following format:

{
    'n_channels': int,
    'channel_names': list of strings,
    'sample_rate': int
    'date': datetime or None
    'length_sec': float
}
"""

import os
from datetime import datetime
import re
from pathlib import Path
from typing import Tuple
import numpy as np
import mne
from mne.channels import make_standard_montage

def _assert_header(header):
    """
    Checks that a standardized header:
        1) contains the right field names
        2) each value has an expected type
        3) the 'length' value is greater than 0
    Args:
        header: dict
    Returns: dict
    """
    field_requirements = [
        ("n_channels", [int]),
        ("channel_names", [list]),
        ("sample_rate", [int]),
        ("date", [datetime, type(None)]),
        ("length", [int])
    ]
    for field, valid_types in field_requirements:
        if field not in header:
            raise (f"Missing value '{field}' from header '{header}'. "
                                          "This could be an error in the code implementation. "
                                          "Please raise this issue on GitHub.")
        type_ = type(header[field])
        if type_ not in valid_types:
            raise (f"Field {field} of type {type_} was not expected, expected one of {valid_types}")
    if header['length'] <= 0:
        raise (f"Expected key 'length' to be a non-zero integer, "
                                    f"but header {header} has value {header['length']}")
    return header

def _standardized_edf_header(raw_edf):
    """
    Header extraction function for RawEDF and Raw objects.
    Reads the number of channels, channel names and sample rate properties
    If existing, reads the date information as well.

    Returns:
        Header information as dict
    """
    # Each tuple below follows the format:
    # 1) output name, 2) edf_obj name, 3) function to apply to the read
    # value, 4) whether a missing value should raise an error.
    header_map = [("n_channels", "nchan", int, True),
                  ("channel_names", "ch_names", list, True),
                  ("sample_rate", "sfreq", int, True),
                  ("date", "meas_date", datetime.utcfromtimestamp, False)]
    if isinstance(raw_edf.info["meas_date"], (tuple, list)):
        assert raw_edf.info["meas_date"][1] == 0
        raw_edf.info["meas_date"] = raw_edf.info["meas_date"][0]
    header = {}
    for renamed, org, transform, raise_err in header_map:
        value = raw_edf.info.get(org)
        try:
            value = transform(value)
        except Exception as e:
            if raise_err:
                raise ("Missing or invalid value in EDF file for key {} "
                                           "- got {}".format(org, value)) from e
        header[renamed] = value
    header["length"] = len(raw_edf)
    return _assert_header(header)

def extract_edf_header(file_path):
    """
    Header reader function for .edf extension files.
    Redirects to mne.io.read_raw_edf.

    Returns:
        A dictionary of header information
    """
    from mne.io import read_raw_edf
    try:
        raw_edf = read_raw_edf(file_path, preload=False,
                           stim_channel=None, verbose=False)
    except ValueError as e:
        def preprocess_edf_in_place(file_path):
            # Read the content of the EDF file
            with open(file_path, 'rb') as file:
                content = file.read()

            # Replace 'NA      ' with '0       ' in the file content
            new_content = content.replace(b'NA      ', b'00.00.00')

            # Write the new content back to the original file
            with open(file_path, 'wb') as file:
                file.write(new_content)
        # If a ValueError occurs, check for the specific error message
        # (error occurs during mnc_ssc processing)
        if "invalid literal for int() with base 10: 'NA      '" in str(e):
            print("Encountered invalid literal error. Preprocessing the EDF file...")
            preprocess_edf_in_place(file_path)
            # Retry reading the EDF file after preprocessing
            raw_edf = read_raw_edf(file_path, preload=False,
                                   stim_channel=None, verbose=False)
            print("EDF file read successfully after preprocessing.")
        else:
            # Re-raise the exception if it's a different error
            raise e

    header = _standardized_edf_header(raw_edf)

    # Manually read channel names as-are in file without renaming, truncations etc. that
    # may be applied by MNE (as of v0.21) to ensure we exclude using the proper names.
    with open(file_path, "rb") as in_f:
        in_f.seek(252)
        n_channels = int(in_f.read(4).decode().rstrip('\x00'))
        channel_names = [in_f.read(16).strip().decode('latin-1') for _ in range(n_channels)]
        # Ignore EDF TAL channels, as is done in MNE too (v0.21)
        channel_names = [chan for chan in channel_names if "EDF Annotatio" not in chan]
        assert len(channel_names) == header["n_channels"], "Manually loaded number of channels " \
                                                           "does not match number read by MNE"
        header["channel_names"] = channel_names
    return header


_EXT_TO_LOADER = {
    "edf": extract_edf_header,
    "EDF": extract_edf_header
}


def extract_header(file_path):
    """
    Loads the header from a data-type file at path 'file_path'.

    Returns:
        dictionary of header information
    """
    fname = os.path.split(os.path.abspath(file_path))[-1]
    _, ext = os.path.splitext(fname)
    load_func = _EXT_TO_LOADER[ext[1:]]
    header = load_func(file_path)
    # Add file location data
    file_path, file_name = os.path.split(file_path)
    header['data_dir'] = file_path
    header["file_name"] = file_name
    return header


# ---------------------------------------------------------------------------
# Channel heuristics (10-20 based)
# ---------------------------------------------------------------------------

EEG_CHANNELS = [
    ch for ch in make_standard_montage("standard_1020").ch_names
    if ch not in ("M1", "M2", "A1", "A2")
]
EEG_REGEX     = re.compile(r"(\bEEG|\b{})".format(r"|\b".join(EEG_CHANNELS)), re.IGNORECASE)
# EOG cues: EOG with an optional index/side (EOG1, EOGL, LEOG, HEOG, EOG(L)),
# E1/E2 (also glued to a reference, E1M2), LOC/ROC (also LOCA2), and
# left/right eye labels. E<n> and LOC/ROC must start the label or follow a
# separator, so Pulse2 or Phase2 are not taken as EOG.
EOG_REGEX     = re.compile(
    r"(EOG(?:\d+|[LRHV])?(?![A-Z0-9])"
    r"|(?<![A-Z0-9])E\d{1,2}(?!\d)"
    r"|(?<![A-Z0-9])[LR]OC(?:(?![A-Z0-9])|(?=[AM]\d))"
    r"|(?<![A-Z0-9])(?:LEFT|RIGHT|L|R)?[\s_-]?EYE(?![A-Z]))",
    re.IGNORECASE,
)
# Chin cues match anywhere in the label so side/index variants such as
# ChinL, ChinR, Chin1, LChin, Chin1-Chin2, Sub-mental, and SubM are
# recognized too (Kinn and Menton are the German and French names).
CHIN_PATTERN  = r"CHIN|SUB[\s_-]?M|MENTAL|MENTON|KINN|(?<![A-Z])MENT(?![A-Z])"
EMG_REGEX     = re.compile(r"(\bEMG|\bMAST|RLEG|LLEG|{})".format(CHIN_PATTERN), re.IGNORECASE)
ECG_REGEX     = re.compile(r"\b(ECG|EKG|ECG1|ECG2)\b", re.IGNORECASE)
CHIN_REGEX    = re.compile(r"({})".format(CHIN_PATTERN), re.IGNORECASE)
MASTOID_REGEX = re.compile(r"(\bA1|\bA2|\bM1|\bM2)", re.IGNORECASE)
NON_SIGNAL_SUFFIX_REGEX = re.compile(r"[_\s-](?:ACC[XYZ]|IMP)$", re.IGNORECASE)


def infer_channel_type_simple(ch_name: str) -> str:
    """Classify a channel name into EEG / EOG / EMG / MASTOID / OTHER / AMBIGUOUS."""
    name = ch_name.strip()
    # Auxiliary accelerometer and impedance channels can inherit an EEG name.
    if NON_SIGNAL_SUFFIX_REGEX.search(name):
        return "OTHER"
    possible_eeg      = bool(re.search(EEG_REGEX,     name))
    possible_eog      = bool(re.search(EOG_REGEX,     name))
    possible_emg      = bool(re.search(EMG_REGEX,     name))
    possible_mastoid  = bool(re.search(MASTOID_REGEX, name))
    n_matches = int(possible_eeg) + int(possible_eog) + int(possible_emg) + int(possible_mastoid)

    if n_matches > 1:
        eeg_and_mastoid = possible_eeg and possible_mastoid
        # Drop the "EEG" prefix (with any separator, e.g. EEG_C3-M2) and check
        # that a real electrode remains next to the mastoid.
        without_prefix = re.sub(r"EEG[\s_-]*", " ", name, flags=re.IGNORECASE)
        if eeg_and_mastoid and bool(re.search(EEG_REGEX, without_prefix)):
            return "EEG"
        if eeg_and_mastoid:
            return "MASTOID"
        # A mastoid reference does not make EOG or chin EMG ambiguous
        # (e.g. E1-M2, LOC-A2, ROC-M1, Chin-M1).
        if possible_mastoid and n_matches == 2:
            if possible_eog:
                return "EOG"
            if possible_emg and re.search(CHIN_REGEX, name):
                return "EMG"
        return "AMBIGUOUS"

    if n_matches == 0:
        return "OTHER"
    if possible_eeg:      return "EEG"
    if possible_eog:      return "EOG"
    if possible_emg:      return "EMG"
    return "MASTOID"


def prepare_channels(physiological_data: dict, physiological_fs: dict):
    """
    Filter and classify channels from raw EDF dicts.

    Returns
    -------
    processed_channels : {canonical_name: 1D np.float32 signal}
    processed_fs       : {canonical_name: sampling_frequency}
    channel_lead_type  : {canonical_name: 'eeg' | 'eog' | 'chin'}
    """
    processed_channels: dict = {}
    processed_fs:       dict = {}
    channel_lead_type:  dict = {}

    for ch_name, sig in physiological_data.items():
        fs = physiological_fs.get(ch_name)
        if sig is None or fs is None:
            continue

        ch_u = ch_name.strip().upper()

        if re.search(ECG_REGEX, ch_u):
            continue                        # ECG not used by any encoder here

        base_type = infer_channel_type_simple(ch_u)

        if base_type == "EEG":
            lead_type = "eeg"
        elif base_type == "EOG":
            lead_type = "eog"
        elif base_type == "EMG":
            lead_type = "chin" if re.search(CHIN_REGEX, ch_u) else None
        else:
            lead_type = None                # mastoid / other / ambiguous

        if lead_type is None:
            continue

        canonical = ch_u.lower()
        processed_channels[canonical] = np.asarray(sig, dtype=np.float32)
        processed_fs[canonical]       = fs
        channel_lead_type[canonical]  = lead_type

    return processed_channels, processed_fs, channel_lead_type



# ---------------------------------------------------------------------------
# EDF filename stem as ID
# ---------------------------------------------------------------------------

def build_metadata_from_edf_path(
    edf_path: Path,
    site_id: str = "SITE001",
) -> dict:
    """
    Use the EDF filename stem as BidsFolder.
    Assumes EDF is already anonymised (e.g. ID0004mwt075303.edf)
    """
    bids_folder = edf_path.stem.strip()
    return {
        "BidsFolder": bids_folder,
        "SiteID": str(site_id),
    }



# ---------------------------------------------------------------------------
# MNE → dict helpers
# ---------------------------------------------------------------------------

def raw_to_physio_dicts(raw: mne.io.BaseRaw) -> Tuple[dict, dict]:
    """Convert MNE Raw to {ch_name: signal} and {ch_name: sfreq} dicts."""
    sfreq = float(raw.info["sfreq"])
    data  = raw.get_data(picks=raw.ch_names)
    physio = {ch: data[i].astype(np.float32) for i, ch in enumerate(raw.ch_names)}
    fs     = {ch: sfreq for ch in raw.ch_names}
    return physio, fs

