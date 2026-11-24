# uSLEEPYLAND-PLUS inference engine

This directory contains the EDF inference engine used by the SLEEPYLAND-PLUS web interface. Run `run.py` directly when you need a command-line batch workflow. The repository includes the model code and pretrained weight folders; keep `weights/` beside `run.py` when running it outside Docker.

For a standalone Docker batch image without the web app, see [Docker-Standalone.md](Docker-Standalone.md).

The web interface and the CLI expose USleep, DeepResNet, and SleepTransformer. Model weights and channel types must match: `run.py` skips a model/lead pair when its weight folder is missing.

## Set up the environment

Run these commands from this directory. Conda and an internet connection are needed for the first installation.

```bash
conda env create -f environment.yaml
conda run -n usleepyland-plus python -m pip install "pip<23"
conda run -n usleepyland-plus python -m pip install -r requirements.txt
conda run -n usleepyland-plus python run.py --help
```

`environment.yaml` creates the Python 3.9 base environment; `requirements.txt` installs the CPU inference libraries. The application does not require PyTorch or CUDA packages. GPU inference requires a separately configured TensorFlow installation and compatible drivers.

## Run one EDF

From this directory on Linux, macOS, or WSL:

```bash
conda run --no-capture-output -n usleepyland-plus python run.py \
  --edf_file /absolute/path/to/recording.edf \
  --output_root /absolute/path/to/results \
  --site_id study-001 \
  --model_name usleep \
  --lead_types eeg,eog \
  --pred_sec 30 \
  --model_set adult \
  --review_software somnomedics
```

In Windows PowerShell, use backticks for line continuation and Windows paths:

```powershell
conda run --no-capture-output -n usleepyland-plus python run.py `
  --edf_file 'C:\data\recording.edf' `
  --output_root 'C:\results' `
  --site_id study-001 `
  --model_name usleep `
  --lead_types eeg,eog `
  --pred_sec 30 `
  --model_set adult `
  --review_software somnomedics
```

The script writes results under `<output_root>/<site_id>/`. It scores from the first sample to the last **complete 30-second epoch** in the EDF. It does not detect lights off or lights on; trim the file first if you only want that interval scored.

## Run a folder with soft voting

`--edf_dir` scans recursively. This example uses all three models and every recognized lead type at 30-second resolution:

```bash
conda run --no-capture-output -n usleepyland-plus python run.py \
  --edf_dir /absolute/path/to/edf-folder \
  --output_root /absolute/path/to/results \
  --site_id batch-001 \
  --model_name usleep,deepresnet,sleeptransformer \
  --lead_types eeg,eog,chin \
  --pred_sec 30 \
  --model_set adult \
  --run_ensemble \
  --ensemble_method soft_vote \
  --review_software compumedics
```

The engine predicts independently from each recognized channel, averages channels within a lead type, then soft-votes across available model/lead probability arrays. It writes lead-specific ensembles and, when several leads are present, an `_all` ensemble. The final ensemble requires 30-second predictions. `--review_software` accepts `somnomedics`, `compumedics`, or `both` (the CLI default).

## Channels, weights, and resolution

Only an EMG channel recognized as **chin** maps to the internal `chin` lead.

### Supported channel names

SLEEPYLAND-PLUS selects channels by their EDF label, ignoring upper/lower case. A label can have up to three parts: an optional **prefix**, the **signal name**, and an optional **reference**. The parts can be joined by `-`, `:`, `/`, `_`, a space, or ` - `. Any combination of the parts listed below is accepted. Rename channels in the EDF if they don't match.

| Modality | Signal name | Optional prefix or description | Optional reference | Not used |
|---|---|---|---|---|
| **EEG** | `EEG`, `EEG1`, `EEG2`, `EEG(sec)`, or any 10-20 electrode (`Fp1`, `Fpz`, `F3`, `F4`, `C3`, `C4`, `Cz`, `P3`, `Pz`, `O1`, `O2`, `Oz`, `T3`, `T4`, …) | `EEG` before the electrode | Mastoid (`M1`, `M2`, `A1`, `A2`), another electrode (`Cz`, `Oz`, `F7`), or `REF`, `AVG`, `LE` | Mastoid-only labels (`M1`, `A2`, `A1-A2`, `EEG A1-A2`) |
| **EOG** | `E1`, `E2`, `LOC`, `ROC`, `EOG`, `EOG1`, `EOG2`, `EOGL`, `EOGR`, `LEOG`, `REOG`, `HEOG`, `VEOG`, `Left Eye`, `Right Eye`, `L-Eye`, `R Eye` | `EOG` before the electrode, or a side/direction after it (`EOG L`, `EOG(L)`, `EOG horizontal`) | Mastoid (`M1`, `M2`, `A1`, `A2`) or the other eye (`E1-E2`, `LOC-ROC`) | EOG referenced to an EEG electrode (`E1-C3`, `E1-Fpz`, `EOG C3`) |
| **EMG** (chin only) | Any label containing `CHIN`, `SUBMENTAL` (also `SUB-MENTAL`, `SUBM`, `SUB-M`), `MENTALIS`, `MENT` (as its own word), `KINN`, or `MENTON`, with any index or side (`Chin1`, `Chin3`, `ChinL`, `ChinZ`, `LChin`, `L Chin`) | `EMG` before or after the chin word (`EMG Chin`, `EMG(chin)`, `Chin EMG`) | Another chin electrode (`Chin1-Chin2`, `LChin-RChin`) or mastoid (`Chin-M1`) | EMG without a chin word (`EMG`, `EMG1`, `EMG1-EMG2`) and leg EMG (`LLeg`, `RLeg`) |
| **ECG** | Not supported yet | | | |

A reference can be glued to the signal name with no separator only when the reference is a mastoid and the signal is a 10-20 electrode, `E1`/`E2`, or `LOC`/`ROC` (`C3M2`, `C3A2`, `E1M2`, `LOCA2`).

Examples of accepted labels, grouped by combination:

| Modality | Combination | Accepted labels |
|---|---|---|
| **EEG** | Signal name only | `C3`, `Fpz`, `EEG`, `EEG2`, `EEG(sec)` |
| **EEG** | Signal name + mastoid reference | `C3-M2`, `C3:M2`, `C3/A2`, `C3_M2`, `C3 M2`, `C3 - M2`, `C3M2`, `C3A2` |
| **EEG** | Signal name + another electrode or reference | `Fpz-Cz`, `Pz-Oz`, `Fp1-F7`, `C3-REF`, `C4-AVG` |
| **EEG** | `EEG` prefix + signal name | `EEG C3` |
| **EEG** | `EEG` prefix + signal name + reference | `EEG C3-M2`, `EEG_C3-M2`, `EEG-C3:A2`, `EEG Fpz-Cz` |
| **EOG** | Signal name only | `E1`, `LOC`, `EOG`, `EOG1`, `EOGL`, `LEOG`, `HEOG`, `Left Eye` |
| **EOG** | Signal name + mastoid reference | `E1-M2`, `E2-M1`, `E1:M2`, `E1/A2`, `E1_M2`, `E1 M2`, `E1M2`, `LOC-A2`, `ROC-A1`, `LOCA2`, `EOG1:A1`, `EOG2-M2`, `Right Eye-M1` |
| **EOG** | Signal name + other eye | `E1-E2`, `LOC-ROC` |
| **EOG** | `EOG` prefix + signal name + reference | `EOG E1-M2`, `EOG_E2-M1`, `EOG LOC-M2` |
| **EOG** | `EOG` + side or direction | `EOG L`, `EOG(L)`, `EOG-L`, `EOG horizontal`, `EOG Left Horiz` |
| **EMG** | Chin word only | `Chin`, `Submental`, `Sub-mental`, `SubM`, `Mentalis`, `Kinn`, `Menton` |
| **EMG** | Chin word + index or side | `Chin1`, `Chin3`, `ChinL`, `ChinR`, `ChinZ`, `Chin Z`, `LChin`, `L Chin` |
| **EMG** | Chin electrode + another chin electrode | `Chin1-Chin2`, `Chin2-Chin3`, `Chin1-Chin3`, `LChin-RChin`, `ChinL-ChinR` |
| **EMG** | Chin electrode + mastoid reference | `Chin-M1`, `Chin1-M2`, `ChinL:A1` |
| **EMG** | `EMG` before or after the chin word | `EMG Chin`, `EMG_Chin1-Chin2`, `EMG(chin)`, `Chin EMG`, `EMG submental`, `EMG Ment` |

If a selected modality produces no result, check the recognized-channel list in the run log.

The weight roots are `weights/adult/` and `weights/children/`. EEG and EOG checkpoints for USleep, DeepResNet, and SleepTransformer are bundled for both groups. Chin EMG checkpoints are bundled for **adult** models only. The prediction lead is called `chin`, but its weight folders end in `_emg` (for example, `weights/adult/usleep-nsrr-2024_emg/`).

USleep accepts `--pred_sec` values `1, 2, 3, 5, 6, 10, 15, 30`; DeepResNet and SleepTransformer produce 30-second predictions. The web interface offers a smaller subset of these resolutions. The CLI accepts `--n_classes` values 2, 3, 4, or 5 (default 5), but the current `.annot` writer requires five probabilities in W, N1, N2, N3, REM order. Use five classes when you need annotation or review-software export.

## Preprocessing and quality control

Each channel follows this order:

1. Optional notch filtering with `--apply_notch --notch_freq 50` (or 60).
2. Optional bandpass filtering with `--apply_bandpass --bandpass_low 0.3 --bandpass_high 35` (example cutoffs).
3. IQR amplitude clipping, enabled by default; disable with `--no_quality_control`.
4. Resampling to 128 Hz if the original rate differs.
5. Robust scaling by default; `--scaling none` disables it.

Notch and bandpass settings apply to EEG and EOG, **not chin EMG**. Use cutoffs appropriate for both selected leads and the original sampling rate. IQR clipping, resampling, and scaling apply to all leads. Clipping limits extreme values but does not remove channels or epochs from the averaged prediction.

The CLI additionally supports experimental `--apply_channel_qc` with `--max_bad_epoch_fraction 0.2`. It flags 30-second epochs with amplitudes beyond the IQR threshold, excludes a channel if its flagged fraction exceeds the threshold, and omits flagged epochs from that channel's lead average. This option is **not exposed in the web interface**. Majority outputs include a `*_VALID.npy` mask; without channel QC, its epochs are normally all valid.

## Find the output

For `recording.edf`, model exports live in `<output_root>/<site_id>/<model>/<resolution>s/`. Under `_eeg/`, `_eog/`, and `_chin/`, a folder named for each recognized channel contains its prediction. A sibling `majority/` folder contains the channel-averaged result for that lead. When multiple leads are available, each model also exports their equal-weight average under `_all/majority/`. Ensemble exports have a matching-channel folder and a `soft-voting/` lead majority under `ensemble/30s/_<lead>/`; `_all/soft-voting/` contains the equal-weight lead average. The single `recording.h5` is at `<output_root>/<site_id>/`.

- `recording_PRED.npy`: NumPy stage probabilities. A 30-second prediction has one row per epoch and five columns for W, N1, N2, N3, and REM.
- `recording_PRED.annot`: tab-separated 30-second rows with chosen stage, channel, start/stop time, and all five stage probabilities in `meta`. Annotations remain at 30 seconds even when USleep outputs finer predictions.
- `recording_VALID.npy`: one Boolean contribution value per epoch in each model's `majority/` folder.
- `<output_root>/<site_id>/recording.h5`: one file per recording. `/models/<model>/<lead>` holds channel and lead probabilities, labels, and processing metadata; `/models/<model>/all/majority_vote` holds the combined lead result. `/ensemble/<lead>/channels/<channel>` holds matching-channel soft votes, `/ensemble/<lead>/majority` holds the channel average, and `/ensemble/all` holds the combined lead result when multiple leads are available.
- `somnomedics/` or `compumedics/`: files generated for the chosen review software when annotation export succeeds.

The runner also writes `sleep_model_extraction.log` in the study folder by default. Use `python run.py --help` for all options, including directory scan filters, custom weights, and logging paths.
