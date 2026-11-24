# SLEEPYLAND-PLUS

SLEEPYLAND-PLUS is a lighter version of [SLEEPYLAND](https://github.com/biomedical-signal-processing/sleepyland) for staging your own EDF sleep recordings. SLEEPYLAND focuses on reproducible evaluation of sleep-staging models on large datasets, with workflows built around NSRR data. SLEEPYLAND-PLUS keeps only what you need to apply the same models to your own data:

- **Your own EDFs, directly.** Nothing depends on NSRR data, credentials or annotations, and none of the NSRR-linked functions and services are included.
- **Preprocessing of your EDFs** before staging: optional notch and bandpass filters, amplitude clipping, resampling and scaling.
- **Quality control on channels and epochs** (coming soon): the step already appears in the interface, but automatic flagging and exclusion of corrupted channels and epochs is not active yet.

It runs USleep, DeepResNet and SleepTransformer through a guided web interface, with per-channel and per-modality results and soft-voting ensembles. You can inspect hypnograms and stage probabilities, and export annotations for review in SOMNOmedics or Compumedics. Everything runs locally on your computer in Docker; uploaded files go to `input/` and results to `output/`.

## Start the web app

SLEEPYLAND-PLUS runs on Docker. You don't need to install it first: if the launcher doesn't find Docker, it opens the Docker Desktop download page for you, and you launch SLEEPYLAND-PLUS again once Docker is installed. To get this out of the way in advance, install Docker for your platform:

- **Windows:** [Docker Desktop for Windows](https://docs.docker.com/desktop/setup/install/windows-install/)
- **macOS:** [Docker Desktop for Mac](https://docs.docker.com/desktop/setup/install/mac-install/) (choose the Apple silicon or Intel download to match your Mac)
- **Linux:** [Docker Engine](https://docs.docker.com/engine/install/) for your distribution, plus the [Compose plugin](https://docs.docker.com/compose/install/linux/) (Compose v2)

Download the launcher for your platform from the [Releases page](https://github.com/biomedical-signal-processing/sleepyland-plus/releases/latest) and double-click it:

- **Windows:** [`SLEEPYLAND-PLUS.exe`](https://github.com/biomedical-signal-processing/sleepyland-plus/releases/latest/download/SLEEPYLAND-PLUS.exe)
- **macOS:** open [`SLEEPYLAND-PLUS.dmg`](https://github.com/biomedical-signal-processing/sleepyland-plus/releases/latest/download/SLEEPYLAND-PLUS.dmg), then double-click `SLEEPYLAND-PLUS.app`
- **Linux:** extract [`SLEEPYLAND-PLUS-linux.tar.gz`](https://github.com/biomedical-signal-processing/sleepyland-plus/releases/latest/download/SLEEPYLAND-PLUS-linux.tar.gz), then double-click `SLEEPYLAND-PLUS.desktop` (or run `./sleepyland-plus`)

A terminal window shows each step: it checks Docker (opening the Docker Desktop download page if Docker is missing, or starting Docker Desktop if it is not running), downloads the images from Docker Hub (the first download can take a while), starts the services and opens [http://localhost:8887](http://localhost:8887). Your files are kept in `SLEEPYLAND-PLUS/input` and `SLEEPYLAND-PLUS/output` in your home folder.

> [!CAUTION]
> **Security warning on first launch.** The launchers are not yet signed, so your operating system warns you the first time you open one.
>
> - **macOS (Gatekeeper):** if macOS says the app "cannot be opened" or "could not be verified", click **Done** (do not move it to the Bin). Open **System Settings → Privacy & Security**, scroll down to the message about `SLEEPYLAND-PLUS.app` and click **Open Anyway**, then confirm with your password. You only need to do this once.
> - **Windows (SmartScreen):** if "Windows protected your PC" appears, click **More info → Run anyway**. If there is no **Run anyway** button (common on managed or work PCs), close the warning, right-click `SLEEPYLAND-PLUS.exe` → **Properties**, tick **Unblock** at the bottom and click **OK**, then double-click it again. If it is still blocked, use [Build from source](#build-from-source-instead) below.
>
> On managed or work laptops without administrator rights, IT policies may block the launcher or the Docker Desktop installation entirely. In that case, ask your IT department for help.

### Build from source instead

Clone or download the **complete repository**; the setup scripts build all services from this source checkout. The first build needs internet access and can take time.

```bash
git clone https://github.com/biomedical-signal-processing/sleepyland-plus.git
cd sleepyland-plus
```

From the repository directory, run the setup script for your platform:

- **Windows PowerShell:** `Set-ExecutionPolicy -Scope Process Bypass`, then `.\win_setup.ps1`
- **macOS:** `chmod +x macos_setup.sh`, then `./macos_setup.sh`
- **Linux:** `chmod +x linux_setup.sh`, then `./linux_setup.sh`

Open [http://localhost:8887](http://localhost:8887) when the services start. To stop them, run `docker compose -p sleepyland-plus down`. To restart without rebuilding, run `docker compose -p sleepyland-plus up -d`. If the page is unavailable, check `docker compose -p sleepyland-plus ps` and `docker compose -p sleepyland-plus logs -f`.

## Use the web interface

1. **Upload EDF files.** SLEEPYLAND-PLUS stages the full duration supplied in each file. Trim to **LIGHTS-OFF to LIGHTS-ON** before upload if you only want that interval scored.
2. **Select modalities.** Choose EEG, EOG, and/or EMG only when their channels are present. EMG requires a recognized **chin** channel; see [Supported channel names](#supported-channel-names) for the labels each modality accepts. Check EDF channel names and signal quality before upload; corrupted channels are not automatically excluded from combined predictions. ECG is unavailable.
3. **Review preprocessing.** Optional notch and bandpass filters apply to EEG and EOG, not chin EMG. IQR amplitude clipping is on by default for all modalities. Resampling to 128 Hz when needed and robust scaling follow. If your EDF is already filtered, choose **Skip optional filters**; IQR clipping remains on unless you turn it off separately. Automatic channel and epoch exclusion is planned but unavailable in the UI.
4. **Choose models and exports.** Select one or more models, Adult or Children weights, and SOMNOmedics or Compumedics as the review software. Optional soft voting averages available model and lead probabilities. USleep alone can use a finer output resolution; multi-model runs use 30 seconds. This checkout bundles chin EMG weights for adult models only.
5. **Name the output folder and predict.** For example, `patient_batch_01` creates `output/patient_batch_01/`. The **Hypnograms** tab shows predicted stages, **Hypnodensity** shows stage probabilities, and **History** reloads saved plots without rerunning inference.

When you are done, click **Stop SLEEPYLAND** at the top right of the page. It stops all SLEEPYLAND containers (cancelling any prediction still running) and keeps your input and output files. Start the launcher again to use the app.

### Supported channel names

SLEEPYLAND-PLUS selects channels by their EDF label, ignoring upper/lower case. A label can have up to three parts: an optional **prefix**, the **signal name**, and an optional **reference**. The parts can be joined by `-`, `:`, `/`, `_`, a space, or ` - `. Any combination of the parts listed below is accepted. Rename channels before upload if they don't match.

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

The web interface shows a short version of this list behind the (i) icon on each modality card.

The Output Folder step includes an expandable folder example and a guide to `.npy`, `.annot`, `.h5`, and review-software files. For model details, exact CLI options, weight availability, and output formats, see the [inference engine README](usleepyland/uSLEEPYLAND-PLUS/README.md). For a standalone Docker batch workflow without the web app, see its [Docker guide](usleepyland/uSLEEPYLAND-PLUS/Docker-Standalone.md).

## Citation

If you use this software, please cite:

> Rossi AD, Metaldi M, Bechny M, Filchenko I, Meer JV, Schmidt MH, Bassetti CLA, Tzovara A, Faraci FD, Fiorillo L. SLEEPYLAND: trust begins with fair evaluation of automatic sleep staging models. *npj Digital Medicine*. 2026;9:55. [doi:10.1038/s41746-025-02237-2](https://doi.org/10.1038/s41746-025-02237-2).

See [LICENSE](LICENSE) for the repository's license. Bundled model implementations and weights may have separate terms.
