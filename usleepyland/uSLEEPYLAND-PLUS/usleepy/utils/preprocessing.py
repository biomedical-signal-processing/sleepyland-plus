"""
Functions for channel-wise scaling of PSG data

Implements the MultiChannelScaler, which fits and applies scalers from the
sklearn.preprocessing module individually to channels of a PSG ndarray,
and the single-channel preprocessing used before prediction (filtering,
IQR clipping, resampling and robust scaling).
"""
import logging
import sklearn.preprocessing as preprocessing
import numpy as np
from typing import List, Optional, Tuple
import mne
from scipy import signal
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)


def assert_scaler(scaler):
    """
    Checks whether a scaler (string) exists in the sklearn.preprocessing
    module.

    Args:
        scaler: String class name representation of a potential
                sklearn.preprocessing Scaler object.

    Returns:
        True if the scaler exists in the module, otherwise False
    """
    if isinstance(scaler, str):
        scaler = [scaler]
    ok = []
    for sc in scaler:
        ok.append(sc in preprocessing.__dict__)
    return all(ok)


def get_scaler(scaler):
    """
    Returns a MultiChannelScaler initialized to perform scaling according to
    the 'scaler' argument

    Uses default parameters for all scalers except QuantileTransformer, which
    uses output data according to a normal distribution instead of the default
    uniform.

    Args:
        scaler: Either a string specifying a single scaler from the
                sklearn.preprocessing module to apply to all PSG channels
                (one such fit to each channel separately), or
                a list of strings specifying what scaler to fit and apply in
                each channel. E.g. ['StandardScaler', 'RobustScaler'] will
                apply the StandardScaler to channel 0 and RobustScaler to
                channel 1.

    Returns:
        A MultiChannelScaler object.
    """
    if isinstance(scaler, str):
        scaler = [scaler]
    scalers = []
    for sc in scaler:
        kwargs = {}  # Currently kwargs cannot be passed to the scalers
        if sc == "QuantileTransformer":
            kwargs["output_distribution"] = "normal"
        scalers.append((preprocessing.__dict__[sc], kwargs))
    return MultiChannelScaler(scalers=scalers)


def apply_scaling(X, scaler):
    """
    Initializes a MultiChannelScaler object and applies it immediately to the
    same data. Also returns the fit scaler

    Args:
        X:      A ndarray, PSG data, shape [N, C]
        scaler: A string or list of strings, see 'get_scaler'

    Returns:
        Transformed X data
        The fit MultiChannelScaler object
    """
    # Get scaler
    multi_scaler = get_scaler(scaler).fit(X)

    # Fit and apply transformation
    return multi_scaler.transform(X), multi_scaler


def batch_scaling(X, batch_scaler):
    """
    Fit and apply scaler to batch X. Used only when a batch_scaler is set.
    This operates on a per-instance and per-channel basis.
    Can be used both on PSG raw data or spectrograms. The X array is modified in-place.

    Args:
        X: (ndarray) A batch of raw data of shape [batch_size, n_epochs, epoch_len, n_channels]
        or a spectrogram of shape [batch_size, n_epochs, n_time, n_freq, n_channels]
    """
    for i, input_ in enumerate(X):
        org_shape = input_.shape
        input_ = input_.reshape(-1, org_shape[-1])
        scaled_input = apply_scaling(input_, batch_scaler)[0]
        X[i] = scaled_input.reshape(org_shape)


class MultiChannelScaler(object):
    """
    Wraps around Scaler objects from the sklearn.preprocessing module,
    initializing, fitting, storing and applying such scalers to/for individual
    channels of a [N, C] shaped ndarray (e.g. PSG data with C channels)
    """
    def __init__(self, scalers, with_centering=True):
        """
        Initializes the scaler object with a set of scaler strings. Does not
        fit or transform any data yet.

        Args:
            scalers:         A list of 2-tuples/lists each of format
                             (scaler class name, kwargs). The outer list should
                             be of length equal to the number of channels of the
                             PSG passed to self.fit, self.transform,
                             self.fit_transform.
            with_centering:  Apply centering to the data. If False, only
                             scaling of the data is applied.
        """
        err = "'scalers' should be a list of 2-tuples/lists (each of format " \
              "(scaler class, kwargs to scaler init), got {}".format(scalers)
        if not isinstance(scalers, (tuple, list, np.ndarray)):
            raise ValueError(err)
        if any([len(s) != 2 for s in scalers]):
            raise ValueError(err)
        # Store scaler class and passed parameters
        self.scaler_tuples = scalers
        self.with_centering = with_centering

        # Store list of initialized scalers fit to each channel
        self.scalers = []

        # Store number of channels
        self.n_channels = None

    def fit(self, X):
        """
        Fit all scalers specified in self.scalers to individual channels of X

        Args:
            X: ndarray of shape [N, C], where C == len(self.scaler_tuples)

        Returns:
            self
        """
        if X.ndim != 2:
            raise ValueError("Invalid shape for X (%s)" % X.shape)
        # Set number of channels
        self.n_channels = X.shape[-1]
        if len(self.scaler_tuples) == 1:
            scaler_tups = self.scaler_tuples * self.n_channels
        else:
            if len(self.scaler_tuples) != self.n_channels:
                raise ValueError("Number of passed scalers ({}) does not "
                                 "match the number of channels in X "
                                 "({})".format(len(self.scaler_tuples),
                                               self.n_channels))
            scaler_tups = self.scaler_tuples

        # Fit the scalers to each channel of X
        fit_scalers = []
        for i, (scaler_cls, scaler_kwargs) in enumerate(scaler_tups):
            try:
                scaler_cls = scaler_cls(**scaler_kwargs,
                                        with_centering=self.with_centering)
            except TypeError:
                scaler_cls = scaler_cls(**scaler_kwargs,
                                        with_mean=self.with_centering)
            xs = X[:, i]
            scaler_cls.fit(xs.reshape(-1, 1))
            fit_scalers.append(scaler_cls)
        self.scalers = fit_scalers
        return self

    def transform(self, X):
        """
        Transform each channel in X according to the (fitted) scalers in
        self.scalers

        Args:
            X: ndarray of shape [N, C], where C == len(self.scalers)

        Returns:
            X, transformed data, shape [N, C]
        """
        if X.shape[-1] != self.n_channels:
            raise ValueError("Invalid input of dimension %i, expected "
                             "last axis with %i channels" % (X.ndim,
                                                             self.n_channels))
        # Prepare volume like X to store results
        transformed = np.empty_like(X)
        for i in range(self.n_channels):
            scl = self.scalers[i]
            s = scl.transform(X[:, i].reshape(-1, 1))
            transformed[:, i] = s.reshape(X.shape[:-1])
        return transformed

    def fit_transform(self, X):
        """ Fits scalers to and immediately transform data in X """
        self.fit(X)
        return self.transform(X)


"""
This module contains functions to compute the spectrogram of the PSG data.
"""

import numpy as np
import scipy.signal as sig

def compute_spectrogram(
    X,
    signal_fs=128,
    resample_fs=100,
    Nfir=100,
    f_cutoff=[0.3, 40],
    window="hamming",
    pass_zero="bandpass",
    scale=True,
    nfft=256,
    log_magnitude=True,
    eps=0.0001
):
    """
    Compute the spectrogram of the batched PSG data. If batch size is missing
    (i.e. X has shape [n_epochs, epoch_len, n_channels]), prior to calling this
    function, add a batch dimension to the data, and squeeze it back after if needed.

    Args:
        X: (ndarray)  A batch of data of shape [batch_size, n_epochs, epoch_len, n_channels]
        signal_fs: (int) The sampling frequency of the PSG data
        resample_fs: (int) The resampling frequency of the PSG data
        Nfir: (int) The number of FIR filter taps
        f_cutoff: (list) The cutoff frequencies for the bandpass filter
        window: (str) The window function for the FIR filter
        pass_zero: (str) The type of filter
        scale: (bool) Scale the filter coefficients
        nfft: (int) The number of FFT points
        log_magnitude: (bool) Compute the log magnitude of the spectrogram
        eps: (float) Small number to avoid log(0)

    Returns:
        The spectrogram of the PSG data of shape [batch_size, n_epochs, n_time, n_freq, n_channels]
    """

    assert (len(X.shape) == 4)

    epoch_len = 30 * signal_fs

    X = sig.resample(
        X,
        int(epoch_len / signal_fs * resample_fs),
        axis=2)

    # bandpass filter
    b = sig.firwin(
        numtaps=Nfir + 1,
        cutoff=f_cutoff,
        window=window,
        pass_zero=pass_zero,
        scale=scale,
        fs=resample_fs)
    X = sig.filtfilt(b, 1, X, axis=2)

    # compute spectrogram
    f, t, S_epoch = sig.spectrogram(
        X,
        fs=resample_fs,
        window=window,
        nperseg=2 * resample_fs,
        noverlap=resample_fs,
        nfft=nfft,
        detrend=False,
        scaling='density',
        mode='complex',
        axis=2)

    if log_magnitude:
        S_epoch = 20 * np.log10(np.abs(S_epoch) + eps)

    # transpose to [batch_size, n_epochs, n_time, n_freq, n_channels]
    S_epoch = S_epoch.transpose(0, 1, 4, 2, 3)

    return S_epoch


# ---------------------------------------------------------------------------
# Single-channel preprocessing  (mirrors original SleepStudy._load() order)
# ---------------------------------------------------------------------------

def clip_noisy_values(
    sig: np.ndarray,
    sample_rate: float,
    period_length_sec: int = 30,
    min_max_times_global_iqr: float = 20.0,
) -> Tuple[np.ndarray, List[int]]:
    """
    Clip extreme values in a 1-D channel signal using a global IQR threshold.

    Mirrors psg_utils clip_noisy_values, adapted for a single 1-D channel
    (the upstream function operates on [N, C] arrays).

    Parameters
    ----------
    sig                     : 1D np.ndarray of raw channel data
    sample_rate             : sampling rate of `sig`
    period_length_sec       : epoch length in seconds (default 30)
    min_max_times_global_iqr: clip threshold = IQR × this factor

    Returns
    -------
    sig       : clipped 1D np.ndarray (in-place modification)
    noisy_idx : list of epoch indices in which at least one value was clipped
    """
    sig = np.asarray(sig, dtype=np.float32)

    # Global IQR threshold
    iqr = np.subtract(*np.percentile(sig, [75, 25]))
    threshold = iqr * min_max_times_global_iqr

    # Identify which epochs contain clipped values (for logging/QA)
    samples_per_epoch = int(sample_rate * period_length_sec)
    n_periods = len(sig) // samples_per_epoch
    if n_periods > 0:
        temp = sig[: n_periods * samples_per_epoch].reshape(n_periods, -1)
        noisy_idx = list(np.unique(np.where(np.abs(temp) > threshold)[0]))
    else:
        noisy_idx = []

    sig = np.clip(sig, -threshold, threshold)
    return sig, noisy_idx


def sig_preprocessing(
    sig: np.ndarray,
    fs: float,
    epoch_sec: int = 30,
    target_fs: int = 128,
    # ── step 2: bandpass filter ───────────────────────────────────────────
    apply_bandpass: bool = False,
    bandpass_low: Optional[float] = None,
    bandpass_high: Optional[float] = None,
    # ── step 1: notch filter ─────────────────────────────────────────────
    apply_notch: bool = False,
    notch_freq: float = 50.0,
    # ── step 3: quality control (IQR clip) ───────────────────────────────
    apply_quality_control: bool = True,
    detect_bad_epochs: bool = False,
    min_max_times_global_iqr: float = 20.0,
    warn_fraction: float = 0.15,
    channel_id: str = "<unknown>",
    # ── step 4: resample (always done when fs != target_fs) ──────────────
    # ── step 5: scaling ──────────────────────────────────────────────────
    scaling: str = "robust",
) -> Tuple[np.ndarray, int, int]:
    """
    Single-channel preprocessing pipeline that mirrors SleepStudy._load().

    Steps (in order)
    ----------------
    1. Notch filtering              (optional, via MNE notch_filter)
    2. Bandpass filtering           (optional, via MNE filter_data)
    3. Quality control / IQR clip   (optional, clip_noisy_values)
    4. Resampling to target_fs      (always applied when fs != target_fs)
    5. Scaling                      (optional: 'robust' | 'usleepy' | 'none')
    6. Trim to full 30-s epochs

    Parameters
    ----------
    sig                       : raw 1-D channel signal
    fs                        : sampling rate of `sig`
    epoch_sec                 : epoch length in seconds (default 30)
    target_fs                 : model input sampling rate (default 128 Hz)
    apply_bandpass            : apply bandpass filter before QC and resampling
    bandpass_low              : low-cut frequency (Hz); None = high-pass only
    bandpass_high             : high-cut frequency (Hz); None = low-pass only
    apply_notch               : apply notch filter before QC and resampling
    notch_freq                : notch frequency in Hz (default 50.0)
    apply_quality_control     : apply IQR-based clip (clip_noisy_values)
    min_max_times_global_iqr  : IQR multiplier for outlier threshold
    warn_fraction             : fraction of noisy epochs that triggers a warning
    channel_id                : channel name used in log messages
    scaling                   : 'robust' or 'usleepy' -> RobustScaler
                                'none'              -> no scaling
    Returns
    -------
    sig_proc        : 1D np.float32, preprocessed and trimmed
    n_periods       : number of full 30-s epochs
    data_per_period : samples per epoch (= epoch_sec * target_fs)
    """
    sig = np.asarray(sig, dtype=np.float64).reshape(-1)

    # ------------------------------------------------------------------
    # Step 1 – Notch filtering
    # ------------------------------------------------------------------
    if apply_notch:
        logger.info("ch=%s | Notch filter | %.1f Hz", channel_id, notch_freq)
        sig2d = sig.reshape(1, -1)
        sig2d = mne.filter.notch_filter(
            sig2d, Fs=fs, freqs=[notch_freq], verbose="ERROR"
        )
        sig = np.asarray(sig2d, dtype=np.float64).reshape(-1)

    # ------------------------------------------------------------------
    # Step 2 – Bandpass filtering
    # ------------------------------------------------------------------
    if apply_bandpass:
        if bandpass_low is None and bandpass_high is None:
            raise ValueError(
                "apply_bandpass=True but both bandpass_low and bandpass_high are None."
            )
        logger.info(
            "ch=%s | Bandpass filter | l_freq=%s Hz | h_freq=%s Hz",
            channel_id, bandpass_low, bandpass_high,
        )
        sig2d = sig.reshape(1, -1)
        sig2d = mne.filter.filter_data(
            sig2d, sfreq=fs,
            l_freq=bandpass_low, h_freq=bandpass_high,
            verbose="ERROR",
        )
        sig = np.asarray(sig2d, dtype=np.float64).reshape(-1)

    # ------------------------------------------------------------------
    # Step 3 – Quality control (IQR-based clip = clip_noisy_values)
    # ------------------------------------------------------------------
    noisy_idx = []
    if apply_quality_control or detect_bad_epochs:
        clipped_sig, noisy_idx = clip_noisy_values(
            sig,
            sample_rate=fs,
            period_length_sec=epoch_sec,
            min_max_times_global_iqr=min_max_times_global_iqr,
        )
        if apply_quality_control:
            sig = clipped_sig
        samples_per_epoch = int(fs * epoch_sec)
        n_periods_qc = len(sig) // samples_per_epoch if samples_per_epoch > 0 else 0
        if n_periods_qc > 0:
            fraction = len(noisy_idx) / n_periods_qc
            qc_msg = (
                "ch=%s | Quality control (IQR clip) affected %d/%d epochs "
                "(%.1f%%) | threshold = IQR × %.1f",
                channel_id, len(noisy_idx), n_periods_qc,
                fraction * 100, min_max_times_global_iqr,
            )
            if fraction >= warn_fraction:
                logger.warning(*qc_msg)
            else:
                logger.debug(*qc_msg)

    # ------------------------------------------------------------------
    # Step 4 – Resample to target_fs
    # ------------------------------------------------------------------
    if float(fs) != float(target_fs):
        logger.info("ch=%s | Resample %.3f Hz -> %d Hz", channel_id, fs, target_fs)
        up   = int(target_fs)
        down = int(round(fs))
        g    = np.gcd(up, down)
        sig  = signal.resample_poly(sig, up // g, down // g).astype(np.float32)
        fs   = float(target_fs)

    # ------------------------------------------------------------------
    # Step 5 – Scaling
    # ------------------------------------------------------------------
    if scaling in {"robust", "usleepy"}:
        logger.info("ch=%s | RobustScaler", channel_id)
        sig = RobustScaler().fit_transform(sig.reshape(-1, 1)).reshape(-1).astype(np.float32)
    elif scaling == "none":
        pass
    else:
        raise ValueError(f"Unknown scaling mode: {scaling!r}. Choose 'robust', 'usleepy', or 'none'.")

    # ------------------------------------------------------------------
    # Step 6 – Trim to full epochs
    # ------------------------------------------------------------------
    data_per_period = int(epoch_sec * target_fs)   # 3840 at 128 Hz / 30 s
    n_periods       = len(sig) // data_per_period
    sig             = sig[: n_periods * data_per_period]

    return sig.astype(np.float32), n_periods, data_per_period, [i for i in noisy_idx if i < n_periods]

