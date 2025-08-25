import glob
import re
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
import ast
from gwpy.timeseries import TimeSeries
import bilby
from scipy.stats import anderson, kstest, norm

def get_event_name(path: str) -> str:
    """Extract GW event name (e.g., GW150914_095045) from filename."""
    match = re.search(r"(GW\d{6}_\d{6})", path)
    if not match:
        raise ValueError(f"Could not extract event name from {path}")
    return match.group(1)


def find_analysis_group(fin: h5py.File) -> str:
    """Return preferred analysis group: default 'C01:IMRPhenomXPHM', else first with 'psds' and 'config_file'."""
    if "C01:IMRPhenomXPHM" in fin and "psds" in fin["C01:IMRPhenomXPHM"]:
        return "C01:IMRPhenomXPHM"
    for k in fin.keys():
        if isinstance(fin[k], h5py.Group):
            group = fin[k]
            has_psds = "psds" in group
            has_config = "config_file" in group or any("config" in subkey for subkey in group.keys())
            if has_psds and has_config:
                return k
    raise KeyError("No group with both 'psds' and config data found in this file")


def find_psd_group(fin: h5py.File) -> str:
    """Return preferred PSD group: default 'C01:IMRPhenomXPHM', else first with 'psds'."""
    if "C01:IMRPhenomXPHM" in fin and "psds" in fin["C01:IMRPhenomXPHM"]:
        return "C01:IMRPhenomXPHM"
    for k in fin.keys():
        if isinstance(fin[k], h5py.Group) and "psds" in fin[k]:
            return k
    raise KeyError("No group with 'psds' found in this file")


def get_pe_paths() -> Dict[str, str]:
    """Get paths to parameter estimation data files."""
    dirs = [
        "/datasets/LIGO/public/gwosc.osgstorage.org/gwdata/zenodo/ligo-virgo-kagra/2021/5546662/1",
        "/datasets/LIGO/public/gwosc.osgstorage.org/gwdata/zenodo/ligo-virgo-kagra/2022/5117702/v2",
    ]
    files = []
    for d in dirs:
        files.extend(glob.glob(f"{d}/*PEDataRelease_mixed_cosmo.h5"))
    paths = {get_event_name(f): f for f in files}
    return paths


def extract_psds(outdir: str, fmin: float = 20.0, fmax: float = 2048.0) -> str:
    """Extract PSDs from GWOSC PEDataRelease files and save them into one HDF5 file."""
    paths = get_pe_paths()
    os.makedirs(outdir, exist_ok=True)
    outfn = os.path.join(outdir, "GWTC3_psds.h5")

    with h5py.File(outfn, "w") as fout:
        for event, fpath in tqdm(paths.items(), desc="Extracting PSDs"):
            with h5py.File(fpath, "r") as fin:
                try:
                    psd_group_name = find_psd_group(fin)
                except KeyError:
                    print(f"[WARN] No psds group found in {fpath}")
                    continue

                psd_data = fin[psd_group_name]["psds"]
                g = fout.create_group(event)
                g.attrs["psd_group"] = psd_group_name

                for ifo_name in psd_data:
                    if ifo_name in ['V1', "K1"]:
                        continue  # skip Virgo/KAGRA

                    freqs = psd_data[ifo_name][:, 0]
                    vals = psd_data[ifo_name][:, 1]
                    mask = (freqs >= fmin) & (freqs <= fmax)
                    freqs, vals = freqs[mask], vals[mask]

                    ifo_group = g.create_group(ifo_name)
                    ifo_group.create_dataset("freqs", data=freqs)
                    ifo_group.create_dataset("psd", data=vals)

    return outfn


def get_specific_configs(file_path: str, h5_config_group_path: str, keys_to_retrieve: List[str]) -> Dict:
    configs_dict = {}

    try:
        if not os.path.exists(file_path):
            print(f"Error: The file '{file_path}' was not found.")
            return configs_dict

        with h5py.File(file_path, 'r') as f:
            if h5_config_group_path not in f:
                print(f"Error: The HDF5 path '{h5_config_group_path}' was not found.")
                return configs_dict

            config_group = f[h5_config_group_path]

            for key in keys_to_retrieve:
                try:
                    obj = config_group[key]

                    if isinstance(obj, h5py.Dataset):
                        content = obj[()]

                        # Handle numpy arrays containing byte strings
                        if isinstance(content, np.ndarray):
                            # Extract the first (and usually only) element from the array
                            if content.size > 0:
                                content = content.item()  # Extract scalar from array
                            else:
                                print(f"Warning: Empty array for key '{key}'")
                                continue

                        # Handle bytes by decoding first
                        if isinstance(content, bytes):
                            content = content.decode('utf-8')

                        # Try to evaluate as Python literal (dict, list, etc.)
                        try:
                            processed_value = ast.literal_eval(content)
                        except (ValueError, SyntaxError) as e:
                            print(f"Info: Storing '{key}' as string (couldn't parse as literal)")
                            # Store as string if it can't be parsed as a Python literal
                            processed_value = content

                        configs_dict[key] = processed_value

                    else:
                        print(f"Warning: '{key}' is not a dataset and was skipped.")

                except KeyError:
                    print(f"Warning: Key '{key}' not found in the HDF5 group.")

                except Exception as e:
                    print(f"Warning: Error processing '{key}': {e}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return configs_dict


def get_gw_event_configs(fpath: str, analysis_group: str = None) -> Dict:
    """Get configuration parameters for a gravitational wave event."""
    with h5py.File(fpath, 'r') as fin:
        if analysis_group is None:
            analysis_group = find_analysis_group(fin)

        # Try different possible config paths
        possible_config_paths = [
            f'{analysis_group}/config_file/config',
            f'{analysis_group}/config',
            f'{analysis_group}/configuration'
        ]

        h5_config_group = None
        for path in possible_config_paths:
            if path in fin:
                h5_config_group = path
                break

        if h5_config_group is None:
            raise KeyError(f"No config group found in {analysis_group}")

    desired_keys = [
        'channel-dict', 'deltaT', 'duration', 'maximum-frequency',
        'minimum-frequency', 'psd-fractional-overlap', 'psd-length',
        'psd-maximum-duration', 'psd-method', 'trigger-time', 'tukey-roll-off'
    ]

    configs = get_specific_configs(fpath, h5_config_group, desired_keys)
    configs['analysis_group'] = analysis_group  # Store which group was used

    # Helper function to safely extract scalar values
    def extract_scalar(value):
        """Extract scalar value from numpy array or return as-is if already scalar."""
        if hasattr(value, 'item'):  # numpy array
            return value.item()
        elif hasattr(value, '__len__') and len(value) == 1:  # single-element sequence
            return value[0]
        else:
            return value

    # Calculate timing parameters if we have the required keys
    if all(key in configs for key in ['trigger-time', 'deltaT', 'duration']):
        trigger_time = float(extract_scalar(configs['trigger-time']))
        delta_t = float(extract_scalar(configs['deltaT']))
        duration = float(extract_scalar(configs['duration']))

        end_time = trigger_time + delta_t
        start_time = end_time - duration
        psd_end_time = start_time

        # Calculate PSD duration with safety check
        psd_max_duration_raw = configs.get('psd-maximum-duration', 32 * duration)
        psd_max_duration = float(extract_scalar(psd_max_duration_raw))
        psd_duration = min(32 * duration, psd_max_duration)
        psd_start_time = psd_end_time - psd_duration

        # Add calculated times to configs
        configs['analysis_start_time'] = start_time
        configs['analysis_end_time'] = end_time
        configs['psd_start_time'] = psd_start_time
        configs['psd_end_time'] = psd_end_time
        configs['psd_duration'] = psd_duration
        configs['postevent_start_time'] = end_time
        configs['postevent_end_time'] = end_time + duration

    return configs


def _get_data_files_and_gps_times(det: str = "L1") -> Dict[int, str]:
    search_str = f"/datasets/LIGO/public/gwosc.osgstorage.org/gwdata/O3b/strain.4k/hdf.v1/{det}/*/*.hdf5"
    files = glob.glob(search_str)

    if not files:
        raise FileNotFoundError(f"No HDF5 files found at {search_str}")

    path_dict = {}
    for f in files:
        match = re.search(r"R1-(\d+)-\d+\.hdf5", f)
        if match:
            gps_start = int(match.group(1))
            path_dict[gps_start] = f

    return dict(sorted(path_dict.items()))


def get_data_dicts() -> Dict[str, Dict[int, str]]:
    return {
        "L1": _get_data_files_and_gps_times("L1"),
        "H1": _get_data_files_and_gps_times("H1"),
    }


def get_fnames_for_range(gps_start: float, gps_end: float, det: str = "L1") -> List[str]:
    gps_start = int(gps_start)
    gps_end = int(gps_end)

    gps_files = _get_data_files_and_gps_times(det)
    start_times = sorted(gps_files.keys())

    files = []

    for i in range(len(start_times)):
        t0 = start_times[i]
        t1 = start_times[i + 1] if i + 1 < len(start_times) else float('inf')

        # Check if [gps_start, gps_end] intersects with [t0, t1]
        if gps_end > t0 and gps_start < t1:
            files.append(gps_files[t0])

    return files


def load_strain_segment(gps_start: float, gps_end: float, detector: str = "L1") -> Tuple[np.ndarray, np.ndarray]:
    files = get_fnames_for_range(gps_start, gps_end, detector)
    if not files:
        raise ValueError(f"No files found for {detector} in time range {gps_start}-{gps_end}")

    try:
        # Use GWPy to read the strain data
        strain_ts = TimeSeries.read(files, format='hdf5.gwosc', start=gps_start, end=gps_end)

        # Convert to numpy arrays
        times = strain_ts.times.value  # Get time array as numpy array
        strain = strain_ts.value  # Get strain values as numpy array

        return times, strain

    except Exception as e:
        raise ValueError(f"Could not read strain data for {detector}: {e}")


def read_strain_data(file_paths: List[str], gps_start: float, gps_end: float,
                     detector: str) -> Tuple[np.ndarray, np.ndarray]:
    if not file_paths:
        raise ValueError(f"No files provided for {detector}")

    try:
        strain_ts = TimeSeries.read(file_paths, format='hdf5.gwosc', start=gps_start, end=gps_end)
        return strain_ts.times.value, strain_ts.value

    except Exception as e:
        raise ValueError(f"Could not read strain data for {detector} from {len(file_paths)} files: {e}")


def get_strain_data_for_event(event_configs: Dict, detector: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    result = {}

    # Get analysis data
    if all(key in event_configs for key in ['analysis_start_time', 'analysis_end_time']):
        try:
            analysis_times, analysis_strain = load_strain_segment(
                event_configs['analysis_start_time'],
                event_configs['analysis_end_time'],
                detector
            )
            result['analysis'] = (analysis_times, analysis_strain)
        except Exception as e:
            print(f"Warning: Could not get analysis data for {detector}: {e}")

    # Get post-event data
    if all(key in event_configs for key in ['postevent_start_time', 'postevent_end_time']):
        try:
            postevent_times, postevent_strain = load_strain_segment(
                event_configs['postevent_start_time'],
                event_configs['postevent_end_time'],
                detector
            )
            result['postevent'] = (postevent_times, postevent_strain)
        except Exception as e:
            print(f"Warning: Could not get postevent data for {detector}: {e}")

    # Get psd data
    if all(key in event_configs for key in ['psd_start_time', 'psd_end_time']):
        try:
            psd_times, psd_strain = load_strain_segment(
                event_configs['psd_start_time'],
                event_configs['psd_end_time'],
                detector
            )
            result['psd'] = (psd_times, psd_strain)
        except Exception as e:
            print(f"Warning: Could not get PSD data for {detector}: {e}")

    return result


def get_welch_psd(strain_data: np.ndarray, times: np.ndarray,
                  analysis_duration: float, roll_off: float = 0.4,
                  overlap: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate Welch PSD estimate from strain data.

    Follows bilby_pipe:
    https://lscsoft.docs.ligo.org/bilby_pipe/master/_modules/bilby_pipe/data_generation.html#DataGenerationInput.__generate_psd

    """
    # Create TimeSeries object
    strain_ts = TimeSeries(strain_data, times=times)

    # Calculate Welch PSD
    psd = strain_ts.psd(
        fftlength=analysis_duration,
        overlap=analysis_duration * overlap,
        window=('tukey', roll_off),
        method='median'
    )

    return psd.frequencies.value, psd.value


def get_fd_data(strain_data: np.ndarray, times: np.ndarray, det: str, roll_off: float, fmin: float, fmax: float):
    """Fixed function with correct variable names and parameters."""
    strain_ts = TimeSeries(strain_data, times=times)
    ifo = bilby.gw.detector.get_empty_interferometer(det)
    ifo.strain_data.roll_off = roll_off
    ifo.maximum_frequency = fmax  # Fixed: was f_f
    ifo.minimum_frequency = fmin  # Fixed: was f_i
    ifo.strain_data.set_from_gwpy_timeseries(strain_ts)

    x = ifo.strain_data.frequency_array
    y = ifo.strain_data.frequency_domain_strain
    Ew = np.sqrt(ifo.strain_data.window_factor)

    I = (x >= fmin) & (x <= fmax)  # Fixed: was f_i and f_f
    return x[I], y[I] / Ew


def get_pval(x_data, y_data, x_psd, y_psd):
    # Keep only the data points where the frequencies exactly match
    I_keep = np.isin(x_data, x_psd)

    # Make sure the PSD values are aligned
    common_freqs = x_data[I_keep]
    y_psd_matched = y_psd[np.isin(x_psd, common_freqs)]

    # Compute the normalized ratio
    ratio_real = np.real(y_data[I_keep] / y_psd_matched)
    ratio_imag = np.imag(y_data[I_keep] / y_psd_matched)

    # Combine real and imaginary parts
    ratio_combined = np.concatenate([ratio_real, ratio_imag])

    # Anderson-Darling and KS test
    a2_stat = anderson(ratio_combined).statistic
    pvalue = kstest(ratio_combined, norm.cdf).pvalue

    return pvalue

