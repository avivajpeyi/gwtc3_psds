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
import numpy as np
import matplotlib.pyplot as plt
import bilby
from gwosc.datasets import event_gps
from gwpy.timeseries import TimeSeries
import h5py
from scipy.stats import anderson, kstest, norm
import os
import h5py
import urllib.request


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


class GWEventData:
    """Class to store and manage gravitational wave event data."""

    def __init__(self, event_name: str):
        self.event_name = event_name
        self.configs = {}
        self.psds = {}  # {detector: {'freqs': array, 'psd': array}}
        self.strain_data = {}  # {detector: {'analysis': (times, strain), 'psd': (times, strain)}}
        self.welch_psds = {}  # {detector: {'freqs': array, 'psd': array}} - computed from strain
        self.postevent_fd = {}  # {detector: {'freqs': array, 'data_fd': array}}
        self.analysis_group = None
        self.fmin = {}  # detector-specific minimum frequencies
        self.fmax = {}  # detector-specific maximum frequencies

    @classmethod
    def from_ozstar(cls, event_name: str):  # Fixed: removed extra parameters
        """Load event data from OZSTAR GWOSC files."""
        instance = cls(event_name)

        # Get PE file path
        pe_paths = get_pe_paths()
        if event_name not in pe_paths:
            raise ValueError(f"Event {event_name} not found in PE data paths")

        pe_file_path = pe_paths[event_name]

        # Load configs
        instance.configs = get_gw_event_configs(pe_file_path)
        instance.analysis_group = instance.configs.get('analysis_group')

        # Extract and convert config values properly - keep detector-specific values as dicts
        def get_config_value(key):
            """Extract config value, preserving detector dictionaries."""
            value = instance.configs.get(key)
            if value is None:
                raise KeyError(f"Config key '{key}' not found")

            # If it's already a dictionary, return it
            if isinstance(value, dict):
                return value
            # If it's a single value, convert to float
            else:
                return float(instance._extract_scalar(value))

        fmin = get_config_value('minimum-frequency')  # Keep as dict
        fmax = get_config_value('maximum-frequency')  # Keep as dict
        analysis_duration = get_config_value('duration')
        overlap = get_config_value('psd-fractional-overlap')
        roll_off = get_config_value('tukey-roll-off')

        # Store the frequency dicts in the instance for later use
        instance.fmin = fmin
        instance.fmax = fmax

        # Load PSDs from PE file
        with h5py.File(pe_file_path, "r") as fin:
            analysis_group = instance.analysis_group or find_analysis_group(fin)
            psd_data = fin[analysis_group]["psds"]

            for ifo_name in psd_data:
                if ifo_name in ['H1', 'L1']:  # Only load H1 and L1
                    freqs = psd_data[ifo_name][:, 0]
                    vals = psd_data[ifo_name][:, 1]

                    # Apply frequency mask using detector-specific limits
                    detector_fmin = float(instance._extract_scalar(fmin.get(ifo_name, fmin.get('H1', 20.0))))
                    detector_fmax = float(instance._extract_scalar(fmax.get(ifo_name, fmax.get('H1', 2048.0))))

                    mask = (freqs >= detector_fmin) & (freqs <= detector_fmax)
                    freqs, vals = freqs[mask], vals[mask]

                    instance.psds[ifo_name] = {
                        'freqs': freqs,
                        'psd': vals
                    }

        # Load strain data
        print(f"Loading strain data for {event_name}...")
        for detector in ['H1', 'L1']:
            if detector in instance.psds:  # Only load for detectors we have PSDs for
                try:
                    strain_data = get_strain_data_for_event(instance.configs, detector)
                    if strain_data:
                        instance.strain_data[detector] = strain_data
                        print(f"  {detector}: Loaded {len(strain_data)} data segments")

                        # Calculate Welch PSD from strain data
                        if 'psd' in strain_data:
                            psd_times, psd_strain = strain_data['psd']

                            # Compute Welch PSD
                            welch_freqs, welch_psd = get_welch_psd(
                                psd_strain, psd_times, analysis_duration, roll_off, overlap
                            )

                            # Apply frequency mask using detector-specific limits
                            detector_fmin = float(instance._extract_scalar(fmin.get(detector, fmin.get('H1', 20.0))))
                            detector_fmax = float(instance._extract_scalar(fmax.get(detector, fmax.get('H1', 2048.0))))

                            mask = (welch_freqs >= detector_fmin) & (welch_freqs <= detector_fmax)
                            welch_freqs, welch_psd = welch_freqs[mask], welch_psd[mask]

                            instance.welch_psds[detector] = {
                                'freqs': welch_freqs,
                                'psd': welch_psd
                            }

                        # get postevent data - Fixed: use 'postevent' instead of 'psd'
                        if 'postevent' in strain_data:
                            postevent_times, postevent_strain = strain_data['postevent']  # Fixed: was using 'psd'

                            # Get detector-specific frequency limits
                            detector_fmin = float(instance._extract_scalar(fmin.get(detector, fmin.get('H1', 20.0))))
                            detector_fmax = float(instance._extract_scalar(fmax.get(detector, fmax.get('H1', 2048.0))))

                            postevent_freqs, postevent_fd = get_fd_data(  # Fixed: was postevnet_fd
                                postevent_strain, postevent_times, detector, roll_off, detector_fmin, detector_fmax
                            )
                            instance.postevent_fd[detector] = {
                                'freqs': postevent_freqs,
                                'datafd': postevent_fd  # Fixed: was using different key name
                            }

                except Exception as e:
                    print(f"  Warning: Could not load strain data for {detector}: {e}")

        return instance

    def _extract_scalar(self, value):
        """Extract scalar value from numpy array or return as-is if already scalar."""
        # Handle bytes that need to be decoded and evaluated
        if isinstance(value, bytes):
            try:
                # Decode bytes to string
                value_str = value.decode('utf-8')
                # Try to evaluate as Python literal (dict, list, etc.)
                value = ast.literal_eval(value_str)
            except (UnicodeDecodeError, ValueError, SyntaxError):
                # If decoding/evaluation fails, return as string
                return value.decode('utf-8') if isinstance(value, bytes) else value

        # Handle string representations of Python objects
        if isinstance(value, str):
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                # If evaluation fails, return as string
                return value

        # Extract scalar from numpy arrays or single-element sequences
        if hasattr(value, 'item'):  # numpy array
            return value.item()
        elif hasattr(value, '__len__') and len(value) == 1:  # single-element sequence
            return value[0]
        else:
            return value

    def to_hdf5(self, output_dir: str = ".", filename: str = None):
        """Save event data to HDF5 file."""
        if filename is None:
            filename = f"{self.event_name}_data.h5"

        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)

        with h5py.File(filepath, "w") as f:
            # Save metadata
            f.attrs['event_name'] = self.event_name
            f.attrs['analysis_group'] = self.analysis_group or 'unknown'

            # Save configs
            config_group = f.create_group("configs")
            for key, value in self.configs.items():
                try:
                    if isinstance(value, (str, bytes)):
                        config_group.create_dataset(key, data=str(value))
                    elif isinstance(value, (int, float, np.number)):
                        config_group.create_dataset(key, data=value)
                    elif isinstance(value, (list, tuple, np.ndarray)):
                        config_group.create_dataset(key, data=value)
                    elif isinstance(value, dict):
                        config_group.create_dataset(key, data=str(value))
                    else:
                        config_group.create_dataset(key, data=str(value))
                except Exception as e:
                    print(f"Warning: Could not save config '{key}': {e}")

            # Save PSDs (from PE file)
            psd_group = f.create_group("psds")
            for detector, psd_data in self.psds.items():
                det_group = psd_group.create_group(detector)
                det_group.create_dataset("freqs", data=psd_data['freqs'])
                det_group.create_dataset("psd", data=psd_data['psd'])

            # Save Welch PSDs (computed from strain data)
            if self.welch_psds:
                welch_group = f.create_group("welch_psds")
                for detector, psd_data in self.welch_psds.items():
                    det_group = welch_group.create_group(detector)
                    det_group.create_dataset("freqs", data=psd_data['freqs'])
                    det_group.create_dataset("psd", data=psd_data['psd'])

            # Save postevent frequency domain data
            if self.postevent_fd:
                fd_group = f.create_group("postevent_fd")
                for detector, fd_data in self.postevent_fd.items():
                    det_group = fd_group.create_group(detector)
                    det_group.create_dataset("freqs", data=fd_data['freqs'])
                    det_group.create_dataset("datafd", data=fd_data['datafd'])

            # Save strain data
            if self.strain_data:
                strain_group = f.create_group("strain_data")
                for detector, data_dict in self.strain_data.items():
                    det_group = strain_group.create_group(detector)
                    for data_type, (times, strain) in data_dict.items():
                        type_group = det_group.create_group(data_type)
                        type_group.create_dataset("times", data=times)
                        type_group.create_dataset("strain", data=strain)
                        # Save metadata
                        type_group.attrs['duration'] = times[-1] - times[0]
                        type_group.attrs['sample_rate'] = len(times) / (times[-1] - times[0])
                        type_group.attrs['start_time'] = times[0]
                        type_group.attrs['end_time'] = times[-1]

        return filepath

    @classmethod
    def from_hdf5(cls, filepath: str):
        """Load event data from HDF5 file."""
        with h5py.File(filepath, "r") as f:
            event_name = f.attrs['event_name']
            if isinstance(event_name, bytes):
                event_name = event_name.decode('utf-8')

            instance = cls(event_name)
            instance.analysis_group = f.attrs.get('analysis_group', 'unknown')
            if isinstance(instance.analysis_group, bytes):
                instance.analysis_group = instance.analysis_group.decode('utf-8')

            # Load configs
            config_group = f["configs"]
            for key in config_group.keys():
                data = config_group[key][()]
                if isinstance(data, bytes):
                    data = data.decode('utf-8')
                instance.configs[key] = data

            # Load PSDs
            psd_group = f["psds"]
            for detector in psd_group.keys():
                det_group = psd_group[detector]
                instance.psds[detector] = {
                    'freqs': det_group["freqs"][:],
                    'psd': det_group["psd"][:]
                }

            # Load Welch PSDs if present
            if "welch_psds" in f:
                welch_group = f["welch_psds"]
                for detector in welch_group.keys():
                    det_group = welch_group[detector]
                    instance.welch_psds[detector] = {
                        'freqs': det_group["freqs"][:],
                        'psd': det_group["psd"][:]
                    }

            # Load postevent frequency domain data if present
            if "postevent_fd" in f:
                fd_group = f["postevent_fd"]
                for detector in fd_group.keys():
                    det_group = fd_group[detector]
                    instance.postevent_fd[detector] = {
                        'freqs': det_group["freqs"][:],
                        'datafd': det_group["datafd"][:]
                    }

            # Load strain data if present
            if "strain_data" in f:
                strain_group = f["strain_data"]
                for detector in strain_group.keys():
                    det_group = strain_group[detector]
                    instance.strain_data[detector] = {}
                    for data_type in det_group.keys():
                        type_group = det_group[data_type]
                        times = type_group["times"][:]
                        strain = type_group["strain"][:]
                        instance.strain_data[detector][data_type] = (times, strain)

        return instance

    def plot_psds(self, output_dir: str = "."):
        """Plot PSDs for this event, optionally comparing PE PSDs with Welch estimates."""
        if not self.psds:
            print("No PSD data to plot")
            return

        # Create subplots for each detector
        n_detectors = len(self.psds)
        fig, axes = plt.subplots(n_detectors, 1, figsize=(10, 4 * n_detectors),
                                 sharex=True, squeeze=False)
        axes = axes.flatten()

        colors = {'H1': 'red', 'L1': 'blue'}

        for i, detector in enumerate(sorted(self.psds.keys())):
            ax = axes[i]
            psd_data = self.psds[detector]

            # Only proceed if we have the required data
            if detector not in self.welch_psds or detector not in self.postevent_fd:
                print(f"Warning: Missing data for {detector}, skipping plot")
                continue

            welch_data = self.welch_psds[detector]
            fd_data = self.postevent_fd[detector]

            # Calculate p-values
            pval_gwtc = get_pval(fd_data['freqs'], fd_data['datafd'], psd_data['freqs'], np.sqrt(psd_data['psd']))
            pval_welch = get_pval(fd_data['freqs'], fd_data['datafd'], welch_data['freqs'], np.sqrt(welch_data['psd']))

            # plot postevent FD data 
            ax.loglog(fd_data['freqs'], np.abs(fd_data['datafd']) ** 2, color='lightgray', alpha=0.4,
                      label='Postevent Data')

            # Plot GWTC PSD
            ax.loglog(psd_data['freqs'], psd_data['psd'],
                      color=colors[detector], linewidth=1.5,
                      label=f'GWTC PSD (pval={pval_gwtc:.3f})')

            # Plot Welch PSD
            ax.loglog(welch_data['freqs'], welch_data['psd'],
                      color=colors[detector], linewidth=2, linestyle='--', alpha=0.3,
                      label=f'Welch PSD (pval={pval_welch:.3f})')

            ax.set_ylabel("PSD [strain²/Hz]", fontsize=12)
            ax.set_title(f"{detector} Power Spectral Density", fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

            # Set reasonable y-axis limits
            if len(psd_data['psd']) > 0:
                ymin = np.min(psd_data['psd']) * 0.1
                ymax = np.max(psd_data['psd']) * 10
                ax.set_ylim(ymin, ymax)

        # Set x-axis label only on bottom subplot
        axes[-1].set_xlabel("Frequency [Hz]", fontsize=12)

        # Overall title
        fig.suptitle(f"{self.event_name} Power Spectral Densities", fontsize=14, y=0.98)
        plt.tight_layout()

        # Save plot
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, f"{self.event_name}_psd.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"PSD plot saved to: {plot_path}")

        plt.close()


def process_and_save_event(event_name: str, output_dir: str = "event_data") -> str:  # Fixed: removed extra parameters
    """Process a single event and save to HDF5."""
    try:
        # Load from OZSTAR - Fixed: removed extra parameters
        event_data = GWEventData.from_ozstar(event_name)

        # Save to HDF5
        filepath = event_data.to_hdf5(output_dir)

        # Generate plots
        event_data.plot_psds(output_dir)

        # Only plot strain if we have the data and the method exists
        if event_data.strain_data and hasattr(event_data, 'plot_strain'):
            event_data.plot_strain('analysis', output_dir, time_window=10.0)  # 10s window
            event_data.plot_strain('psd', output_dir)

        print(f"Successfully processed {event_name}")
        print(f"Data saved to: {filepath}")
        print("-" * 50)

        return filepath

    except Exception as e:
        print(f"Error processing {event_name}: {e}")
        return None


def process_all_events(output_dir: str = "event_data"):  # Fixed: removed extra parameters
    """Process all available events and save to individual HDF5 files."""
    pe_paths = get_pe_paths()

    print(f"Found {len(pe_paths)} events to process")
    print(f"Output directory: {output_dir}")
    print("=" * 50)

    successful = []
    failed = []

    for event_name in tqdm(pe_paths.keys(), desc="Processing events"):
        filepath = process_and_save_event(event_name, output_dir)  # Fixed: removed extra parameters
        if filepath:
            successful.append(event_name)
        else:
            failed.append(event_name)

    print(f"\nProcessing complete:")
    print(f"Successful: {len(successful)} events")
    print(f"Failed: {len(failed)} events")

    if failed:
        print(f"Failed events: {failed}")

    return successful, failed


def main():
    """Main processing function with examples."""
    # Initialize paths
    pe_paths = get_pe_paths()
    data_paths = get_data_dicts()

    print(f"Found {len(pe_paths)} PE data files")
    print(f"Found {len(data_paths['L1'])} L1 strain files")
    print(f"Found {len(data_paths['H1'])} H1 strain files")

    # Example 1: Process a single event with strain data
    if pe_paths:
        first_event = list(pe_paths.keys())[0]
        print(f"\n=== Example 1: Processing single event with strain data ===")
        print(f"Processing: {first_event}")

        try:
            # Load from OZSTAR with strain data - Fixed: removed extra parameters
            event_data = GWEventData.from_ozstar(first_event)

            # Save to HDF5
            filepath = event_data.to_hdf5("example_output")
            print(f"Saved to: {filepath}")

            # Generate plots
            event_data.plot_psds("example_output")

            # Test loading from HDF5
            print(f"\n=== Example 2: Loading from HDF5 ===")
            loaded_data = GWEventData.from_hdf5(filepath)
            print(f"Loaded: {loaded_data}")

        except Exception as e:
            print(f"Error in examples: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n=== Processing all events ===")
    successful, failed = process_all_events()


if __name__ == "__main__":
    main()