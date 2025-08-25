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
from .utils import *


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

    def __repr__(self):
        return f"<GWEventData.{self.event_name}>"

