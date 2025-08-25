from tqdm.auto import tqdm
from .gweventdata import GWEventData
from .utils import get_pe_paths, get_data_dicts
import os

HERE = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = f"{HERE}/event_data"

def process_and_save_event(event_name: str, output_dir: str = DATA_DIR) -> str:
    """Process a single event and save to HDF5."""
    try:
        event_data = GWEventData.from_ozstar(event_name)
        filepath = event_data.to_hdf5(output_dir)
        event_data.plot_psds(output_dir)
        print(f"Successfully processed {event_name}")
        print(f"Data saved to: {filepath}")
        print("-" * 50)

        return filepath

    except Exception as e:
        print(f"Error processing {event_name}: {e}")
        return None


def process_all_events(output_dir: str = DATA_DIR):  # Fixed: removed extra parameters
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
            filepath = event_data.to_hdf5(DATA_DIR)
            print(f"Saved to: {filepath}")

            # Generate plots
            event_data.plot_psds(DATA_DIR)

            # Test loading from HDF5
            print(f"\n=== Example 2: Loading from HDF5 ===")
            loaded_data = GWEventData.from_hdf5(filepath)
            print(f"Loaded: {loaded_data}")

        except Exception as e:
            print(f"Error in examples: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n=== Processing all events ===")
    process_all_events()

