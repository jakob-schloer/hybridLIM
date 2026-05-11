"""
Retrieve CERA-20C data from the ECMWF archive

@author: Jakob Schloer
"""

import argparse
import os
import subprocess
import tempfile
from typing import Any
from typing import Dict

import numpy as np
import pandas as pd
import xarray as xr


def create_mars_request_file(request: Dict[str, Any], target_file: str) -> str:
    """Creates a Mars request file from a dictionary of parameters.

    Args:
        request (dict): The Mars request parameters.
        target_file (str): The target output file.

    Returns:
        str: The path to the temporary Mars request file.
    """
    # Create temporary file for Mars request
    fd, mars_file = tempfile.mkstemp(suffix=".mars", text=True)

    with os.fdopen(fd, "w") as f:
        f.write("RETRIEVE,\n")

        for key, value in request.items():
            if isinstance(value, list):
                # Convert list to comma-separated string
                if all(isinstance(v, (int, float)) for v in value):
                    value_str = "/".join(map(str, value))
                else:
                    value_str = "/".join(map(str, value))
            else:
                value_str = str(value)

            # Convert key to uppercase for Mars format
            mars_key = key.upper()
            f.write(f"    {mars_key} = {value_str},\n")

        f.write(f"    TARGET = '{target_file}'")

    return mars_file


def retrieve_cera20c(
    date: str,
    variables: list[str],
    number: int,
    output_directory: str,
    overwrite: bool = False,
) -> str:
    """Retrieve forecast data from ECMWF service.

    Args:
        date (str): Initialization date in YYYY-MM-DD format.
        variables (list[str]): List of variables to retrieve.
        number (int): Number of the ensemble member.
        output_directory (str): Output directory path.
        overwrite (bool): Whether to overwrite existing files.

    Returns:
        str: Path to the retrieved forecast file.
    """
    # Supported variables: sst, ssh
    var_to_id = {
        "sst": "151189",
        "ssh": "151145",
    }
    assert all(
        var in var_to_id for var in variables
    ), f"Variables {variables} not supported. Supported variables: {list(var_to_id.keys())}"

    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Variable to parameter id
    param_ids = []
    for var in variables:
        if var not in var_to_id:
            print(f"Variable {var} not supported and will be skipped. Supported variables: {list(var_to_id.keys())}")
            continue
        param_ids.append(var_to_id[var])

    # Define the request parameters
    request = {
        "class": "ep",
        "stream": "edmo",
        "type": "an",
        "expver": 1,
        "levtype": "o2d",
        "date": pd.to_datetime(date).strftime("%Y%m%d"),
        "param": param_ids,
        "number": number,
    }

    # Execute the request
    filename = os.path.join(output_directory, f"cera20c_{date}_member_{number}.nc")

    if not os.path.exists(filename) or overwrite:
        print(f"Retrieving forecast for date={date}.", flush=True)

        # Create Mars request file
        mars_file = create_mars_request_file(request, filename)

        try:
            # Execute mars command
            subprocess.run(["mars", mars_file], check=True, capture_output=True, text=True)
            print(f"Mars request completed successfully for {filename}", flush=True)

        except subprocess.CalledProcessError as e:
            print(f"Error executing mars request or conversion: {e}", flush=True)
            print(f"Command stdout: {e.stdout}", flush=True)
            print(f"Command stderr: {e.stderr}", flush=True)
            raise

    else:
        print(f"File {filename} already exists, skipping retrieval.", flush=True)

    return filename


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="/scratch/ecm1922/weather-quest/ifs/fc_probabilities/")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    # args = arg_parser()
    # For testing
    args = argparse.Namespace(
        output_dir="/scratch/ecm1922/hybridLIM/data/cera20c/",
        overwrite=True,
    )

    dates = np.arange("2000-01", "2011-01", dtype="datetime64[M]")
    overwrite = args.overwrite
    for date in dates:
        date = str(np.datetime_as_string(date, unit="D"))
        print(f"Retrieving CERA-20C for date={date}...", flush=True)
        for variable in ["sst", "ssh"]:
            output_dir = os.path.join(args.output_dir, variable)
            for number in range(10):
                filename = retrieve_cera20c(date, [variable], number, output_dir, overwrite=overwrite)

                # Rename coordinates
                ds = xr.open_dataset(filename)
                break
            break
        break


if __name__ == "__main__":
    main()
