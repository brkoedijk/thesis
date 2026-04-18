# parameters.py
import numpy as np
from pathlib import Path

# Load the result of your calibration script
_PARAMS_FILE = Path(__file__).resolve().parent / "nwe_wind_params.npz"
calib_data = np.load(_PARAMS_FILE)

DEFAULT_A_MATRIX = calib_data["A"]
DEFAULT_A_AGG = calib_data["A_agg"]
DEFAULT_SIGMA_MATRIX = calib_data["sigma"]
DEFAULT_SIGMA_AGG = calib_data["sigma_agg"]
