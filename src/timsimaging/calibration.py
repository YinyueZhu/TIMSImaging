import struct
from pyteomics.mass import calculate_mass
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd


# tuning mix CCS reference
# Stow, S. M., et al. (2017) Anal Chem 89(17): 9048-9055.

ccs_ref = {
    "positive": dict(
        formula=[
            "C5H12NO2",
            "C6H19N3O6P3",
            "C12H19F12N3O6P3",
            "C18H19F24N3O6P3",
            "C24H19F36N3O6P3",
            "C30H19F48N3O6P3",
            "C36H19F60N3O6P3",
            "C42H19F72N3O6P3",
            "C48H19F84N3O6P3",
            "C54H19F96N3O6P3",
        ],
        mass=[
            118.086255,
            322.048121,
            622.028960,
            922.009798,
            1221.990637,
            1521.971475,
            1821.952313,
            2121.933152,
            2421.913992,
            2721.894829,
        ],
        ccs=[
            121.30,
            153.73,
            202.96,
            243.64,
            282.20,
            316.96,
            351.25,
            383.03,
            412.96,
            441.21,
        ],
    ),
    "negative": dict(
        formula=[
            "C2F3O2",
            "C6HF9N3O",
            "C12HF21N3O",
            "C20H18F27N3O8P3",
            "C26H18F39N3O8P3",
            "C32H18F51N3O8P3",
            "C38H18F63N3O8P3",
            "C44H18F75N3O8P3",
            "C50H18F87N3O8P3",
            "C56H18F99N3O8P3",
        ],
        mass=[
            112.985587,
            301.998139,
            601.978977,
            1033.988109,
            1333.968947,
            1633.949786,
            1933.930624,
            2233.911463,
            2533.892301,
            2833.873139,
        ],
        ccs=[
            108.23,
            140.04,
            180.77,
            255.34,
            284.76,
            319.03,
            352.55,
            380.74,
            412.99,
            432.62,
        ],
    ),
}

buffer_mass = 28.013406

# A simple linear model
class CCS_calibration:
    def __init__(self, calibrants, raw_mob, polarity="+", buffer_mass=buffer_mass):
        if polarity == "+":
            df_ref = pd.DataFrame(ccs_ref["positive"])
        elif polarity == "-":
            df_ref = pd.DataFrame(ccs_ref["negative"])
        df_ref.set_index("formula", inplace=True)
        try:
            self.ref = df_ref.loc[calibrants]
        except:
            raise KeyError("Unknown calibrant!")
        self.buffer_mass = buffer_mass
        self.model = LinearRegression()
        ref_mass = self.ref["mass"].to_numpy()
        ref_ccs = self.ref["ccs"].to_numpy()
        reduced_mass = (1 / ref_mass + 1 / self.buffer_mass) ** 0.5
        X = np.array(raw_mob) * reduced_mass
        self.model.fit(X.reshape(-1, 1), ref_ccs)

    def transform(self, mz, mobility, charge=1):
        mass = np.array(mz)*np.array(charge)
        reduced_mass = (1/mass+1/self.buffer_mass)**0.5
        X = np.array(mobility)*reduced_mass
        return self.model.predict(X.reshape(-1,1))

# Bruker's internal calibration
import os
import ctypes
from alphatims.bruker import BRUKER_DLL_FILE_NAME

class CCS_Bruker_Calibration:
    def __init__(self) -> None:
        bruker_dll = ctypes.cdll.LoadLibrary(os.path.realpath(BRUKER_DLL_FILE_NAME))  # or libtimsdata.so on Linux

        bruker_dll.tims_oneoverk0_to_ccs_for_mz.argtypes = [ctypes.c_double, ctypes.c_int, ctypes.c_double] #1/K_0, charge, mz
        bruker_dll.tims_oneoverk0_to_ccs_for_mz.restype = ctypes.c_double
        self.handle = bruker_dll
    def transform(self, mz, mobility, charge=1):
        mz = np.array(mz)
        mobility = np.array(mobility)
        result = np.empty(len(mz), dtype=np.float64)
        for i in range(len(mz)):
            result[i] = self.handle.tims_oneoverk0_to_ccs_for_mz(mobility[i], charge, mz[i])
        return result

