import numpy as np

class SpectroscopyDataset:
    """
    Core container class for ultrafast spectroscopy data.

    This object stores the data matrix alongside its corresponding time 
    and wavelength axes. It provides GUI-agnostic methods to manipulate, 
    normalize, and process the dataset directly.

    Parameters
    ----------
    data_matrix : array_like
        2D array representing the spectroscopy signal (Z axis).
    wl : array_like
        1D array containing the wavelength values (X axis).
    td : array_like
        1D array containing the time delay values (Y axis).

    Attributes
    ----------
    data : ndarray
        The processed 2D data matrix.
    wl : ndarray
        The 1D wavelength axis.
    td : ndarray
        The 1D time delay axis.
    native_wl_unit : str
        The original spectral unit of the loaded data (defaults to "nm").
    native_time_unit : str
        The original temporal unit of the loaded data (defaults to "ps").
    """
    def __init__(self, data_matrix, wl, td):
        self.data = np.array(data_matrix, dtype=float)
        self.wl = np.array(wl, dtype=float)
        self.td = np.array(td, dtype=float)
        
        # SPARK asume por defecto estas unidades a menos que el usuario indique lo contrario
        self.native_wl_unit = "nm"
        self.native_time_unit = "ps"

    def copy(self):
        """
        Creates a deep, independent copy of the dataset.

        Returns
        -------
        SpectroscopyDataset
            A new instance with copied arrays and inherited units.
        """
        new_ds = SpectroscopyDataset(self.data.copy(), self.wl.copy(), self.td.copy())
        # Heredamos las unidades nativas para que no se pierdan al hacer copias de seguridad
        new_ds.native_wl_unit = getattr(self, 'native_wl_unit', 'nm')
        new_ds.native_time_unit = getattr(self, 'native_time_unit', 'ps')
        return new_ds

    def apply_baseline_correction(self, n_pts):
        """
        Subtracts the average of the first 'n_pts' time points from the entire matrix.

        Parameters
        ----------
        n_pts : int
            Number of initial time points to use for calculating the baseline average.
        """
        if n_pts > 0 and self.data.shape[1] >= n_pts:
            baseline = np.mean(self.data[:, :n_pts], axis=1, keepdims=True)
            self.data -= baseline

    def normalize(self):
        """
        Normalizes the entire data matrix so its maximum absolute value equals 1.
        """
        max_abs_val = np.nanmax(np.abs(self.data))
        if max_abs_val != 0:
            self.data /= max_abs_val

    def crop_time(self, t_min, t_max):
        """
        Truncates the time axis and the data matrix to a specified time range.

        Parameters
        ----------
        t_min : float
            Minimum time delay value to keep.
        t_max : float
            Maximum time delay value to keep.
        """
        mask = (self.td >= t_min) & (self.td <= t_max)
        if np.any(mask):
            self.td = self.td[mask]
            self.data = self.data[:, mask]

    def crop_wavelength(self, wl_min, wl_max):
        """
        Truncates the wavelength axis and data matrix to a specified spectral range.

        Parameters
        ----------
        wl_min : float
            Minimum wavelength value to keep.
        wl_max : float
            Maximum wavelength value to keep.
        """
        mask = (self.wl >= wl_min) & (self.wl <= wl_max)
        if np.any(mask):
            self.wl = self.wl[mask]
            self.data = self.data[mask, :]
            
    def get_shape(self):
        """
        Retrieves the current dimensions of the data matrix.

        Returns
        -------
        tuple
            Shape of the data matrix as (n_wavelengths, n_time_delays).
        """
        return self.data.shape

    def exclude_wavelength_ranges(self, exclude_str):
        """
        Removes specific wavelength bands from the dataset.

        Parameters
        ----------
        exclude_str : str
            A comma-separated list of ranges to exclude (e.g., '490-540, 600-615').
        """
        if not exclude_str: return
        mask_exclude = np.zeros_like(self.wl, dtype=bool)
        ranges = exclude_str.split(',')
        for r in ranges:
            try:
                parts = r.split('-')
                if len(parts) == 2:
                    c_min, c_max = float(parts[0].strip()), float(parts[1].strip())
                    mask_exclude |= (self.wl >= min(c_min, c_max)) & (self.wl <= max(c_min, c_max))
            except ValueError:
                pass
        self.wl = self.wl[~mask_exclude]
        self.data = self.data[~mask_exclude, :]
        
    def zero_negative_times(self):
        """
        Sets all signal intensities at negative time delays to zero.
        Useful for eliminating pre-pulse noise artifacts.
        """
        mask_neg = self.td < 0
        if np.any(mask_neg):
            self.data[:, mask_neg] = 0.0
            
    def apply_binning(self, b_size):
        """
        Averages adjacent spectral channels (wavelengths) to reduce noise.

        Parameters
        ----------
        b_size : int
            Number of adjacent wavelength points to bin together.
        """
        if b_size > 1:
            n_wl = self.data.shape[0]
            new_len = n_wl // b_size
            if new_len > 0:
                self.data = self.data[:new_len*b_size, :]
                self.data = self.data.reshape(new_len, b_size, self.data.shape[1]).mean(axis=1)
                self.wl = self.wl[:new_len*b_size]
                self.wl = self.wl.reshape(new_len, b_size).mean(axis=1)

    def apply_time_binning(self, n_pts):
        """
        Averages adjacent time points along the delay axis.

        Groups every 'n_pts' adjacent time points and calculates their mean, 
        downsampling the temporal resolution.

        Parameters
        ----------
        n_pts : int
            Number of adjacent time points to bin together.
        """
        if n_pts <= 1:
            return
            
        num_wl, num_td = self.data.shape
        num_bins = num_td // n_pts
        
        if num_bins == 0:
            print("Warning: Binning size is larger than the number of time points.")
            return
            
        trunc_length = num_bins * n_pts
        
        # 1. Truncate excess data and reshape
        # Changes shape from (WL, TD) -> (WL, num_bins, n_pts) and averages the last axis
        self.data = self.data[:, :trunc_length].reshape(num_wl, num_bins, n_pts).mean(axis=2)
        
        # 2. Apply equivalent reshape and mean to time delay array
        self.td = self.td[:trunc_length].reshape(num_bins, n_pts).mean(axis=1)