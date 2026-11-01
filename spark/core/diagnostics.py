import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def analyze(result, param_names=None):
    """
    Prints a professional summary with the fit results and 
    the statistics from the optimization engine (SciPy).

    Parameters
    ----------
    result : FitResult
        The result object returned by the global fitting engine.
    param_names : list of str, optional
        A list of custom names for the fitted parameters. If None, 
        default names (w, t0, Tau X) will be generated automatically.
    """
    print("="*50)
    print("               SPARK FIT DIAGNOSTICS")
    print("="*50)
    
    # 1. Convergence statistics
    res = result.scipy_res
    print("[Convergence Status]")
    print(f" Message: {res.message}")
    print(f" Success: {res.success}")
    print(f" Cost (Sum of Squares): {res.cost:.4e}")
    print(f" Function Evaluations:  {res.nfev}")
    print("-" * 50)
    
    # 2. Optimized parameters
    print("[Optimized Parameters]")
    if param_names is None:
        # Default names if none are specified
        param_names = ["w (IRF FWHM)", "t0 (Time Zero)"] + [f"Tau {i+1}" for i in range(len(result.x)-2)]
        
    for name, val in zip(param_names, result.x):
        print(f" {name:<15}: {val:>10.4f}")
    print("="*50)


def plot_spectra(dataset, target_times):
    """
    Plots the spectra at the exact (or closest) requested target times.

    Parameters
    ----------
    dataset : SpectroscopyDataset
        The dataset containing the spectral data and time delays.
    target_times : list of float
        List of target times in picoseconds (ps), e.g., [0.1, 1.0, 10.0].
    """
    plt.figure(figsize=(8, 5))
    
    for t in target_times:
        # Find the index of the closest time in the dataset
        idx = np.abs(dataset.td - t).argmin()
        actual_t = dataset.td[idx]
        
        plt.plot(dataset.wl, dataset.data[:, idx], label=f'{actual_t:.2f} ps')
        
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("ΔA")
    plt.title("Transient Spectra")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_kinetics_fit(dataset, result, target_wavelengths, use_symlog=True):
    """
    Plots the experimental data (scatter) and the global fit (line) 
    for the requested wavelengths.

    Parameters
    ----------
    dataset : SpectroscopyDataset
        The original experimental dataset.
    result : FitResult
        The result object containing the fitted matrix.
    target_wavelengths : list of float
        List of target wavelengths in nanometers (nm), e.g., [450, 500, 600].
    use_symlog : bool, optional
        Whether to use a symmetrical logarithmic scale for the time axis. 
        Defaults to True.
    """
    plt.figure(figsize=(8, 5))
    
    for w in target_wavelengths:
        # Find the index of the closest wavelength
        idx = np.abs(dataset.wl - w).argmin()
        actual_w = dataset.wl[idx]
        
        # Experimental points (scatter)
        p = plt.plot(dataset.td, dataset.data[idx, :], 'o', markersize=4, alpha=0.6, label=f'Data {actual_w:.1f} nm')
        color = p[0].get_color()
        
        # Fit line
        plt.plot(dataset.td, result.fitres[idx, :], '-', color=color, linewidth=2, label=f'Fit {actual_w:.1f} nm')
        
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    
    
    if use_symlog:
        plt.xscale('symlog', linthresh=1.0) 
        
    plt.xlabel("Delay Time (ps)")
    plt.ylabel("ΔA")
    plt.title("Kinetics & Global Fit")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_associated_spectra(dataset, result, num_species=None):
    """
    Plots the associated spectra (DAS or SAS) extracted by the VarPro fit.
    Automatically extracts the time constants (taus) for the legend.

    Parameters
    ----------
    dataset : SpectroscopyDataset
        The original experimental dataset.
    result : FitResult
        The result object containing the extracted amplitudes.
    num_species : int, optional
        The number of kinetic species to plot. If None, it is inferred 
        from the number of fitted parameters.
    """
    plt.figure(figsize=(8, 5))
    
    # If the number of species is not passed, deduce it from the parameters
    if num_species is None:
        num_species = len(result.x) - 2 # Subtract w and t0
        
    taus = result.x[2 : 2 + num_species]
    
    # Plot the kinetic species
    for i in range(num_species):
        plt.plot(dataset.wl, result.amplitudes[i, :], linewidth=2, 
                 label=f'Species {i+1} ($\\tau$ = {taus[i]:.2f} ps)')
        
    # If the amplitude matrix has more rows than species, there is a Coherent Artifact
    if result.amplitudes.shape[0] > num_species:
        plt.plot(dataset.wl, result.amplitudes[num_species, :], 
                 linestyle=':', color='gray', label='Coherent Artifact')
        
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Amplitude ($\\Delta$A)")
    plt.title("Associated Spectra (DAS / SAS)")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def plot_single_map(dataset, title="2D Map", use_symlog=True):
    """
    Plots a single 2D contour/heatmap from a SpectroscopyDataset object.
    Ideal for visualizing the raw file or the effect of pre-processing.

    Parameters
    ----------
    dataset : SpectroscopyDataset
        The dataset to visualize.
    title : str, optional
        The title of the plot. Defaults to "2D Map".
    use_symlog : bool, optional
        Whether to use a symmetrical logarithmic scale for the time axis. 
        Defaults to True.
    """
    plt.figure(figsize=(8, 5))
    
    
    max_val = np.nanmax(np.abs(dataset.data))
    norm = mcolors.TwoSlopeNorm(vmin=-max_val, vcenter=0, vmax=max_val)
    
    
    im = plt.pcolormesh(dataset.td, dataset.wl, dataset.data, 
                        cmap='jet', norm=norm, shading='nearest')
    
    if use_symlog:
        plt.gca().set_xscale('symlog', linthresh=1.0)
        
    plt.xlabel("Delay Time (ps)")
    plt.ylabel("Wavelength (nm)")
    plt.title(title)
    plt.colorbar(im, label="Amplitude ($\\Delta$A)")
    plt.tight_layout()
    plt.show()


def plot_fit_maps(dataset, result, use_symlog=True, vmin=None, vmax=None):
    """
    Plots a 1x3 panel comparing: Experimental Data, Global Fit, and Residuals Matrix.

    Parameters
    ----------
    dataset : SpectroscopyDataset
        The original experimental dataset.
    result : FitResult
        The result object containing the fitted data and residuals.
    use_symlog : bool, optional
        Whether to use a symmetrical logarithmic scale for the time axis. 
        Defaults to True.
    vmin : float, optional
        Minimum amplitude value for the color scale. 
    vmax : float, optional
        Maximum amplitude value for the color scale.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    
    # If no limits are given, calculate a symmetric scale relative to the data maximum
    if vmin is None or vmax is None:
        max_val = np.nanmax(np.abs(dataset.data))
        vmin, vmax = -max_val, max_val
        
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    cmap = 'jet' 
    
    # 1. Experimental data
    axes[0].pcolormesh(dataset.td, dataset.wl, dataset.data, cmap=cmap, norm=norm, shading='nearest')
    axes[0].set_title("Processed Data")
    
    # 2. Fitted surface (Fit)
    axes[1].pcolormesh(dataset.td, dataset.wl, result.fitres, cmap=cmap, norm=norm, shading='nearest')
    axes[1].set_title("Global Fit")
    
    # 3. Residuals (Error)
    im = axes[2].pcolormesh(dataset.td, dataset.wl, result.resid, cmap=cmap, norm=norm, shading='nearest')
    axes[2].set_title("Residuals")
    
    for ax in axes:
        if use_symlog:
            ax.set_xscale('symlog', linthresh=1.0)
        ax.set_xlabel("Delay Time (ps)")
        
    axes[0].set_ylabel("Wavelength (nm)")
    
    
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Amplitude ($\\Delta$A)", aspect=30, pad=0.02)
    
    plt.show()