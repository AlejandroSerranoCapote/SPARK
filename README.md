

<div align="center">
            
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyQt5](https://img.shields.io/badge/GUI-PyQt5-41CD52?logo=qt&logoColor=white)](https://pypi.org/project/PyQt5/)
[![Platform](https://img.shields.io/badge/Platform-Windows%20|%20Linux-lightgrey)](#)
[![Docs](https://img.shields.io/badge/Docs-ReadTheDocs-blue)](https://ultrafast-spectroscopy-analyzer.readthedocs.io/en/latest/index.html)

# Spectroscopy Pump-probe Analysis and Research Kit (SPARK)

<img width="2049" height="1000" alt="CabeceraGithub" src="https://github.com/user-attachments/assets/56e8a8fb-b4f4-4760-9299-17787bd8baf3" />



**A comprehensive, open-source GUI suite for the processing and global target analysis of ultrafast spectroscopy data.**


<br/>



</div>

## Installation

**Requirements:** Python 3.8 or later.

### Easy installation via .bat files

The fastest way to get started on Windows is using the included batch scripts:

* **Step 1:** Execute `Install.bat`
* **Step 2:** Execute `Run.bat`
  
### Installation via cmd
```bash
# 1. Clone the repository
git clone https://github.com/AlejandroSerranoCapote/SPARK.git
cd SPARK

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch
python SPARK.py
```

### Build a standalone executable (Windows)

```bash
pyinstaller --onefile --noconsole --icon=icon.ico \
            --exclude-module PyQt6 \
            "SPARK.py"
```

The `.exe` will appear in the `dist/` folder with no Python installation required on the target machine.

---

## Table of Contents

- [Overview](#overview)
- [Supported Data Formats & Generated Files](#supported-data-formats--generated-files)
- [Supported Techniques](#supported-techniques)
- [Modules](#modules)
- [Mathematical Background](#mathematical-background)
- [Error Estimation](#error-estimation)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)


---

## Overview

**SPARK** is a full-featured desktop application for the processing, visualization, and global target analysis of ultrafast spectroscopy data. It combines a rigorous mathematical engine — built around Variable Projection (VarPro), Bateman-equation population dynamics, and profile-likelihood confidence intervals — with an interactive graphical interface designed to take raw experimental data all the way to publication-quality figures without requiring any scripting.

The core fitting engine implements a **true VarPro algorithm**: at each step of the non-linear optimizer, the spectral amplitudes are eliminated analytically via linear least squares, so the optimizer only ever searches over the physically meaningful non-linear parameters ($w$, $t_0$, and the lifetimes $\tau_i$). This dramatically reduces the dimensionality of the non-linear search space and improves both convergence speed and robustness compared to naive full-parameter optimization.

<img width="896" height="557" alt="image" src="https://github.com/user-attachments/assets/1fda0ce0-ae71-4a0d-b530-4358b772608d" />

---

## Supported Data Formats & Generated Files

This section describes the file formats SPARK can import, how the Universal Importer handles them, and the directory structure of the exported results.

### Input Data & The Universal Importer

SPARK features a highly flexible **Universal Data Importer** (`import_wizard`) designed to read exported data from almost any spectrometer. It accepts the following file extensions: **`.csv`, `.txt`, `.dat`, and `.npy`**.

#### The Standard Matrix Layout
For automatic, seamless importing, your text/CSV files should ideally be structured as a 2D matrix where:
*   **Row 0 (Header):** Time delays (ps).
*   **Column 0 (Index):** Wavelengths (nm).
*   **Remaining Cells:** The transient absorption or emission data ($\Delta A$).
   
```math
\text{ΔA(λ,t)} =
\begin{bmatrix}
λ \setminus t & -1.00 & -0.50 & 0.00 & 0.50 & 1.00 \\
400 & 0.002 & 0.005 & 0.010 & 0.004 & 0.001 \\
410 & 0.001 & 0.004 & 0.008 & 0.003 & 0.000
\end{bmatrix}
```
*(Note: The top-left cell can be empty, `0`, or text).*

#### The Smart Fallback GUI
If your file contains complex headers, metadata footers (e.g., date, sample name, pump energy), or uses non-standard delimiters, **the software will not crash**. 
Instead, it will automatically launch the **Manual Data Importer**. This Origin-style spreadsheet interface allows you to visually map your data:
1.  Click **"Auto-Detect Standard Layout"** to automatically isolate the numeric matrix from any surrounding text.
2.  Alternatively, manually highlight columns/rows and assign them to the Wavelength (WL), Time Delay (TD), and Data axes.
3.  A smart validation system ensures dimensions match before allowing the import.

*For advanced users: SPARK also natively reads NumPy dictionaries (`.npy`) containing the exact keys `data_c`, `WL`, and `TD`.*

---

### Output Data & Generated Files

When you execute a Global Fit or export plots, SPARK generates a structured results folder (e.g., `<dataset_name>_Results/` or `Batch_Results/`). The updated mathematical engine generates the following files:

```text
<dataset_name>_Results/
│
├── fit/ (or Batch_Results/<name>/)
│   ├── GFitResults.npy          # Complete Python dictionary with all fit matrices and errors
│   ├── Corrected_data.txt       # Pre-processed experimental 2D matrix
│   ├── Fitted_data.txt          # Global fit 2D matrix
│   ├── Residuals.txt            # Fit residuals 2D matrix
│   ├── Amplitudes.txt           # SAS/DAS amplitudes and errors (Includes kinetic params in the header)
│   ├── WL.txt                   # Cropped/Binned wavelengths axis
│   ├── TD.txt                   # Cropped/Binned time delays axis
│   └── <dataset>_FitReport.pdf  # Comprehensive, publication-quality vector PDF report
│
├── Plots/
│   ├── DAS.png                  # Decay/Species Associated Spectra plot (and Oscillation if applicable)
│   ├── Residuals_Map.png        # 2D Residuals heatmap
│   ├── Trace_<WL>nm.png         # Exported kinetic trace plot at a specific wavelength
│   └── Trace_<WL>nm.txt         # Exported kinetic trace data (TD vs Exp vs Fit)
│
├── Spectrum_<delay>ps.txt       # (Optional) Exported spectral cross-section at a specific time delay
└── project_name.proj            # (Optional) SPARK workspace save file (UI states, guesses, and matrices)
```
## Supported Techniques

| Technique | Description |
|---|---|
| **TAS** | Transient Absorption Spectroscopy |
| **FLUPS** | Fluorescence Up-Conversion Spectroscopy |
| **TCSPC** | Time-Correlated Single Photon Counting |
| **XFEL / Time-scans** | X-ray Free Electron Laser pump-probe experiments |

---

## Modules

### FLUPS Analyzer
> **Input Requirement:** A single data matrix (the experimental measurement).

Loads raw FLUPS data matrices and provides a fully interactive 2D map with real-time spectral and kinetic cross-sections under the cursor. Chirp ($t_0$) correction is available both manually (user-selected points fitted to a polynomial or non-linear dispersion model) and automatically via a global-intensity-thresholding algorithm that identifies the half-rise point of the signal at each wavelength. Corrected data is exported as `.npy` for direct loading into the Global Fit Panel.

### TAS Analyzer
> **Input Requirement:** Two data matrices (the experimental sample measurement and the solvent background).

Extends the FLUPS workflow for Transient Absorption data. The solvent background is subtracted in real time as the user adjusts the amplitude and temporal shift sliders, with the subtraction computed via bilinear interpolation onto the measurement grid at each slider event. Pump scatter removal, SymLog/Linear axis switching, etc. A blitting-based rendering engine ensures smooth cursor tracking even on large datasets.

### Global Fit Panel
The main analysis engine. Supports simultaneous loading of multiple datasets, pre-processing (baseline subtraction, spectral and temporal cropping, wavelength exclusion zones, binning, normalization), SVD rank diagnosis, and global fitting. After optimization, results include DAS/SAS spectra with error bars, kinetic trace exploration, 2D residual maps, and a full parameter identifiability report. Batch fitting over all loaded datasets is available with a single button, producing individual result folders per dataset.

### 2D Mapper
Assembles 2D pump-probe maps from collections of individual time-scan `.npy` files — useful for XFEL or any scanning-based experiment where each kinetic trace at a different photon energy or wavelength is stored as a separate file. The assembled matrix is exported in the same `.npy` format consumed by the Global Fit Panel.

---

## Mathematical Background

All models convolve the kinetic response with a **Gaussian IRF** of FWHM $w$ centered at $t_0$:

$$\text{IRF}(t) = \frac{1}{w\sqrt{\pi}} \exp\left[-\left(\frac{t - t_0}{w}\right)^2\right]$$

The convolution integral is evaluated analytically using the scaled complementary error function (`erfcx`), which avoids catastrophic cancellation at large arguments and is numerically stable across the full dynamic range of typical ultrafast experiments (sub-100 fs to nanoseconds):

$$\left[\text{IRF} \otimes e^{-t/\tau}\right](t) = \frac{1}{2}\exp\!\left(\frac{\sigma^2}{2\tau^2} - \frac{t-t_0}{\tau}\right) \cdot \text{erfc}\!\left(\frac{\sigma^2 - \tau(t-t_0)}{\sqrt{2}\,\sigma\tau}\right)$$

where $\sigma = w / (2\sqrt{2\ln 2})$ is the Gaussian standard deviation.

---

### Variable Projection (VarPro)

The 2D data surface $\Delta A(t, \lambda)$ is modelled as:

$$\mathbf{D} = \mathbf{C}(\boldsymbol{\theta}) \cdot \mathbf{S}^\top + \boldsymbol{\varepsilon}$$

where $\mathbf{C}(\boldsymbol{\theta}) \in \mathbb{R}^{N_t \times n}$ is the concentration matrix (non-linearly dependent on $\boldsymbol{\theta} = \{w, t_0, \tau_1,\ldots,\tau_n\}$), $\mathbf{S}^\top \in \mathbb{R}^{n \times N_\lambda}$ is the matrix of spectral amplitudes (linear parameters), and $\boldsymbol{\varepsilon}$ is the residual.

At each evaluation of the residual function, $\mathbf{S}^\top$ is eliminated analytically:

$$\hat{\mathbf{S}}^\top(\boldsymbol{\theta}) = \mathbf{C}(\boldsymbol{\theta})^\dagger \cdot \mathbf{D}$$

where $\mathbf{C}^\dagger$ is the Moore-Penrose pseudoinverse, computed via `numpy.linalg.lstsq`. The non-linear optimizer (`scipy.optimize.least_squares`, TRF method with soft-L1 loss) therefore only ever minimizes over the small vector $\boldsymbol{\theta}$, regardless of how many wavelengths are in the dataset.

---

### Parallel Model — Decay Associated Spectra (DAS)

Components decay independently:

$$\Delta A(t, \lambda) = \text{IRF}(t) \otimes \sum_{i=1}^{n} A_i(\lambda)\, e^{-t/\tau_i}$$

Each column $A_i(\lambda)$ of $\mathbf{S}^\top$ is the **DAS** of component $i$.

---

### Sequential Model — Species Associated Spectra (SAS)

Successive population transfer: $S_1 \xrightarrow{k_1} S_2 \xrightarrow{k_2} \cdots \xrightarrow{k_n} \text{GS}$

Populations governed by the Bateman equations:

$$C_n(t) = \left(\prod_{j=1}^{n-1} k_j\right) \sum_{j=1}^{n} \frac{e^{-k_j t}}{\displaystyle\prod_{\substack{p=1\\p\neq j}}^{n}(k_p - k_j)}$$

$$\Delta A(t, \lambda) = \text{IRF}(t) \otimes \sum_{i=1}^{n} \text{SAS}_i(\lambda)\, C_i(t)$$

The Bateman equations are evaluated via the same vectorized `erfcx`-based kernel as the parallel model, so all $n$ species concentrations are computed in a single batched matrix operation.

---

### Damped Oscillation Model

Superposition of population relaxation and a coherent vibrational wavepacket:

$$\Delta A(t, \lambda) = \left[\text{IRF}(t) \otimes \sum_{i=1}^{n} A_i(\lambda)\, e^{-t/\tau_i}\right] + B(\lambda)\cdot S_\text{osc}(t)$$

$$S_\text{osc}(t) = \frac{1}{2}\left[1 + \text{erf}\!\left(\frac{t-t_0}{\sqrt{2}\,w}\right)\right] e^{-\alpha(t-t_0)} \sin\\big(\omega(t-t_0)+\phi\big)$$

The error function step simulates the convolution of the oscillation onset with the IRF without requiring numerical integration. VarPro still applies: $B(\lambda)$ is solved linearly alongside the $A_i(\lambda)$, so $\{\alpha, \omega, \phi\}$ are the only additional non-linear parameters.

---

### Custom K-Matrix Model

Arbitrary kinetic schemes are defined interactively through a **visual Jablonski/Grotrian diagram builder** — states are drawn as draggable boxes and transitions as directed arrows, each labeled with a parameter type (`tau` for a lifetime, `gamma` for a branching ratio) and a variable name. The software then:

1. Assembles the rate matrix **K** from the transition graph.
2. Diagonalizes **K** analytically: $\mathbf{K} = \mathbf{V}\,\boldsymbol{\Lambda}\,\mathbf{V}^{-1}$.
3. Propagates the initial population $P_0$ (automatically identified as the state with no incoming arrows) through the eigenbasis.
4. Constructs the concentration matrix $\mathbf{C}(t)$ and passes it to the VarPro solver.

The eigenvector decomposition handles schemes of arbitrary topology — branched pathways, parallel channels, and cycles — without requiring the user to derive Bateman-like equations by hand.

---

### Coherent Artifact Model

When the coherent artifact checkbox is active, three additional basis functions are appended to $\mathbf{C}$:

$$\phi_0(t) = \text{IRF}(t), \qquad \phi_1(t) = \frac{d}{dt}\text{IRF}(t), \qquad \phi_2(t) = \frac{d^2}{dt^2}\text{IRF}(t)$$

These three terms absorb Raman scattering ($\phi_0$) and cross-phase modulation / XPM ($\phi_1$, $\phi_2$) simultaneously. Because they enter the model linearly, VarPro handles them at zero extra cost to the non-linear optimizer.

---

## Error Estimation

Rigorous uncertainty quantification is a central design goal of the fitting engine. The software implements a **three-tier** framework, each tier adding more information at increasing computational cost.

---

### Tier 1 — Asymptotic Covariance (always computed)

After convergence, the Jacobian $\mathbf{J}$ of the residual vector with respect to the **non-linear parameters only** (VarPro ensures that amplitude columns are structurally zero and are explicitly excluded before the SVD) is decomposed as:

$$\mathbf{J} = \mathbf{U}\,\mathbf{S}\,\mathbf{V}^\top$$

The parameter covariance is estimated as:

$$\text{Cov}(\hat{\boldsymbol{\theta}}) \approx (\mathbf{J}^\top\mathbf{J})^{-1}\,\hat{\sigma}^2 = \mathbf{V}\,\mathbf{S}^{-2}\,\mathbf{V}^\top\cdot\hat{\sigma}^2$$

where the noise variance is estimated from the fit residuals, correctly accounting for **both** the non-linear parameters and the analytically eliminated linear amplitudes in the degrees-of-freedom count:

$$\hat{\sigma}^2 = \frac{\|\boldsymbol{\varepsilon}\|^2}{N_t N_\lambda - (n_\text{nl} + N_\lambda \cdot n_\text{species})}$$

Standard errors on lifetimes and IRF parameters are the square roots of the diagonal of $\text{Cov}(\hat{\boldsymbol{\theta}})$.

> **Note:** The standard errors assume local linearity of the model around the optimum and homoscedastic Gaussian noise. They are most reliable when the residuals are structurally flat (confirmed via the residual map) and the condition number is low.

---

### Tier 2 — Parameter Correlation and Identifiability (always computed)

The same SVD yields two additional diagnostics that are reported automatically alongside the standard errors.

**Condition number:**

$$\kappa(\mathbf{J}) = \frac{s_1}{s_{n_\text{nl}}}$$

where $s_1 \geq \cdots \geq s_{n_\text{nl}}$ are the singular values of $\mathbf{J}$. The condition number measures how "flat" the $\chi^2$ hypersurface is in the worst direction:

| $\kappa$ | Interpretation |
|---|---|
| $< 10^2$ | Well-conditioned. Standard errors are reliable. |
| $10^2$–$10^4$ | Moderate correlation between some parameters. Inspect the correlation matrix. |
| $> 10^4$ | At least one parameter combination is poorly determined. Standard errors may be misleading. |
| $> 10^6$ | Near-singular. Consider fixing a parameter or merging two components. |

**Sloppy direction:** The right singular vector $\mathbf{v}_{n_\text{nl}}$ (corresponding to $s_{n_\text{nl}}$) identifies the linear combination of parameters that the data constrains least. For example, a large projection onto $\{\tau_2, \tau_3\}$ with opposite signs means that increasing $\tau_2$ while decreasing $\tau_3$ hardly changes $\chi^2$ — the two lifetimes are not independently resolvable at the available time resolution and noise level.

**Full correlation matrix:** The normalized covariance

$$R_{ij} = \frac{\text{Cov}_{ij}}{\sqrt{\text{Cov}_{ii}\,\text{Cov}_{jj}}}$$

is computed and displayed as a color-coded heatmap. Off-diagonal entries near $\pm 1$ flag parameter pairs that cannot be determined independently.

---

### Tier 3 — Profile Likelihood Confidence Intervals (on demand)

When the condition number is high or the model is strongly non-linear (as is the case for K-matrix models with branching ratios), the quadratic approximation underlying Tier 1 may underestimate the true uncertainty — particularly when a parameter is bounded below (e.g., a lifetime cannot be negative) or when two components have similar timescales.

The profile likelihood method makes no linearity assumption. For each target parameter $\theta_k$, the parameter is fixed to a grid of values $\theta_k^{(j)}$ spanning several standard errors around the optimum, and all remaining free parameters are re-optimized:

$$\chi^2_\text{prof}(\theta_k^{(j)}) = \min_{\boldsymbol{\theta}_{-k}}\, \|\mathbf{D} - \mathbf{C}(\theta_k^{(j)},\boldsymbol{\theta}_{-k})\,\hat{\mathbf{S}}^\top\|^2_F$$

The confidence interval at level $1-\alpha$ is defined as the connected region where the profile does not exceed the likelihood-ratio threshold:

$$\Delta\chi^2(\theta_k) = \chi^2_\text{prof}(\theta_k) - \chi^2_\text{min} \leq \chi^2_{1,\,1-\alpha}$$

For $\alpha = 0.05$ the threshold is $\chi^2_{1,\,0.95} = 3.84$.

The resulting interval is **asymmetric in general**: a lifetime bounded by zero will show a longer upper tail than lower tail, which the symmetric Tier-1 interval cannot represent. The profile curve itself is plotted interactively, so the user can immediately see whether the $\chi^2$ surface is parabolic (well-behaved, Tier-1 errors are fine) or flat/asymmetric (Tier-3 intervals are necessary).

The computational cost is $N_\text{steps}$ full refits per parameter. The analysis is therefore run selectively — on the specific parameters flagged by the Tier-2 condition number or correlation matrix — rather than exhaustively.

---



### Dependencies

| Package | Purpose |
|---|---|
| `PyQt5` | GUI framework |
| `numpy` | Array operations and linear algebra |
| `scipy` | Non-linear optimization, interpolation, special functions (`erfcx`) |
| `matplotlib` | All plotting and interactive canvases |
| `pandas` | CSV loading and data cleaning |

---

## Quick Start

### Loading and preprocessing data

1. Launch the application and select **FLUPS Analyzer** or **TAS Analyzer**.
2. Click **Load CSV** and select your data file (TAS requires measurement + solvent files).
3. Use the wavelength sliders to crop the spectral range.
4. Apply chirp correction: **Auto-Chirp** for automated $t_0$ detection, or **Select $t_0$ points** → **Fit $t_0$** for manual selection.
5. Click **Global Fit** to pass the corrected data to the fitting engine.

### Running a global fit

1. In the **Global Fit Panel**, switch to the **2. Fit** tab.
2. Select the number of components and model type.
3. Optionally run **SVD Analysis** to determine the number of photo-active species.
4. Click **Edit Initial Guesses** to set starting lifetimes and bounds.
5. Click **RUN FIT**. Results appear in the **Fit Result** and **Residuals** tabs.
6. Click **Show Plots / Results** to open the DAS/SAS viewer and kinetic trace explorer.

### Assessing fit quality and parameter reliability

1. After fitting, **Show Plots / Results** opens the results summary automatically.
2. The **condition number** (green/orange/red) gives an immediate signal of whether the parameters are reliably determined.
3. If the condition number is high, the **sloppy direction** text identifies which parameter combination is problematic.
4. Click **Analyze Identifiability (Profile Likelihood)** to open the interactive profile dialog, select a parameter, and compute its asymmetric confidence interval.

### Using the Custom Model Builder

1. Select **Custom GUI Model** and click **Open Visual Model Builder**.
2. Add excited states with **Add State** (e.g., `S1*`, `3CT`, `S0`).
3. Connect them with **Connect States**, specifying `tau` or `gamma` and a variable name.
4. Click **Load & compile kinetic model**. The K-matrix is assembled and diagonalized automatically.

---

## Screenshots

<details>
<summary><b>FLUPS Analyzer</b></summary>
<br/>
<img width="1627" height="940" alt="FLUPS Analyzer" src="https://github.com/user-attachments/assets/51393cb7-b67a-49d3-be14-8a833d98d621"/>
</details>

<details>
<summary><b>TAS Analyzer</b></summary>
<br/>
<img width="1623" height="953" alt="TAS Analyzer" src="https://github.com/user-attachments/assets/a3fd3c13-9dbd-42bc-80c5-c33d68c9a29f"/>
</details>

<details>
<summary><b>Global Fit Panel</b></summary>
<br/>
<img width="1913" height="1030" alt="Global Fit" src="https://github.com/user-attachments/assets/64382479-8c5b-442d-8c24-e16f0b1d2b97"/>
</details>

<details>
<summary><b>SVD Analysis</b></summary>
<br/>
<img width="1920" height="1020" alt="SVD" src="https://github.com/user-attachments/assets/a1fe822d-9932-428b-b64f-3dade367e0e4"/>
</details>

<details>
<summary><b>Species Associated Spectra & Kinetic Traces</b></summary>
<br/>
<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/7b76616a-86f5-4897-a873-b6ee7c6a2a6c" alt="SAS Spectra" width="100%"/></td>
    <td><img src="https://github.com/user-attachments/assets/95bd9d22-72ce-4181-8a25-529fa3d56f20" alt="Trace Viewer" width="100%"/></td>
  </tr>
</table>
</details>

---

## Contributing

Contributions are welcome. To propose a change:

1. Fork the repository and create a feature branch: `git checkout -b feature/your-feature`.
2. Make your changes and add tests if applicable.
3. Open a Pull Request describing the motivation and implementation.

For bug reports or feature requests, please open a [GitHub Issue](https://github.com/AlejandroSerranoCapote/Ultrafast-Spectroscopy-Analyzer/issues) with a minimal reproducible example (data file + settings) where possible.

---

## Citation

If you use this software in published work, please cite it as:

```bibtex
@software{SerranoCapote_SPARK_2025,
  author  = {Serrano Capote, Alejandro},
  title   = {{SPARK}},
  year    = {2025},
  version = {1.4},
  url     = {https://github.com/AlejandroSerranoCapote/SPARK},
  license = {GPL-3.0}
}
```

---

## License

This project is licensed under the **GNU General Public License v3.0**.  
See the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <sub>Developed by <a href="https://github.com/AlejandroSerranoCapote">Alejandro Serrano Capote</a></sub>
</div>
