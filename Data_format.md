#  Supported data formats and generated files

This document describes the file formats that the software can import and export.

---

##  Input data

###  FLUPS (*Fluorescence Up-Conversion Spectroscopy*)
Allows a `.csv` file with the following format:

| Row / Column     | Content                  |
|------------------|--------------------------|
| First row        | Time delays (ps)         |
| First column     | Wavelengths (nm)         |
| Remaining matrix | Matrix ΔA(λ, t)          |

Example:
```math
\text{ΔA(λ,t)} =
\begin{bmatrix}
λ \setminus t & -1.00 & -0.50 & 0.00 & 0.50 & 1.00 \\
400 & 0.002 & 0.005 & 0.010 & 0.004 & 0.001 \\
410 & 0.001 & 0.004 & 0.008 & 0.003 & 0.000
\end{bmatrix}
```
It also admits 3 `.txt` files with the following form:
```math
\text{ΔA(λ,t)} =
\begin{bmatrix}
 0.002 & 0.005 & 0.010 & 0.004 & 0.001 \\
0.001 & 0.004 & 0.008 & 0.003 & 0.000
\end{bmatrix}
```
```math
\text{λ} =
\begin{bmatrix}
  450& -475 & 500 & 525 & 550 & ... 
\end{bmatrix}
```
```math
\text{t} =
\begin{bmatrix}
  -1.00 & -0.50 & 0.00 & 0.50 & 1.00  & ... 
\end{bmatrix}
```
---

###  TAS (*Transient Absorption Spectroscopy*)
It requieres **two files**:

1. **Experimental measurement** (`sample.csv`)  
2. **Solvent measurement** (`solvent.csv`)  

Both must have the same structure as in FLUPS (`.csv`):
- Row 1 → Time delays
- Column 1 → Wavelengths  
- Remaining matrix → ΔA(λ, t)

Example:
```math
\text{ΔA(λ,t)} =
\begin{bmatrix}
λ \setminus t & -1.00 & -0.50 & 0.00 & 0.50 & 1.00 \\
400 & 0.002 & 0.005 & 0.010 & 0.004 & 0.001 \\
410 & 0.001 & 0.004 & 0.008 & 0.003 & 0.000
\end{bmatrix}
```

The software combines both matrices, subtracts the solvent, and applies the user-defined corrections.

---

##  Output data

```text
<file_name>_Results/
│
├── WL.txt                 → Wavelengths (nm)
├── TD.txt                 → Time delays (ps)
├── treated_data.npy       → Corrected data in NumPy format
├── t0_fit.txt             → Fit curve t₀(λ)
├── fit_params.txt         → Fit model parameters
├── kin.txt                → Kinetics (ΔA vs time)
├── spec.txt               → Spectra (ΔA vs λ)
│
├── Fit/                   → File with global fit results
│   ├── Amplitudes.txt     → DAS/SAS Amplitudes 
│   ├── GFit_resid.txt     → Fit residuals
│   ├── GFit.txt           → Kinetic fit for all the wavelengths
│   ├── GFitResults.npy    → .npy dictionary of NumPy with all the data
│   ├── TD.txt             → Time delays (ps)
│   └── WL.txt             → Wavelengths (nm)
│
└── Plots/                 → File with the fit plots
    ├── DAS.png            → SAS/DAS Plot
    ├── Fit_xxxnm.png      → Kinetic fit plot at a specific wavelength
    ├── Fit_xxxnm.txt      → Kinetic fit at a specific wavelength
    └── Residual.png       → Residual fit plots
```

##  Additional notes

- `.npy` files can be loaded directly in Python with `numpy.load()`.
- File names generates automatically depends of the name of the input file

---
