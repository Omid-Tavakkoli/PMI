## Pore Morphology-Based Initializer (PMI)

This repository contains (`PMI.py`) for initializing lattice Boltzmann simualtions of multiphase flow in porous media. It reads a raw binary image (`uint8`, 0 = pore, 1 = solid), targets specific wetting-phase saturations via morphological operations, and writes labeled fluid distrinutions to disk. For detailed methodology explanation see:

> Tavakkoli, O., Da Wang, Y., Mostaghimi, P., Armstrong, R.T., 2025. "A pore morphology-based initializer to accelerate lattice Boltzmann simulations of capillary-dominated flow with variable wettability" *Physics of Fluids* **37**, **9**. [doi:10.1063/5.0285656](http://dx.doi.org/10.1063/5.0285656)

---

### Features
- **Parallel processing** with Python multiprocessing
- **Caching and multi-pass morphology** for large structuring elements
- **Automatic kernel-size search and refinement** to match target saturation(s)

---

### Installation
1) Python 3.9+ recommended.
2) Install dependencies:
- **pip**:

```bash
pip install -r requirements.txt
```

- **conda**:

```bash
conda create -n pmi -c conda-forge python=3.11 numpy scikit-image
conda activate pmi
```

---

### Inputs
- `domain.raw`: 3D binary image stored as `uint8` with shape `(Z, Y, X)`
  - Values: 0 = pore, 1 = solid
- `input.txt`: runtime configuration (see below).

---

### Configuration (`input.txt`)
Keys recognized by `PMI.py`:

- `filename`: path to the raw image (e.g., `domain.raw`)
- `filesize_x`, `filesize_y`, `filesize_z`: integer dimensions (X, Y, Z)
- `theta`: contact angle in degrees
- `target_saturations`: comma-separated list of target saturations (0–1)
- `tol`: tolerance for the kernel-size search
- `num_threads`: number of CPU cores to use (int)

Example:

```ini
filename = domain.raw
filesize_x = 100
filesize_y = 100
filesize_z = 100
theta = 30.0
target_saturations = 0.20, 0.50, 0.80
tol = 1e-6
num_threads = 8
```
---

### Usage
Run from the project directory:

```bash
python PMI.py
```
---

### Outputs
- `result_sat*.raw`: `uint8` 3D arrays labeled as:
  - 0 = solid
  - 1 = non-wetting phase (NWP)
  - 2 = wetting phase (WP)

---

## Re-using the code

If you use this implementation in academic work, please cite the paper above. A BibTeX entry is:

```bibtex
@article{Tavakkoli2025,
  title = {A pore morphology-based initializer to accelerate lattice Boltzmann simulations of capillary-dominated flow with variable wettability},
  volume = {37},
  ISSN = {1089-7666},
  url = {http://dx.doi.org/10.1063/5.0285656},
  DOI = {10.1063/5.0285656},
  number = {9},
  journal = {Physics of Fluids},
  publisher = {AIP Publishing},
  author = {Tavakkoli,  Omid and Wang,  Ying Da and Mostaghimi,  Peyman and Armstrong,  Ryan T.},
  year = {2025},
  month = sep 
}
```

---


