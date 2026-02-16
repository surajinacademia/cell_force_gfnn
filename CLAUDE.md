# CLAUDE.md

## Project Overview

This repository implements **Green's Function Neural Networks (GFNN)** for predicting cellular traction forces from fluorescence images of the protein zyxin. It accompanies the paper ["Zyxin is all you need: machine learning adherent cell mechanics"](https://arxiv.org/abs/2303.00176), generating the results for **Figure 6**.

The pipeline combines two approaches:
1. **GFNN** — A neural network that learns a nonlinear mapping from zyxin intensity images to traction force fields using the Clebsch decomposition: `F = grad(phi) + psi * grad(chi)`, where each scalar field is obtained via a learnable FFT-based convolution (Green's function) applied to zyxin.
2. **SINDy** (Sparse Identification of Nonlinear Dynamics) — After GFNN training, SINDy discovers a parsimonious symbolic PDE linking zyxin derivatives to force, using ElasticNet-regularized regression over a library of candidate terms.

### Related Repositories
- U-Net analysis: https://github.com/schmittms/cell_force_prediction
- Physical bottleneck analysis: https://github.com/schmittms/physical_bottleneck

## Repository Structure

```
cell_force_gfnn/
├── gfnn_models.py              # PyTorch model definitions (ClebschGFNN, RealGFNN, RealCoulombGFNN)
├── gfnn_data_processing.py     # Data transforms, dataset class, scalar term computation
├── training.py                 # Main training loop with CLI (argparse)
├── sindy_adding_terms.py       # SINDy term-discovery pipeline
├── collect_dataset.py          # One-off script to build HDF5 dataset from .npy files
├── GFNNCellMechanics.ipynb     # Main analysis notebook: GFNN evaluation & figure generation
├── CoulombElectrostatics.ipynb # Toy Coulomb demo validating the methodology
├── SINDyAddingTerms.ipynb      # SINDy results plotting across cells
├── SINDy_11cell1.ipynb         # Detailed SINDy analysis on cell 11_cell_1
├── slurm_python.slurm          # SLURM GPU job script for model training
├── sindy_adding_terms.slurm    # SLURM CPU array job script for SINDy
├── Figures/                    # Pre-generated output figures (SVG, PNG, PDF)
├── UnseenWashout_RealGFNN_down=4_beta=0.1  # Saved PyTorch model checkpoint
├── README.md
└── CLAUDE.md                   # This file
```

This is a **research scripts repository** — it is not packaged as a Python library. There is no `setup.py`, `pyproject.toml`, or `requirements.txt`.

## Key Source Modules

### `gfnn_models.py` — Model Definitions
- **`ConvBlock`**: Depthwise convolution + 1x1 convolution with `sin` activation (SIREN-like).
- **`ClebschGFNN`**: Full Clebsch GFNN using `torch.fft.fft2` (complex FFT). Learnable kernel stored in Fourier space.
- **`RealGFNN`**: Memory-efficient version using `torch.fft.rfft2` (real FFT). Same physics as `ClebschGFNN`.
- **`RealCoulombGFNN`**: Subclass of `RealGFNN` that drops the `grad(phi)` term, keeping only the `psi * grad(chi)` Coulomb form.
- Each model has a `get_transform()` method returning a `torchvision.transforms.Compose` pipeline.

### `gfnn_data_processing.py` — Data Pipeline
Key transform classes (used in `torchvision.transforms.Compose` pipelines):
- **`CellCrop`**: Center-crop images around cell mask.
- **`Threshold`**: Filter force vectors by magnitude.
- **`Downsample`**: Spatial downsampling via `skimage.measure.block_reduce`.
- **`FourierCutoff`**: Low-pass Fourier filter on zyxin input.
- **`GetXY`**: Extracts `(zyxin, F_ml)` as `(X, Y)` training pairs.
- **`GetScalarTerms`**: Computes six scalar invariants of zyxin for SINDy: `zeta`, `|grad zeta|^2`, `nabla^2 zeta`, `zeta^2`, `zeta |grad zeta|^2`, `zeta nabla^2 zeta`.
- **`Dataset`**: PyTorch Dataset loading from HDF5 file.

### `training.py` — Training Script
CLI entry point with `argparse`. Key arguments:
| Argument | Default | Description |
|---|---|---|
| `--base_lr` | `1e-4` | Learning rate for conv_in parameters |
| `--kernel_lr` | `1e-2` | Learning rate for kernel parameters |
| `--beta` | `0.1` | Sparsity regularization penalty |
| `--cell_crop` | `1024` | Crop size around cell center |
| `--nmax` | `50` | Fourier cutoff frequency |
| `--downsample` | `4` | Spatial downsampling factor |
| `--epochs` | `200` | Number of training epochs |
| `--scheduler` | `False` | Enable ReduceLROnPlateau |
| `--real` | `False` | Use RealGFNN (rfft2) |
| `--coulomb` | `False` | Use RealCoulombGFNN |

The loss function is `MSE + beta * kernel_sparsity`, where kernel sparsity penalizes all Fourier modes except DC.

### `sindy_adding_terms.py` — SINDy Pipeline
- Builds a library of candidate force terms by convolving scalar invariants with radial Green's functions (`1/r`, `r`, `log(r)`, exponential decays).
- Forms Clebsch-type products and gradient terms as the candidate library.
- Fits ElasticNet models at many regularization strengths to discover which terms are most important.
- CLI takes a single integer argument (cell index) for SLURM array parallelism.

## Dependencies

Inferred from imports (no `requirements.txt` exists):

**Core:**
- `torch` (PyTorch) — model definitions, training, FFT operations
- `torchvision` — transform composition
- `numpy` — array operations
- `pandas` — dataframe handling for dataset metadata
- `h5py` — HDF5 file I/O

**Scientific:**
- `scikit-image` (`skimage.measure.block_reduce`) — image downsampling
- `scikit-learn` (`ElasticNet`, `train_test_split`, `StandardScaler`) — SINDy fitting
- `pysindy` — sparse identification of nonlinear dynamics
- `scipy` — interpolation (in notebooks)

**Utilities:**
- `tqdm` — progress bars
- `matplotlib`, `seaborn` — plotting (in notebooks)

The SLURM scripts reference a conda environment at `/project/vitelli/ml_venv`.

## Running the Code

### Training a GFNN model
```bash
python training.py --real --epochs 200 --downsample 4 --beta 0.1
```

### Running SINDy analysis for a specific cell
```bash
python sindy_adding_terms.py <cell_index>
```

### SLURM submission (Midway3 cluster)
```bash
# GPU training
sbatch slurm_python.slurm

# SINDy array job (cells 0-30, 5 repeats each)
sbatch sindy_adding_terms.slurm
```

### Data
- Raw data is expected at hardcoded paths (e.g., `/home/jcolen/CellProject/data/`).
- The `data/` directory is gitignored.
- `collect_dataset.py` consolidates per-cell `.npy` files into `data/cell_dataset.h5`.

## Development Conventions

### Code Style
- **No formal linting or formatting tools** are configured (no black, flake8, mypy, etc.).
- Mixed indentation: some files use tabs (notably class bodies in `gfnn_models.py`), others use spaces.
- When editing, **match the surrounding indentation style** in each file.

### Testing
- **No test suite exists.** There are no test files, no testing framework, and no CI/CD.
- Validation is done through notebook-based analysis and visual inspection of outputs.

### Hardcoded Paths
Several scripts contain hardcoded absolute paths to data directories on the original compute cluster:
- `/home/jcolen/CellProject/data/` (in `gfnn_data_processing.py`, `training.py`)
- `/project/vitelli/cell_stress/` (in `sindy_adding_terms.py`)
- `/project/vitelli/ml_venv` (in SLURM scripts)

These paths would need to be updated for use on a different system.

### Notebooks
- Jupyter notebooks (`.ipynb`) are marked as `linguist-vendored` in `.gitattributes` to exclude them from GitHub language statistics.
- Notebooks contain inline analysis code, visualization, and figure generation.
- The main results notebook is `GFNNCellMechanics.ipynb`.

### Model Checkpoints
- Saved as PyTorch files containing `state_dict`, `hparams`, `epoch`, and `loss` keys.
- The included checkpoint `UnseenWashout_RealGFNN_down=4_beta=0.1` is a trained `RealGFNN` model.

## Architecture Notes

### Clebsch Decomposition
The core physics insight is that any 2D vector field can be decomposed as:
```
F = grad(phi) + psi * grad(chi)
```
where `phi`, `chi`, and `psi` are scalar fields. In the GFNN, `phi` and `chi` are obtained by convolving the zyxin image with learnable Fourier-space kernels, while `psi` is a nonlinear function of zyxin computed by the `ConvBlock` layers.

### FFT-based Convolution
The Green's function convolution is implemented efficiently in Fourier space:
```
output = IFFT(kernel * FFT(input))
```
where `kernel` is a learnable complex-valued parameter in Fourier space. The `RealGFNN` variant uses `rfft2`/`irfft2` for approximately 2x memory savings.

### Training Strategy
- Separate learning rates for the `conv_in` layers (lower) and the `kernel` parameters (higher).
- Sparsity regularization on the Fourier kernel encourages compact, interpretable Green's functions.
- Gradient clipping (max norm 1.0) is applied during training.
