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

---

## Plotting Guide — Styles, Standards, and Practices

All visualization in this repository is confined to the four Jupyter notebooks. The Python scripts (`training.py`, `sindy_adding_terms.py`, etc.) contain zero plotting code.

### Plotting Libraries
```python
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import matplotlib.gridspec
import matplotlib.font_manager
import seaborn as sns              # GFNNCellMechanics, SINDy_11cell1 only
from scipy.interpolate import RectBivariateSpline, interp1d
%matplotlib inline
```

### Which Notebook Generates Which Figure

| Notebook | Output File | Description |
|---|---|---|
| `GFNNCellMechanics.ipynb` | `Figures/Fig4Row.svg` | Main GFNN results figure (zyxin, forces, kernels, Clebsch fields) |
| `GFNNCellMechanics.ipynb` | `Figures/SIFig_Clebsch.pdf` | Supplementary Clebsch decomposition analysis |
| `GFNNCellMechanics.ipynb` | `data/gfnn_movie_frames/frame_*.png` | Animated movie frames |
| `CoulombElectrostatics.ipynb` | `Figures/Coulomb_Clebsch.png` | Toy Coulomb electrostatics validation |
| `SINDyAddingTerms.ipynb` | `Figures/adding_terms_sindy.svg` | SINDy term discovery across cells |
| `SINDy_11cell1.ipynb` | `sindy_11cell1_halfrow.svg` | SINDy single-cell detailed analysis |
| `SINDy_11cell1.ipynb` | `sindy_11cell1_halfrow_SMALL.svg` | Smaller version of above |

---

### Publication Figure Style (Canonical rcParams)

Two figure scales are used: **presentation** (larger fonts/lines) and **publication** (compact, journal-ready). The publication style is the final style used for paper figures.

#### Publication Style (compact, for journal panels)
```python
lw = .75
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = lw
plt.rcParams['lines.linewidth'] = lw
plt.rcParams['axes.labelsize'] = 7
plt.rcParams['axes.labelpad'] = 0
plt.rcParams['font.size'] = 8
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.width'] = lw
plt.rcParams['ytick.major.width'] = lw
plt.rcParams['xtick.major.size'] = 2
plt.rcParams['ytick.major.size'] = 2
plt.rcParams['xtick.labelsize'] = 7
plt.rcParams['ytick.labelsize'] = 7
plt.rcParams['legend.framealpha'] = 0.
plt.rcParams['legend.fontsize'] = 6            # always font.size - 2
plt.rcParams['legend.handletextpad'] = 0.5
plt.rcParams['legend.handlelength'] = .5
plt.rcParams['legend.columnspacing'] = .5
plt.rcParams['legend.borderpad'] = 0.
plt.rcParams['image.origin'] = 'lower'
plt.rcParams['image.cmap'] = 'inferno'
matplotlib.rcParams['pdf.fonttype'] = 42        # TrueType for PDF
matplotlib.rcParams['ps.fonttype'] = 42         # TrueType for PostScript
```

#### Presentation Style (larger, for talks/exploration)
```python
lw = 1.5
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = lw
plt.rcParams['font.size'] = 10
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.width'] = lw
plt.rcParams['ytick.major.width'] = lw
plt.rcParams['legend.framealpha'] = 0.
plt.rcParams['legend.fontsize'] = 8             # font.size - 2
plt.rcParams['legend.handletextpad'] = 0.5
plt.rcParams['legend.handlelength'] = 1.
```

#### SINDy Analysis Style
```python
matplotlib.rcParams['axes.linewidth'] = 2
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['font.style'] = 'normal'
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Arial'
```

**Key rules to follow:**
- Font is always **Arial** (sans-serif).
- Ticks always point **inward** (`direction='in'`).
- Tick width always matches `axes.linewidth`.
- Legend is always **frameless** (`framealpha=0`).
- Legend font is always **`font.size - 2`**.
- Image origin is always **`'lower'`**.
- Default colormap is **`'inferno'`**.
- PDF/PS font type is always **42** (TrueType) for vector compatibility.

---

### Color Conventions

#### Colormaps by Data Type

| Data | Colormap | Typical vmin/vmax |
|---|---|---|
| Force magnitude | `'inferno'` | `vmin=0.5`, `vmax=3` (or 4) |
| Zyxin fluorescence | `'Greys_r'` | `vmin=0`, `vmax=3` (or 4) |
| phi scalar field | `'Blues'` / `'Blues_r'` | auto |
| psi (xi) scalar field | `'Oranges'` / `'Oranges_r'` | auto |
| chi scalar field | `'Greens'` / `'Greens_r'` | auto |
| xi Clebsch input | `'RdPu'` | `vmin=0.1*std`, `vmax=5*std` |
| chi Clebsch input | `'YlOrBr'` | `vmin=0`, `vmax=5*std` |
| Charge density (bipolar) | `'bwr'` | symmetric: `vmin=-vmax`, `vmax=max(abs)` |

For scalar field colormaps, NaN pixels outside the cell mask are set to white:
```python
plt.cm.Blues.set_bad('white')
plt.cm.Oranges.set_bad('white')
plt.cm.Greens.set_bad('white')
```

#### Line/Marker Colors by Meaning

| Meaning | Color |
|---|---|
| Experimental data | `'grey'` |
| GFNN model prediction | `'blue'` |
| SINDy equation model | `'red'` |
| UNet/ML model | `'blue'` |
| Wash-in/wash-out markers | `'green'` |
| Analytic reference | `'black'` (dashed) |
| Quiver arrows on dark bg | `'white'` (`'w'`) |
| Quiver arrows on light bg | `'black'` (`'k'`) |

#### Green's Function Kernel Colors
```python
# Publication style:
colors = ['red', 'deeppink', 'gold']     # phi, psi, chi kernels
# Exploration style:
colors = ['tab:blue', 'tab:orange', 'tab:green']
```

---

### Vector Field Plotting

#### `make_vector_field` Helper (used in all notebooks)
```python
from skimage.measure import block_reduce

def make_vector_field(v, downsample=8, threshold=0.3):
    Y, X = np.mgrid[:v.shape[-2], :v.shape[-1]]
    X = block_reduce(X, (downsample, downsample), np.mean)
    Y = block_reduce(Y, (downsample, downsample), np.mean)
    vx = block_reduce(v[0], (downsample, downsample), np.mean)
    vy = block_reduce(v[1], (downsample, downsample), np.mean)
    mask = vx**2 + vy**2 > threshold**2
    return X[mask], Y[mask], vx[mask], vy[mask]
```
Common parameter values: `downsample` = 8 (default), 14 (SINDy), `6*scale` (movies). `threshold` = 0.3, 0.4, or 0.6.

#### Standard Quiver Settings
```python
qwargs = {'color': 'w', 'width': 0.005, 'scale': 1e1}
# Variants: scale = 1.5e1, 22, or 2e1
```

#### `vector_plot` Helper
```python
def vector_plot(ax, y, scale=1):
    kwargs = {'origin': 'lower', 'cmap': 'inferno', 'vmin': 0.5, 'vmax': 3}
    qwargs = {'color': 'w', 'width': 0.005, 'scale': 1e1}
    threshold = .3
    ax.imshow(np.linalg.norm(y, axis=0), **kwargs)
    ax.quiver(*make_vector_field(y, threshold=threshold, downsample=8*scale), **qwargs)
    ax.set(xticks=[], yticks=[])
```
Pattern: always show force magnitude heatmap (inferno) with white quiver arrows overlaid.

---

### Image Display Conventions

#### Zyxin Images
```python
ax.imshow(x, cmap='Greys_r', vmin=0, vmax=4)    # or vmax=3
```

#### Force Magnitude Images
```python
kwargs = {'origin': 'lower', 'cmap': 'inferno', 'vmin': 0.5, 'vmax': 3}
ax.imshow(np.linalg.norm(y, axis=0), **kwargs)
```

#### Axis Formatting for Image Panels
```python
ax.set(xticks=[], yticks=[])   # remove tick marks
# or:
ax.axis('off')                 # also removes spines
```

---

### Colorbars

Colorbars are always drawn as **inset axes** on top of image panels, using white text and outlines for contrast against the dark `inferno` background.

#### Horizontal Colorbar (main figure panels)
```python
ax_hidden = ax.inset_axes([0.0, 0.0, 0.0, 0.0])
img = ax_hidden.imshow([[0, vmax]], cmap="inferno", origin='lower')
img.set_visible(False)
ax_hidden.set_visible(False)
cax = ax.inset_axes([0.63, 0.13, 0.25, 0.03])    # [x0, y0, width, height]
cbar = fig.colorbar(img, cax=cax, ticks=[0, 3], orientation='horizontal')
cbar.ax.xaxis.set_tick_params(color='w', width=1)
cbar.ax.set_xticklabels([0, '%d kPa' % vmax], fontsize=6)
cbar.outline.set_edgecolor('w')
cbar.outline.set_linewidth(.5)
plt.setp(plt.getp(cax, 'xticklabels'), color='w')
```

#### Vertical Colorbar (movie frames)
```python
cax = ax.inset_axes([0.72, 0.05, 0.03, 0.20])
cbar = fig.colorbar(img, cax=cax, ticks=[0.5, 3])
cbar.ax.yaxis.set_tick_params(color='w', size=0., pad=1.)
cbar.ax.set_yticklabels(['0.5', '3 kPa'], fontsize=4)
cbar.outline.set_edgecolor('w')
cbar.outline.set_linewidth(.5)
```

**Rules:** White text, white outline, linewidth 0.5, fontsize 4–6, always uses an invisible hidden axes trick.

---

### Scale Bars

Scale bars are white rectangles with white text, positioned in the lower-right of image panels.

```python
um_per_px = 4 * 0.17    # 0.68 um/pixel (downsampled x4 from 0.17 um/pixel native)
barsize = 20             # micrometers
width = barsize / um_per_px
scale_loc = (xlim[1] - xd*0.07 - width, ylim[0] + yd*0.05)
rect = patch.Rectangle(xy=scale_loc, width=width,
                        height=(2*800/960)/um_per_px, color='w')
ax.add_patch(rect)
ax.text(scale_loc[0] + barsize/(2*um_per_px),
        (scale_loc[1] + (2*800/960)/um_per_px)*1.05,
        f'{barsize} $\\mu m$', color='w', ha='center', va='bottom', fontsize=7)
```

For movie frames: `um_per_px = 0.17` (no downsampling factor), `fontsize=4`.

---

### Annotations and Text Overlays

#### Panel Labels on Images
White text for dark backgrounds (force images), black for light backgrounds (scalar fields):
```python
x0, y0 = 0.03, 0.97
fig.text(x0, y0, '$\\zeta(\\mathbf{r})$',
         ha='left', va='top', color='white', transform=ax.transAxes)
fig.text(x0, y0, '$f_{\\phi}$',
         ha='left', va='top', color='black', transform=ax.transAxes)
```

#### Sub-panel Letter Labels
Bold letters, positioned outside the axes frame:
```python
ax.text(-0.36, 1.1, 'a', transform=ax.transAxes, fontweight='bold')
# or via fig.text:
fig.text(0.01, 1, 'a', ha='left', va='center', fontweight='bold')
```

#### Wash-in/Wash-out Annotations
Green dashed vertical lines with rotated text:
```python
ax.axvline(30, color='green', zorder=-10, ls='--')
ax.text(28, -0.02, "+ROCK inhib.", fontsize=5,
        ha='right', va='bottom', color='green', rotation=90)
```

#### Reference Lines
```python
ax.axhline(0.1, color='grey', linestyle='--', zorder=-1)   # threshold
ax.axvline(x=15, color='red', linestyle='--')               # training boundary
ax.fill_betweenx(y=ax.get_ylim(), x1=0, x2=15,
                 color='red', alpha=0.3)                     # shaded training window
```

---

### Legend Styling

Consistent rules across all notebooks:
```python
plt.rcParams['legend.framealpha'] = 0.        # always transparent
plt.rcParams['legend.fontsize'] = font_size - 2
plt.rcParams['legend.handletextpad'] = 0.5
plt.rcParams['legend.handlelength'] = .5      # 1.0 or 1.25 for presentation
plt.rcParams['legend.columnspacing'] = .5
plt.rcParams['legend.borderpad'] = 0.
```

Placement patterns:
```python
ax.legend(ncol=3, bbox_to_anchor=[0.5, 1], loc='lower center')   # centered above
ax.legend(loc='upper right')                                       # inside, top-right
```

---

### Line Plot Patterns

#### Time-Series Force Plots
```python
ax.plot(t, F_exp, color='grey', label='Exp')
ax.plot(t, F_gfnn, color='blue', label='GFNN')
ax.plot(t, F_sindy, color='red', label='Equation')
```

#### Animated Time-Series (movie frames)
Full trajectory dashed, progress solid, scatter dot at current time:
```python
ax.plot(t, F, color='grey', linestyle='--')
ax.plot(t[:tmax], F[:tmax], color='grey', lw=1., label='Exp')
ax.scatter(t[tmax], F[tmax], color='grey', s=5)
```

#### Green's Function Radial Profiles
```python
ax.plot(r * conversion, G_r, label=labels[i], color=colors[i], linewidth=1.5)
```

#### Analytic References
```python
ax.plot(r, G_r, label=r'$-\frac{1}{2\pi} \log(r)$', color='black', linestyle='--')
```

---

### Time-Series Axis Formatting

```python
ax.set(xlim=[0, 120], xticks=[0, 40, 80, 120], xlabel='Time (min)')
ax.set(ylim=[-0.05, 1.05], yticks=[0, 0.5, 1], ylabel='Total Force (A.U.)')
```

Green's function radial axes:
```python
ax.set(xlim=[0, rmax], xticks=[0, rmax//2, rmax], xlabel='r ($\\mu m$)')
ax.set(ylim=[-0.05, 1.05], yticks=[0, 0.5, 1], ylabel='G(r) (A.U.)')
```

---

### Receptive Field Circle Overlays

Circles drawn on image panels showing the effective radius of learned Green's functions:
```python
c = plt.Circle(center, radius, edgecolor=color, fill=False)
ax.add_patch(c)
# radius = pixel value where G(r) drops below 0.1
```

---

### Figure Sizes and Layouts

| Purpose | figsize | dpi | Layout |
|---|---|---|---|
| Main results panel (2×6) | `(7.2, 2.5)` | 200 | `constrained_layout=True` |
| Exploratory fields (3×4) | `(6.5, 5)` | 150 | `constrained_layout=True` |
| Movie frame (1×4) | `(4, 1.5)` | 300 | `constrained_layout=True` |
| Clebsch residual (4×3) | `(3, 4)` | 200 | `constrained_layout=True` |
| Time-series comparison (1×2) | `(4, 2)` | 250 | `tight_layout()` |
| Coulomb demo (1×5) | `(6, 1.5)` | 300 | `constrained_layout=True` |
| SINDy broken axis | `(2, 2)` | 200 | GridSpec |
| SINDy cell detail (1×3) | `(6.5, 3)` | 150 | default |

**Rules:** Prefer `constrained_layout=True` over `tight_layout()`. DPI ranges from 150 (exploration) to 300 (final figures/movies).

---

### Broken Axis Pattern (SINDyAddingTerms)

Used when x-axis has a large gap (e.g., 0–150 and ~10^6):
```python
fig = plt.figure(dpi=200, figsize=(2, 2))
gs = matplotlib.gridspec.GridSpec(1, 2, width_ratios=[1, 0.2])
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharey=ax1)

ax1.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax1.yaxis.tick_left()
ax2.yaxis.tick_right()
ax2.tick_params(labelright=False)

# Diagonal break lines
d = .03
wspace = 0.08
plt.subplots_adjust(wspace=wspace)
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
ax1.plot((1 + wspace/2. - d, 1 + wspace/2. + d), (-d, +d), **kwargs)
ax1.plot((1 - d, 1 + d), (1-d, 1+d), **kwargs)
ax1.plot((1 + wspace/2. - d, 1 + wspace/2. + d), (1-d, 1+d), **kwargs)
```

---

### Saving Figures

```python
# SVG for publication figures (vector, no DPI needed):
plt.savefig('Figures/Fig4Row.svg', bbox_inches='tight')

# PNG for raster/movie output:
plt.savefig('frame_%03d.png' % idx, bbox_inches='tight', dpi=300)

# PDF with TrueType fonts:
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.savefig('Figures/SIFig_Clebsch.pdf', bbox_inches='tight')
```

**Rules:**
- Always use `bbox_inches='tight'`.
- SVG for final publication figures.
- PNG at 300 DPI for movie frames / raster.
- PDF with fonttype 42 when PDF output needed.

---

### Reusable Helper Functions

| Function | Location | Purpose |
|---|---|---|
| `make_vector_field(v, downsample, threshold)` | All notebooks | Downsamples 2D vector field, masks by magnitude threshold |
| `vector_plot(ax, y, scale)` | GFNNCellMechanics | imshow magnitude + quiver overlay on single axis |
| `fields_plot(model, x, y0, mask)` | GFNNCellMechanics | 3×4 grid: zyxin, F_exp, F_pred, kernel profiles |
| `plot_greens_functions(ax, ax_G, model, ...)` | GFNNCellMechanics | Radial G(r) curves + receptive field circles |
| `receptive_field(ax, center, G_r, color, rval)` | GFNNCellMechanics | Draws `plt.Circle` at threshold radius |
| `sum_force_plot(ax, ...)` | GFNNCellMechanics | MinMaxScaler + time-series total force |
| `add_colorbar(ax)` | GFNNCellMechanics | White inset colorbar on dark image |
| `add_scalebar(ax)` | GFNNCellMechanics | White scale bar + µm label |
| `smooth(x, window)` | All notebooks | 1D moving average via `np.convolve` |
| `rescale(x)` | GFNNCellMechanics | Min-max normalize to [0, 1] |
| `plot_force(ax, x, f, v, scale)` | CoulombElectrostatics | pcolormesh + scatter charges + quiver |
| `scale_equiv_plot(ax, df)` | SINDy_11cell1 | MinMaxScaler time-series, 3-line comparison |

---

### Font Size Reference

| Context | Size |
|---|---|
| Base font (`font.size`) | 6–8 (publication), 10–14 (presentation) |
| Axis labels (`axes.labelsize`) | 6–8 (publication), 10 (presentation) |
| Tick labels | 6–7 (publication) |
| Legend | `font.size - 2` (always) |
| Scale bar text | 4 (movie), 7 (main figure) |
| Colorbar tick labels | 4–6 |
| Wash-in annotations | 5 |
| Panel letter labels | Same as `font.size`, `fontweight='bold'` |
| Movie title | 9, `weight='heavy'` |

---

### Physical Units and Conversions

| Quantity | Value | Notes |
|---|---|---|
| Native pixel size | 0.17 µm/pixel | From microscope |
| Downsampled pixel size | 0.68 µm/pixel | At `downsample=4` (0.17 × 4) |
| Force units | kPa | Traction stress |
| Time axis | minutes | Frame interval varies per cell |
| Scale bar default | 20 µm | White bar in lower-right |
