# fishplotpy

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
<!-- Add other badges if you set up CI/CD, PyPI release, etc. -->

A Python implementation for visualizing clonal evolution dynamics using "fish plots" (also known as Muller plots).

This package provides tools to create visualizations showing the temporal changes in the frequencies of clones within a population, often used in cancer genomics and evolutionary biology.

This package is a translation of the R package [`fishplot`](https://github.com/chrisamiller/fishplot) by Chris Miller et al.

![fishplot_showcase](https://github.com/user-attachments/assets/8e38d020-0658-4328-a21e-a9c44730967e)


## Features

*   Calculates plot layout based on clonal fractions and parent-child relationships.
*   Supports multiple shapes for drawing clones:
    *   `polygon` (with adjustable starting ramp)
    *   `spline` (smooth curves, recommended)
    *   `bezier` (Note: Visual output may differ from the R version's `Hmisc::bezier`)
*   Customizable appearance: colors, background (solid/gradient), borders.
*   Optional vertical lines and timepoint labels.
*   Clone annotations (e.g., for mutations) with customizable position, angle, color, size.
*   Legend generation.
*   Option to add spacing between independent founder clones.

## Installation

```bash
# Option 1: Install from source after cloning
git clone https://github.com/sayitobar/fishplotpy.git
cd fishplotpy
pip install .

# Option 2: Install directly from GitHub
pip install git+https://github.com/sayitobar/fishplotpy.git
```

**Dependencies:**
*   numpy
*   pandas
*   matplotlib
*   scipy

## Basic Usage

```python
# Basic Usage Example for README.md - Refer to usage.py for more complex usage.

import numpy as np
import matplotlib.pyplot as plt
from fishplotpy import FishPlotData, fishplot

# 1. Define Minimal Input Data
# Timepoints for measurements
timepoints = [0.0, 30.0, 75.0, 150.0]

# Clonal fractions (clones = rows, timepoints = columns)
# Using the corrected data from the simple example
frac_table = np.array([
    [100,  2,  2, 98],  # Clone 1 (Parent 0)
    [ 45,  0,  0,  0],  # Clone 2 (Parent 1)
    [  0,  0,  2, 95],  # Clone 3 (Parent 1)
    [  0,  0,  1, 40]   # Clone 4 (Parent 3)
])

# Define parent-child relationships (0 for founders, otherwise 1-based index)
parents = [0, 1, 1, 3]

# 2. Create the FishPlotData object
# This holds the data and calculates necessary info like nesting levels
fp_data = FishPlotData(
   frac_table=frac_table,
   parents=parents,
   timepoints=timepoints
)

# 3. Calculate the plotting layout
# This determines the vertical positions and shapes
fp_data.layout_clones()

# 4. Create the fish plot
# Create a Matplotlib figure and axes
fig, ax = plt.subplots(figsize=(7, 4.5))

# Generate the plot onto the axes
fishplot(fp_data, ax=ax, shape="spline") # Using recommended spline shape

# 5. Display the plot
plt.tight_layout()
plt.show()

```

## Key Options

*   **`FishPlotData(...)`**:
    *   `colors`: List of color strings for clones.
    *   `clone_annots`: List of annotation strings.
    *   `clone_annots_*`: Parameters to style annotations (angle, col, pos, cex, offset).
    *   `fix_missing_clones`: Set to `True` to handle clones disappearing and reappearing (use with caution, may affect validation).
*   **`layout_clones(...)`**:
    *   `separate_independent_clones`: Set to `True` to add space between founder clones.
*   **`fishplot(...)`**:
    *   `shape`: 'spline', 'polygon', 'bezier'.
    *   `ramp_angle`: Steepness for polygon start (0-1).
    *   `pad_left_frac`: Adjust padding before first timepoint.
    *   `bg_type`: 'gradient', 'solid', 'none'.
    *   `bg_col`: Color string or list of 3 colors for gradient.
    *   `border`, `col_border`: Clone outline style.
    *   `use_annot_outline`: Add white outline to annotations for contrast.
*   **`draw_legend(...)`**: Control legend appearance (ncol, loc, bbox_to_anchor, cex, etc.)

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

This package is a Python translation of the original R `fishplot` package developed by Chris Miller and colleagues. Please cite their work as well if you use this tool.
