"""
to use: pytest --mpl-generate-path=tests/baseline_images/test_plot
or all: pytest --mpl
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt

# Assuming your package is installed or in the path
from fishplotpy import FishPlotData, fishplot, draw_legend
from fishplotpy.data import DEFAULT_COLORS


# Define the input data from the simple R example
TIMEPOINTS = [0, 30, 75, 150]
FRAC_TABLE = np.array([
    [100,  2,  2, 98],
    [ 45,  0,  0,  0],
    [  0,  0,  2, 95],
    [  0,  0,  1, 40]
])
# Note: R parent indices are 1-based. Keep them 1-based for input.
PARENTS = [0, 1, 1, 3]
CLONE_LABELS = ["Clone 1", "Clone 2", "Clone 3", "Clone 4"]
SAMPLE_TIMES = [0, 150]
VLAB = ["day 0", "day 150"]  # Match R example's vlab

CLONE_ANNOTS = ["TP53,MET", "NF1", "", "8q+,6p-"]


# --- Data for Panel B Test ---
TIMEPOINTS_B = [0, 120]
FRAC_TABLE_B = np.array([
    [100, 100], # R Clone 1: parent 0
    [ 98,  44], # R Clone 2: parent 1
    [ 46,   1], # R Clone 3: parent 2
    [  1,  55]  # R Clone 4: parent 1
])
# R parents = c(0,1,2,1)
PARENTS_B = [0, 1, 2, 1]
COLORS_B = ["#1B9E77", "#D95F02", "#7570B3", "#E7298A"] # R's colors
VLAB_B = ["Primary", "Post-AI"]

# --- Data for Panel C Test ---
TIMEPOINTS_C = [0, 34, 69, 187, 334, 505, 530]
FRAC_TABLE_C = np.array([
    # Transposed for easier numpy entry (rows are timepoints here)
    [99.0,   1.0,   3.0,   1.0,   3.0,  80.0,   0.1],    # Time 0
    [30.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0],    # Time 34
    [ 2.0,   0.1,   2.5,   0.9,   0.9,  76.0, 0.005],  # Time 69
    [60.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0],    # Time 187
    [ 0.0,   0.0,   0.0,   0.0,   0.1,  60.0, 0.001],  # Time 334
    [ 2.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0],    # Time 505
    [ 1.0,   1.0,   1.0,  10.0,  20.0,  15.0,   0.0]     # Time 530
])
# R parents = c(0,1,1,1,3,4,0)
PARENTS_C = [0, 1, 1, 1, 3, 4, 0]
COLORS_C = ["#888888", "#EF0000", "#8FFF40", "#FF6000", "#50FFAF", "#FFCF00", "#0070FF"]  # R's colors
VLINES_C = [0, 34, 69, 187, 334, 505, 530, 650, 750]  # Note: includes extra lines beyond data

# --- Data for Figure 2 Panel B Test (Annotations) ---
TIMEPOINTS_F2B = [0, 30, 200, 423]
# R matrix:
# c(100, 38, 24, 00,
#   002, 00, 01, 00,
#   002, 00, 01, 01,
#   100, 00, 98, 30)
# R fills by column:
FRAC_TABLE_F2B = np.array([
    [100,  2,  2, 100], # Clone 1 (P0)
    [ 38,  0,  0,   0], # Clone 2 (P1)
    [ 24,  1,  1,  98], # Clone 3 (P1)
    [  0,  0,  1,  30]  # Clone 4 (P3)
])
PARENTS_F2B = [0, 1, 1, 3] # R parents = c(0,1,1,3)
ANNOTS_F2B = ["DNMT3A,FLT3", "NPM1", "MET", "ETV6,WNK1-WAC,\nMYO18B"]
VLINES_F2B = [0, 423]
VLAB_F2B = [0, 423]


# Decorator for visual regression test using pytest-mpl
# tolerance=15 is often needed due to slight differences in rendering/aliasing
# between runs or systems, or slight differences from R's base graphics. Adjust as needed.
@pytest.mark.mpl_image_compare(baseline_dir='baseline_images/test_plot',
                               filename='simple_spline_plot.png',
                               tolerance=15)
def test_simple_spline_plot():
    """
    Tests creating a simple fishplot using the 'spline' shape,
    matching the first example in R's test.R output pdf.
    """
    # 1. Create FishPlotData object
    fish_data = FishPlotData(
        frac_table=FRAC_TABLE,
        parents=PARENTS,
        timepoints=TIMEPOINTS,
        clone_labels=CLONE_LABELS
    )

    # Assert basic properties (optional unit test part)
    assert fish_data.n_clones == 4
    assert fish_data.n_timepoints == 4
    assert not fish_data.has_layout

    # 2. Calculate layout
    fish_data.layout_clones()
    assert fish_data.has_layout
    assert len(fish_data.xpos) == fish_data.n_clones
    assert len(fish_data.ytop) == fish_data.n_clones
    assert len(fish_data.ybtm) == fish_data.n_clones
    # Check if coordinates were generated (example check on first clone)
    assert len(fish_data.xpos[0]) > 0

    # 3. Create the plot
    # Create a figure explicitly for the test
    fig, ax = plt.subplots(figsize=(8, 4))  # Match R's pdf aspect ratio roughly

    # Call fishplot (R example uses shape="spline", solid bg for this specific one)
    fig, ax = fishplot(
        fish_data,
        ax=ax,
        shape="spline",
        title_btm="633734", # From R example's pdf test
        vlines=SAMPLE_TIMES,
        vlab=VLAB,
        cex_vlab=0.7, # Match R cex.vlab
        bg_type="solid", # R example used solid for this test case
        bg_col="white" # Default solid is often white
    )

    # Assertions after plotting (optional)
    assert fig is not None
    assert ax is not None
    assert ax.get_ylim() == (0, 100)

    # Return the figure for pytest-mpl comparison
    # NOTE: DO NOT call plt.show() or fig.savefig() here when using pytest-mpl
    return fig


@pytest.mark.mpl_image_compare(baseline_dir='baseline_images/test_plot',
                               filename='simple_spline_plot_with_legend.png',
                               tolerance=15)
def test_simple_plot_with_legend():
    """
    Tests creating the simple plot and adding a legend below it.
    """
    # --- Setup fish data and layout (same as previous test) ---
    fish_data = FishPlotData(
        frac_table=FRAC_TABLE,
        parents=PARENTS,
        timepoints=TIMEPOINTS,
        clone_labels=CLONE_LABELS
    )
    fish_data.layout_clones()

    # --- Create the plot ---
    fig, ax = plt.subplots(figsize=(8, 5))  # Slightly taller figure for legend space
    fig, ax = fishplot(
        fish_data,
        ax=ax,
        shape="spline",
        title_btm="633734",
        vlines=SAMPLE_TIMES,
        vlab=VLAB,
        cex_vlab=0.7,
        bg_type="solid",
        bg_col="white"
    )

    # --- Add the legend ---
    # Place legend below the plot using bbox_to_anchor
    draw_legend(fish_data, fig=fig, ax=ax,  # Pass fig and ax for context
                loc='upper center', # Place anchor point at upper center of legend box
                bbox_to_anchor=(0.5, 0.1),  # Anchor legend at x=0.5, y=0.1 in figure coords
                ncol=4)  # Try 4 columns for 4 clones

    # Adjust layout to prevent legend overlap (may need tweaking)
    fig.tight_layout(rect=[0, 0.1, 1, 1])  # Leave space at bottom (y=0.1)

    return fig

# --- Add more tests here for other shapes, options, annotations, edge cases ---

# Example test for polygon shape
@pytest.mark.mpl_image_compare(baseline_dir='baseline_images/test_plot',
                               filename='simple_polygon_plot.png',
                               tolerance=15)
def test_simple_polygon_plot():
    """Tests creating a simple fishplot using the 'polygon' shape."""
    fish_data = FishPlotData(frac_table=FRAC_TABLE, parents=PARENTS, timepoints=TIMEPOINTS)
    fish_data.layout_clones()
    fig, ax = plt.subplots(figsize=(8, 4))
    fig, ax = fishplot(fish_data, ax=ax, shape="polygon", vlines=SAMPLE_TIMES, vlab=VLAB)
    return fig


# --- Test for Annotations ---
@pytest.mark.mpl_image_compare(baseline_dir='baseline_images/test_plot',
                               filename='simple_plot_with_annotations_outline.png',  # New filename
                               tolerance=15)
def test_simple_plot_with_annotations_outline():
    """
    Tests creating a simple plot with clone annotations using an outline
    for better readability. Based on Fig 2A from R test script.
    """
    fish_data = FishPlotData(
        frac_table=FRAC_TABLE,
        parents=PARENTS,
        timepoints=TIMEPOINTS,
        clone_labels=CLONE_LABELS,
        clone_annots=CLONE_ANNOTS
    )
    fish_data.layout_clones()
    fig, ax = plt.subplots(figsize=(5, 5))

    # Call fishplot with use_annot_outline=True
    fig, ax = fishplot(
        fish_data,
        ax=ax,
        shape="spline",
        title_btm="633734",
        vlines=SAMPLE_TIMES,
        vlab=SAMPLE_TIMES,
        cex_vlab=0.7,
        bg_type="solid",
        bg_col="white",
        use_annot_outline=True  # <<< Enable outline
    )
    return fig

# --- Test for Custom Annotation Styling ---
@pytest.mark.mpl_image_compare(baseline_dir='baseline_images/test_plot',
                               filename='custom_annotation_style.png',
                               tolerance=15)
def test_custom_annotation_style():
    """
    Tests customizing annotation position, color, angle, and offset.
    """
    # Use slightly different annotations for clarity
    custom_annots = ["Founder", "Sub1", "Sub2", "Sub4"]

    # Customize styling: move pos=4 (right), change color, angle, size, offset
    fish_data = FishPlotData(
        frac_table=FRAC_TABLE,
        parents=PARENTS,
        timepoints=TIMEPOINTS,
        clone_labels=CLONE_LABELS,
        clone_annots=custom_annots,
        clone_annots_angle=30,   # Rotate labels
        clone_annots_col='darkgreen',  # Change color
        clone_annots_pos=4,      # Position to the right
        clone_annots_cex=0.9,    # Increase size slightly
        clone_annots_offset=0.5  # Increase offset
    )
    fish_data.layout_clones()
    fig, ax = plt.subplots(figsize=(5.5, 5))  # Adjust size if needed

    fig, ax = fishplot(
        fish_data,
        ax=ax,
        shape="spline",
        vlines=SAMPLE_TIMES,
        vlab=SAMPLE_TIMES,
        use_annot_outline=False  # Test without outline here
    )
    return fig


# --- Test for Polygon Shape, Ramp Angle, Padding (Panel B) ---
@pytest.mark.mpl_image_compare(baseline_dir='baseline_images/test_plot',
                               filename='panel_b_polygon.png',
                               tolerance=15)
def test_panel_b_polygon_shape():
    """
    Tests polygon shape, custom ramp_angle, pad_left_frac, and colors.
    Based on Fig 1B from R test script.
    """
    fish_data = FishPlotData(
        frac_table=FRAC_TABLE_B,
        parents=PARENTS_B,
        timepoints=TIMEPOINTS_B,
        colors=COLORS_B
    )
    fish_data.layout_clones()  # Default: separate_independent_clones=False

    fig, ax = plt.subplots(figsize=(5, 4))  # Adjust size

    fig, ax = fishplot(
        fish_data,
        ax=ax,
        shape="polygon",
        vlines=TIMEPOINTS_B,  # Match R example
        vlab=VLAB_B,
        title_btm="BRC32",
        ramp_angle=1.0,  # R uses 1 for this panel
        pad_left_frac=0.3,  # R uses 0.3 for this panel
        cex_vlab=0.8,  # Match R
        cex_title=0.7,  # Match R
        # Using default background gradient
    )
    return fig


# --- Test for Separate Independent Clones (Panel C) ---
@pytest.mark.mpl_image_compare(baseline_dir='baseline_images/test_plot',
                               filename='panel_c_separate_clones.png',
                               tolerance=20)  # Increase tolerance slightly for complex plot
def test_panel_c_separate_clones():
    """
    Tests layout with separate_independent_clones=True and spline shape.
    Based on Fig 1C from R test script.
    """
    fish_data = FishPlotData(
        frac_table=FRAC_TABLE_C,
        parents=PARENTS_C,
        timepoints=TIMEPOINTS_C,
        colors=COLORS_C,
        # Add fix_missing_clones=True because data C has some 0s between non-zeros (e.g. clone 6)
        fix_missing_clones=True
    )

    # Calculate layout WITH separation
    fish_data.layout_clones(separate_independent_clones=True)

    fig, ax = plt.subplots(figsize=(8, 5))  # Wider plot for many timepoints

    fig, ax = fishplot(
        fish_data,
        ax=ax,
        shape="spline",
        vlines=VLINES_C,  # Use the extended vlines from R
        vlab=VLINES_C,  # Label with the vline values
        title_btm="AML31",
        cex_vlab=0.9,  # Match R
        # Using default background gradient
    )
    return fig


# --- Test for Bezier Shape ---
@pytest.mark.mpl_image_compare(baseline_dir='baseline_images/test_plot',
                               filename='simple_bezier_plot.png',
                               tolerance=25)  # Bezier might differ more, increase tolerance
def test_simple_bezier_plot():
    """Tests creating a simple fishplot using the 'bezier' shape."""
    # Using the initial simple dataset (FRAC_TABLE, PARENTS, TIMEPOINTS)
    fish_data = FishPlotData(frac_table=FRAC_TABLE, parents=PARENTS, timepoints=TIMEPOINTS)
    fish_data.layout_clones()
    fig, ax = plt.subplots(figsize=(8, 4))
    fig, ax = fishplot(fish_data, ax=ax, shape="bezier", vlines=SAMPLE_TIMES, vlab=VLAB)
    return fig


# --- Test for Legend Customization ---
@pytest.mark.mpl_image_compare(baseline_dir='baseline_images/test_plot',
                               filename='custom_legend_spacing.png',
                               tolerance=15)
def test_custom_legend_spacing():
    """ Tests customizing legend spacing parameters. """
    # Using the initial simple dataset
    fish_data = FishPlotData(
        frac_table=FRAC_TABLE,
        parents=PARENTS,
        timepoints=TIMEPOINTS,
        clone_labels=CLONE_LABELS
    )
    fish_data.layout_clones()
    fig, ax = plt.subplots(figsize=(8, 5))

    # Adjust bottom margin *before* adding legend
    fig.subplots_adjust(bottom=0.25)  # Increase bottom margin to 25% of fig height

    fig, ax = fishplot(fish_data, ax=ax, shape="spline", vlines=SAMPLE_TIMES, vlab=VLAB)

    draw_legend(fish_data, fig=fig, ax=ax,
                loc='upper center',  # Keep anchor point on legend (upper center)
                bbox_to_anchor=(0.5, 0.15),  # Place anchor point at X=0.5, Y=0.15 in FIGURE coords
                ncol=4,
                labelspacing=0.2,
                columnspacing=0.5,
                handletextpad=0.4,
                handlelength=1.5
               )
    # Remove tight_layout as it might interfere
    # fig.tight_layout(rect=[0, 0.1, 1, 1])
    return fig


# --- Test for Clone Order Robustness ---
@pytest.mark.mpl_image_compare(baseline_dir='baseline_images/test_plot',
                               filename='shuffled_clone_order.png',
                               save_diff_figure=True,
                               tolerance=15)
def test_shuffled_clone_order():
    """
    Tests if shuffling fraction table row order affects the final plot,
    keeping original parent array but shuffling labels/colors.
    """
    n_clones = FRAC_TABLE.shape[0]
    original_indices = list(range(n_clones))

    # --- Use the specific shuffle from R's ordertest ---
    # R used frac.table[c(4,3,2,1),], which corresponds to 0-based indices [3, 2, 1, 0]
    shuffled_indices = [3, 2, 1, 0]

    # --- Create mapping dictionaries ---
    map_orig_idx_to_shuffled_idx = {orig: shuff for orig, shuff in zip(original_indices, shuffled_indices)}
    map_shuffled_idx_to_orig_idx = {shuff: orig for orig, shuff in zip(original_indices, shuffled_indices)}

    # --- Shuffle data ---
    shuffled_fracs = FRAC_TABLE[shuffled_indices, :]
    shuffled_labels = [f"Shuffled Clone {i+1}" for i in range(n_clones)]

    # --- Assign default colors based on ORIGINAL order and then shuffle them ---
    original_colors = DEFAULT_COLORS[:n_clones]
    shuffled_colors = [original_colors[map_shuffled_idx_to_orig_idx[i]] for i in range(n_clones)]

    # --- Correctly Remap Parent IDs ---
    shuffled_parents = np.zeros(n_clones, dtype=int)
    original_parents_ids = PARENTS  # [0, 1, 1, 3]

    # (Using Take 6 logic which calculates [2, 4, 4, 0] for this specific shuffle)
    for original_idx in range(n_clones):
        shuffled_idx = map_orig_idx_to_shuffled_idx[original_idx]
        original_parent_id = original_parents_ids[original_idx]
        if original_parent_id == 0:
            new_parent_id = 0
        else:
            original_parent_idx = original_parent_id - 1
            new_parent_shuffled_idx = map_orig_idx_to_shuffled_idx[original_parent_idx]
            new_parent_id = new_parent_shuffled_idx + 1
        shuffled_parents[shuffled_idx] = new_parent_id
    # For shuffle [3, 2, 1, 0], this calculates [2, 4, 4, 0], which matches R test.

    # --- Create FishPlotData ---
    fish_data_shuffled = FishPlotData(frac_table=shuffled_fracs,
                                      parents=shuffled_parents,  # Should be [2, 4, 4, 0]
                                      timepoints=TIMEPOINTS,
                                      clone_labels=shuffled_labels,
                                      colors=shuffled_colors)
    fish_data_shuffled.layout_clones()

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 4))
    fig, ax = fishplot(fish_data_shuffled, ax=ax, shape="spline", vlines=SAMPLE_TIMES, vlab=VLAB)

    return fig


# --- Test for Specific Annotation Style (Fig 2B) ---
@pytest.mark.mpl_image_compare(baseline_dir='baseline_images/test_plot',
                               filename='fig2b_annotation_style.png',
                               tolerance=15)
def test_fig2b_annotation_style():
    """ Tests specific annotation style from R test Fig 2B. """
    fish_data = FishPlotData(
        frac_table=FRAC_TABLE_F2B,
        parents=PARENTS_F2B,
        timepoints=TIMEPOINTS_F2B,
        clone_annots=ANNOTS_F2B,
        clone_annots_angle=30,  # Specific angle
        clone_annots_col="green",  # Specific color
        # Using default pos=2, cex=0.7, offset=0.2 from R example
    )
    fish_data.layout_clones()
    fig, ax = plt.subplots(figsize=(6, 5))  # Adjust size

    # Plot settings from R test
    fig, ax = fishplot(
        fish_data,
        ax=ax,
        shape="spline",
        vlines=VLINES_F2B,
        vlab=VLAB_F2B,
        title="Sample 150288",  # Add title from R test
        cex_title=0.9,
        cex_vlab=0.8,
        # Default background gradient
        use_annot_outline=True  # Add outline for green text maybe? Optional.
    )

    return fig
