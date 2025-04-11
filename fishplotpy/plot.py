import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import matplotlib.path as mpath
import matplotlib.patheffects as pe
import numpy as np
from scipy.interpolate import CubicSpline  # For spline shape
# Note: Using CubicSpline which is common. R uses stats::spline. Results might differ slightly.
from typing import Optional, Sequence, Tuple, Union, Dict, Any, List
import warnings

# Assuming FishPlotData is in the same package or accessible
try:
    from .data import FishPlotData
except ImportError:
    # Allow running script directly for testing if data.py is in sys.path
    from data import FishPlotData


def _map_pos_to_ha_va(pos: int) -> Tuple[str, str]:
    """Maps R-style position codes (1-4) to matplotlib ha/va."""
    if pos == 1:  # Below
        return 'center', 'top'
    elif pos == 2:  # Left
        return 'right', 'center'
    elif pos == 3:  # Above
        return 'center', 'bottom'
    elif pos == 4:  # Right
        return 'left', 'center'
    else:  # Default to right
        return 'left', 'center'


def _annot_clone(ax: plt.Axes, x: float, y: float, annot: str,
                 angle: float, col: str, pos: int, cex: float, offset: float, use_outline: bool = False):
    """Adds annotation text to the plot."""
    if not annot:  # Skip if annotation is empty
        return

    ha, va = _map_pos_to_ha_va(pos)
    # Use a base font size and scale with cex, or pass cex directly if using relative units
    # Let's assume cex is a multiplier for a default reasonable size e.g., 8
    fontsize = plt.rcParams['font.size'] * cex  # Scale default font size

    # Basic offset implementation:
    # Calculate offset in display coordinates (pixels) and convert back? Simpler:
    # Approximate offset in data coordinates based on position.
    # This is a rough heuristic. A fixed pixel offset would be more robust but complex.
    # Let's use a fraction of axis range as offset unit.
    x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    offset_scale_x = x_range * 0.01  # Scale offset relative to x-axis range
    offset_scale_y = y_range * 0.01  # Scale offset relative to y-axis range

    x_off, y_off = 0.0, 0.0
    if pos == 1:  # Below -> offset pushes down
        y_off = -offset * offset_scale_y
    elif pos == 2:  # Left -> offset pushes left
        x_off = -offset * offset_scale_x
    elif pos == 3:  # Above -> offset pushes up
        y_off = offset * offset_scale_y
    elif pos == 4:  # Right -> offset pushes right
        x_off = offset * offset_scale_x

    # Path effects for outline (optional)
    path_effects = None
    if use_outline:
        # Outline with white, adjust linewidth as needed
        path_effects = [pe.withStroke(linewidth=1.5, foreground='white')]

    ax.text(x + x_off, y + y_off, annot,  # Apply offset
            ha=ha, va=va,
            rotation=angle,
            color=col,
            fontsize=fontsize,
            rotation_mode='anchor',  # Rotate around the anchor point (ha/va)
            transform=ax.transData,
            path_effects=path_effects  # Add path effects
            )


def _draw_clust_polygon(ax: plt.Axes, xpos: np.ndarray, ytop: np.ndarray, ybtm: np.ndarray,
                        color: str, nest_level: int, pad_left: float,
                        border: float, col_border: str, ramp_angle: float,
                        annot_data: Dict[str, Any], clone_idx: int, fish_data: FishPlotData, use_outline: bool = False):
    """Draws a single clone using polygons."""
    if len(xpos) == 0:
        return  # Nothing to draw

    # Calculate origin
    xst = xpos[0] - pad_left * (0.6 ** nest_level)
    yst = (ytop[0] + ybtm[0]) / 2.0
    # Store origin
    if fish_data is not None and clone_idx < len(fish_data.xst_yst):
        fish_data.xst_yst[clone_idx] = (xst, yst)

    # R's angled rampup logic
    xangle = (xst + xpos[0]) / 2.0
    yangle_top = yst + abs(yst - ytop[0]) * ramp_angle
    yangle_btm = yst - abs(yst - ybtm[0]) * ramp_angle

    # Create polygon vertices
    # Path: Start -> AngleBottom -> BottomPoints -> ReverseTopPoints -> AngleTop -> Start
    x_coords = np.concatenate(([xst, xangle], xpos, xpos[::-1], [xangle, xst]))
    y_coords = np.concatenate(([yst, yangle_btm], ybtm, ytop[::-1], [yangle_top, yst]))

    polygon = patches.Polygon(
        np.column_stack((x_coords, y_coords)),
        facecolor=color,
        edgecolor=col_border,
        linewidth=border
    )
    ax.add_patch(polygon)

    # Add annotation
    if annot_data['annot']:
        _annot_clone(ax, xst, yst, annot=annot_data['annot'], angle=annot_data['angle'],
                     col=annot_data['col'], pos=annot_data['pos'], cex=annot_data['cex'],
                     offset=annot_data['offset'], use_outline=use_outline)


def _draw_clust_bezier(ax: plt.Axes, xpos: np.ndarray, ytop: np.ndarray, ybtm: np.ndarray,
                       color: str, nest_level: int, pad_left: float,
                       border: float, col_border: str,
                       annot_data: Dict[str, Any], clone_idx: int, fish_data: FishPlotData, use_outline: bool = False):
    """Draws a single clone using Bezier curves (via matplotlib Path)."""
    if len(xpos) == 0:
        return

    # R implementation Hmisc::bezier adds 'flank' points.
    # Let's replicate this logic by adding extra points around the originals.
    if len(xpos) > 1:
        x_range = np.max(xpos) - np.min(xpos)
        flank = x_range * 0.01  # R's flank calculation
    else:
        flank = 0.01  # Default small flank if only one point

    # Expand points with flanks (R uses rbind trick, we use repeat/reshape)
    # Structure: x-2f, x-f, x, x+f, x+2f for each original x
    num_pts = len(xpos)
    x_rep = np.repeat(xpos, 5)
    flank_mult = np.tile(np.array([-2, -1, 0, 1, 2]), num_pts)
    x_flanked = x_rep + flank_mult * flank

    ytop_flanked = np.repeat(ytop, 5)
    ybtm_flanked = np.repeat(ybtm, 5)

    # Starting point
    xst = xpos[0] - pad_left * (0.6 ** nest_level)
    yst = (ytop[0] + ybtm[0]) / 2.0
    # Store origin
    if fish_data is not None and clone_idx < len(fish_data.xst_yst):
        fish_data.xst_yst[clone_idx] = (xst, yst)

    # Create Path vertices and codes for Bezier
    # We need CUBIC4 Bezier: 1 start point, 3 control points per segment.
    # The flank points in R seem to influence the curve shape differently than
    # standard Bezier control points. Hmisc::bezier might be doing something custom.
    # A simpler approach using matplotlib Path with standard quadratic/cubic Beziers
    # might not replicate Hmisc::bezier exactly.
    # Let's try using Path with the flanked points directly as curve vertices.

    # Top path: Start -> Flanked Top Points
    x_top_curve = np.concatenate(([xst], x_flanked))
    y_top_curve = np.concatenate(([yst], ytop_flanked))

    # Bottom path: Start -> Flanked Bottom Points
    x_btm_curve = np.concatenate(([xst], x_flanked))
    y_btm_curve = np.concatenate(([yst], ybtm_flanked))

    # Combine top and reversed bottom paths
    verts = np.concatenate(
        (np.column_stack((x_top_curve, y_top_curve)),
         np.column_stack((x_btm_curve, y_btm_curve))[::-1, :])  # Reverse bottom path
    )

    # Create path codes: MOVETO, LINETO for all subsequent points
    codes = [mpath.Path.MOVETO] + [mpath.Path.LINETO] * (len(verts) - 1)

    path = mpath.Path(verts, codes)
    patch = patches.PathPatch(path, facecolor=color, edgecolor=col_border, linewidth=border)
    ax.add_patch(patch)

    # Add annotation at the starting point
    if annot_data['annot']:
        _annot_clone(ax, xst, yst, annot=annot_data['annot'], angle=annot_data['angle'],
                     col=annot_data['col'], pos=annot_data['pos'], cex=annot_data['cex'],
                     offset=annot_data['offset'], use_outline=use_outline)


def _draw_clust_spline(ax: plt.Axes, xpos: np.ndarray, ytop: np.ndarray, ybtm: np.ndarray,
                       color: str, nest_level: int, pad_left: float,
                       border: float, col_border: str,
                       annot_data: Dict[str, Any], clone_idx: int, fish_data: FishPlotData, use_outline: bool = False):
    """Draws a single clone using splined curves."""
    if len(xpos) == 0:
        print("Skipping drawing for clone with no timepoints.")  # Mimic R warning
        return
    if len(xpos) < 2:
        # Cannot draw spline with < 2 points, draw as polygon?
        # Or just skip/draw a point? R's spline might handle this.
        # For now, draw simple polygon if only 1 point.
        if len(xpos) == 1:
            _draw_clust_polygon(ax, xpos, ytop, ybtm, color, nest_level, pad_left,
                                border, col_border, ramp_angle=0.5,  # Use default ramp
                                annot_data=annot_data, fish_data=fish_data, clone_idx=clone_idx,
                                use_outline=use_outline)
        return

    # Replicate R's flank point logic for splines
    x_range = np.max(xpos) - np.min(xpos)
    flank = x_range * 0.001  # Smaller flank for spline in R

    # Expand points with flanks
    num_pts = len(xpos)
    x_rep = np.repeat(xpos, 5)
    flank_mult = np.tile(np.array([-2, -1, 0, 1, 2]), num_pts)
    x_flanked = x_rep + flank_mult * flank

    ytop_flanked = np.repeat(ytop, 5)
    ybtm_flanked = np.repeat(ybtm, 5)

    # Starting point calculation
    xst = xpos[0] - pad_left * (0.6 ** nest_level)
    yst = (ytop[0] + ybtm[0]) / 2.0
    # Store origin
    if fish_data is not None and clone_idx < len(fish_data.xst_yst):
        fish_data.xst_yst[clone_idx] = (xst, yst)

    # R's fix for starts hanging outside parent (heuristic)
    if yst > 85 or yst < 15:
        xst = (xst + xpos[0]) / 2.0

    # R adds flanks around start point too for spline
    xst_flanked = np.array([xst - 2 * flank, xst - flank, xst, xst + flank, xst + 2 * flank])
    yst_flanked = np.repeat(yst, 5)

    # Combine start flanks and data flanks for interpolation
    x_interp_pts = np.concatenate((xst_flanked, x_flanked))
    ytop_interp_pts = np.concatenate((yst_flanked, ytop_flanked))
    ybtm_interp_pts = np.concatenate((yst_flanked, ybtm_flanked))

    # Ensure x points are unique and sorted for spline interpolation
    # Using unique points might alter shape slightly compared to R's direct spline call
    unique_x_top, top_idx = np.unique(x_interp_pts, return_index=True)
    unique_x_btm, btm_idx = np.unique(x_interp_pts, return_index=True)

    if len(unique_x_top) < 4 or len(unique_x_btm) < 4:
        warnings.warn("Not enough unique points for cubic spline after adding flanks. Falling back to polygon drawing.",
                      UserWarning)
        _draw_clust_polygon(ax, xpos, ytop, ybtm, color, nest_level, pad_left,
                            border, col_border, ramp_angle=0.5,
                            annot_data=annot_data, fish_data=fish_data, clone_idx=clone_idx, use_outline=use_outline)
        return

    # Create cubic splines
    # Use bc_type='natural' or 'clamped' if boundary conditions are known/needed. Default is 'not-a-knot'.
    spline_top = CubicSpline(unique_x_top, ytop_interp_pts[top_idx])
    spline_btm = CubicSpline(unique_x_btm, ybtm_interp_pts[btm_idx])

    # Evaluate splines over a finer range for smooth plotting
    x_smooth = np.linspace(np.min(x_interp_pts), np.max(x_interp_pts), 200)
    ytop_smooth = spline_top(x_smooth)
    ybtm_smooth = spline_btm(x_smooth)

    # Clip y values to avoid going outside 0-100 range significantly?
    # ytop_smooth = np.clip(ytop_smooth, -5, 105) # Generous clipping
    # ybtm_smooth = np.clip(ybtm_smooth, -5, 105)

    # Create polygon from the smooth spline curves
    x_poly = np.concatenate((x_smooth, x_smooth[::-1]))
    y_poly = np.concatenate((ybtm_smooth, ytop_smooth[::-1]))

    polygon = patches.Polygon(np.column_stack((x_poly, y_poly)),
                              facecolor=color,
                              edgecolor=col_border,
                              linewidth=border)
    ax.add_patch(polygon)

    # Add annotation at the *central* start point
    if annot_data['annot']:
        _annot_clone(ax, xst, yst, annot=annot_data['annot'], angle=annot_data['angle'],
                     col=annot_data['col'], pos=annot_data['pos'], cex=annot_data['cex'],
                     offset=annot_data['offset'], use_outline=use_outline)


def _draw_background(ax: plt.Axes, bg_type: str, bg_col: Union[str, List[str]]):
    """Draws the plot background (solid or gradient)."""
    if bg_type == "solid":
        if isinstance(bg_col, list):
            bg_col = bg_col[0]  # Use first color if list provided
        if not isinstance(bg_col, str):
            warnings.warn(f"Invalid bg_col '{bg_col}' for solid background. Using white.")
            bg_col = 'white'
        ax.set_facecolor(bg_col)
    elif bg_type == "gradient":
        if not isinstance(bg_col, list) or len(bg_col) != 3:
            warnings.warn("Gradient background requires a list of 3 colors. Using default gradient.")
            # R's default: c("bisque","darkgoldenrod1","darkorange3")
            bg_col = ['bisque', '#FFB90F', '#CD6600']

        # Create a vertical gradient
        cmap = mcolors.LinearSegmentedColormap.from_list("fish_gradient", bg_col)
        # Draw gradient on a rectangle covering the axes
        # Need axes limits to place the image correctly. Use imshow.
        # Get axes limits (usually 0-100 for y)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        # Create an image array: shape (N, M, 3[RGB]), gradient varies vertically
        gradient = np.linspace(1, 0, 256)  # Reversed: darkorange3 at top (100), bisque at bottom (0)
        gradient_map = cmap(gradient)[:, :3]  # Get RGB, drop alpha
        # Reshape for imshow: (rows, 1, 3) -> vertical gradient
        img = gradient_map.reshape(256, 1, 3)

        # Display the image, stretching to axes limits
        ax.imshow(img, aspect='auto', extent=[*xlim, *ylim], origin='lower', zorder=-1)  # zorder=-1 puts it behind data
    else:
        # Default white background if type is invalid or "none"
        ax.set_facecolor('white')


def fishplot(fish_data: FishPlotData,
             ax: Optional[plt.Axes] = None,
             shape: str = "spline",
             vlines: Optional[Sequence[Union[int, float]]] = None,
             vlab: Optional[Sequence[str]] = None,
             col_vline: str = "#cccccc",
             border: float = 0.5,
             col_border: str = "#777777",
             pad_left_frac: float = 0.2,
             ramp_angle: float = 0.5,
             title: Optional[str] = None,
             title_btm: Optional[str] = None,
             cex_title: float = 1.0,  # Multiplier for default font size
             cex_vlab: float = 0.7,  # Multiplier for default font size
             bg_type: str = "gradient",
             bg_col: Union[str, List[str]] = ['bisque', '#FFB90F', '#CD6600'],
             use_annot_outline: bool = False):
    """
    Generates a fishplot visualization (Muller plot) on a Matplotlib axis.

    This function takes a prepared FishPlotData object (where layout has been
    calculated) and draws the fishplot, depicting clonal evolution over time.

    Args:
        fish_data (FishPlotData):
            A FishPlotData object instance where the `layout_clones()` method
            has already been successfully run to calculate coordinates.
        ax (matplotlib.axes.Axes, optional):
            A Matplotlib Axes object to draw the plot onto. If None, a new
            figure and axes are created automatically using `plt.subplots()`.
            Defaults to None.
        shape (str, optional):
            The shape used to draw clone boundaries. Options:
            - 'spline': Smooth curves generated using cubic splines (Recommended).
            - 'polygon': Straight line segments connecting calculated points,
                         with a starting ramp defined by `ramp_angle`.
            - 'bezier': Curves generated using matplotlib Paths based on flanked
                        points. Note: The visual output may differ from the
                        `Hmisc::bezier` implementation in the original R package.
            Defaults to "spline".
        vlines (Sequence[Union[int, float]], optional):
            List or array of x-coordinates (timepoints) at which to draw vertical
            reference lines across the plot. Defaults to None.
        vlab (Sequence[str], optional):
            List of labels corresponding to the `vlines`. Must have the same length
            as `vlines` if provided. Labels are drawn above the plot area.
            Defaults to None.
        col_vline (str, optional):
            Color string for the vertical lines specified by `vlines`.
            Defaults to "#cccccc" (light grey).
        border (float, optional):
            Line width for the border drawn around each clone shape. Set to 0
            to hide borders. Defaults to 0.5.
        col_border (str, optional):
            Color string for the clone shape borders. Defaults to "#777777" (mid-grey).
        pad_left_frac (float, optional):
            Fraction of the total time range to add as padding to the left of the
            first measured timepoint. Controls the length of the "origin ramp".
            Defaults to 0.2 (20%).
        ramp_angle (float, optional):
            Steepness factor (0 to 1) for the initial ramp when `shape='polygon'`.
            A value of 0 creates a horizontal start, 1 creates a steeper angle.
            Only affects 'polygon' shape. Defaults to 0.5.
        title (str, optional):
            Main title for the plot, typically displayed above the plotting area.
            Best added via `fig.suptitle()` or `ax.set_title()` after calling `fishplot`.
            If provided here, it might interfere with auto-layout. Defaults to None.
        title_btm (str, optional):
            Text label displayed at the bottom left, inside the plot area, useful
            for sample IDs or other metadata. Defaults to None.
        cex_title (float, optional):
            Font size multiplier relative to Matplotlib's default for the bottom title
            (`title_btm`) if provided. Defaults to 1.0. Note: For the main plot title,
            set fontsize directly when calling `ax.set_title` or `fig.suptitle`.
        cex_vlab (float, optional):
            Font size multiplier relative to Matplotlib's default for the `vlab`
            labels placed above the plot. Defaults to 0.7.
        bg_type (str, optional):
            Type of background fill for the plot area. Options: 'gradient', 'solid', 'none'.
            Defaults to "gradient".
        bg_col (str | List[str], optional):
            Background color. If `bg_type='solid'`, provide a single color string.
            If `bg_type='gradient'`, provide a list of 3 color strings defining
            the gradient from bottom to top.
            Defaults to ['bisque', '#FFB90F', '#CD6600'] (R default gradient colors).
        use_annot_outline (bool, optional):
            If True, add a thin white outline to clone annotations specified in
            `fish_data` to improve contrast against clone colors. Requires the
            `matplotlib.patheffects` module. Defaults to False.
    Returns:
        tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
            The Matplotlib Figure and Axes objects containing the plot. If an `ax`
            was passed in, the returned objects are the same as (or derived from)
            the input.
    Raises:
        RuntimeError: If `fish_data.layout_clones()` has not been called before
            passing the object to this function.
    """
    if not fish_data.has_layout:
        raise RuntimeError("Input fish_data object must have layout calculated. Call fish_data.layout_clones() first.")

    # --- Setup Figure and Axes ---
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))  # Default figure size
    else:
        fig = ax.get_figure()

    # --- Calculate padding and set limits ---
    time_min = np.min(fish_data.timepoints)
    time_max = np.max(fish_data.timepoints)
    time_range = time_max - time_min if time_max > time_min else 1.0
    pad_left = time_range * pad_left_frac

    xlim = (time_min - pad_left, time_max)
    ylim = (0, 100)  # Fishplots typically use 0-100 for y-axis
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # --- Apply Background ---
    _draw_background(ax, bg_type, bg_col)

    # --- Turn off standard axes decorations ---
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # --- Draw Clones (Parent -> Child order) ---
    # Use a queue similar to layout, ensuring parents drawn before children
    parents_queue = [0]  # Start with virtual root parent
    processed_clones = set()  # Keep track of clones already drawn

    while parents_queue:
        parent_1based = parents_queue.pop(0)

        # Find children of this parent
        children_mask = (fish_data.parents == parent_1based)
        children_indices = np.where(children_mask)[0]  # 0-based indices

        # Draw these children
        for clone_idx in children_indices:
            if clone_idx in processed_clones:
                continue  # Should not happen in tree but safety check
            processed_clones.add(clone_idx)

            # Get data for this clone
            plot_data = fish_data.get_plot_data(clone_idx)
            xpos = plot_data['x']
            ytop = plot_data['ytop']
            ybtm = plot_data['ybtm']
            color = plot_data['color']
            nest_level = plot_data['nest_level']
            # Annotation data bundle
            annot_data = {
                'annot': plot_data['annot'],
                'angle': fish_data.clone_annots_angle,
                'col': fish_data.clone_annots_col,
                'pos': fish_data.clone_annots_pos,
                'cex': fish_data.clone_annots_cex,
                'offset': fish_data.clone_annots_offset,
            }

            # Adjust padding based on parent (R logic: less padding for subclones)
            current_pad_left = pad_left
            if parent_1based > 0:
                current_pad_left = pad_left * 0.4  # R's reduction factor

            draw_args = {
                "ax": ax, "xpos": xpos, "ytop": ytop, "ybtm": ybtm,
                "color": color, "nest_level": nest_level, "pad_left": current_pad_left,
                "border": border, "col_border": col_border, "annot_data": annot_data,
                "use_outline": use_annot_outline, "clone_idx": clone_idx, "fish_data": fish_data
            }

            # Call the appropriate drawing function
            if shape == "polygon":
                draw_args["ramp_angle"] = ramp_angle  # Polygon specific
                _draw_clust_polygon(**draw_args)
            elif shape == "spline":
                _draw_clust_spline(**draw_args)
            elif shape == "bezier":
                _draw_clust_bezier(**draw_args)
            else:
                warnings.warn(f"Unknown shape '{shape}'. Using 'polygon'.")
                draw_args["ramp_angle"] = ramp_angle
                _draw_clust_polygon(**draw_args)

            # Add this clone's index (as 1-based ID) to queue to process its children
            parents_queue.append(clone_idx + 1)

    # --- Add Vertical Lines and Labels ---
    if vlines is not None:
        vlines = np.asarray(vlines)
        # Add vertical lines that span the main plotting area (0-100)
        ax.vlines(vlines, ymin=0, ymax=100, color=col_vline, linewidth=1.0,
                  zorder=1)  # zorder 1 places above background but below clones

        if vlab is not None:
            if len(vlab) != len(vlines):
                warnings.warn("Length of vlab must match length of vlines. Skipping labels.")
            else:
                # Add labels above the plot area
                y_label_pos = 103  # Position above the 100% mark (adjust as needed)
                fontsize = plt.rcParams['font.size'] * cex_vlab
                for x_coord, label in zip(vlines, vlab):
                    ax.text(x_coord, y_label_pos, label,
                            ha='center', va='bottom',  # Center label horizontally
                            fontsize=fontsize,
                            color='grey')  # Match R's grey20 approx

    # --- Add Titles ---
    title_fontsize = plt.rcParams['font.size'] * cex_title
    if title:
        # R positions title centrally above the plot. Use fig.suptitle or ax.set_title.
        # ax.set_title might overlap vlab. fig.suptitle is often better.
        fig.suptitle(title, fontsize=title_fontsize, y=0.95)  # Adjust y as needed

    if title_btm:
        # R places this at bottom left, slightly outside the main x-range
        x_btm_pos = xlim[0] - pad_left * 0.1  # Position slightly left of padded area
        y_btm_pos = 2  # R uses y=2
        ax.text(x_btm_pos, y_btm_pos, title_btm,
                ha='left', va='bottom',  # Left align text
                fontsize=title_fontsize)

    # Adjust layout slightly to prevent titles/labels overlapping axes
    # fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Example adjustment

    return fig, ax


def draw_legend(fish_data: FishPlotData,
                ax: Optional[plt.Axes] = None,
                fig: Optional[plt.Figure] = None,
                ncol: Optional[int] = None,
                loc: str = 'lower center',
                bbox_to_anchor: Optional[Tuple[float, float]] = (0.5, -0.2),
                cex: float = 1.0,
                frameon: bool = False,
                title: Optional[str] = None,
                title_fontsize: Optional[Union[str, float]] = None,  # Added (requires mpl 3.7+)
                labelspacing: float = 0.5,
                columnspacing: float = 1.0,
                handletextpad: float = 0.8,
                handlelength: float = 2.0):
    """
    Draws a legend for the fishplot, typically placed below the main plot.

    Uses the clone labels and colors stored within the FishPlotData object.

    Args:
        fish_data (FishPlotData):
            The FishPlotData object containing clone labels (`clone_labels`)
            and colors (`colors`).
        ax (matplotlib.axes.Axes, optional):
            The Axes object associated with the fishplot. While the legend is
            usually added to the figure, providing the `ax` can help if
            positioning relies on axes coordinates (less common for figure legends).
            Defaults to None.
        fig (matplotlib.figure.Figure, optional):
            The Figure object to add the legend to. If None, tries to get the
            figure from `ax`. If `ax` is also None, defaults to the current
            figure (`plt.gcf()`). It's generally recommended to pass the
            explicit figure object. Defaults to None.
        ncol (int, optional):
            Number of columns in the legend. If None, a reasonable number of
            columns is estimated based on the number of clones.
            Defaults to None.
        loc (str, optional):
            The location of the legend's anchor point within its bounding box.
            Common values: 'upper left', 'upper center', 'lower center', 'center left', etc.
            See matplotlib documentation for `legend()` locations.
            Used in conjunction with `bbox_to_anchor`. Defaults to 'lower center'.
        bbox_to_anchor (Tuple[float, float], optional):
            Specifies the coordinates (typically in figure fraction, 0-1) where
            the legend's `loc` anchor point should be placed. Allows placing the
            legend precisely, often outside the main axes area.
            Defaults to (0.5, -0.15), placing the legend's bottom-center slightly
            below the figure's horizontal center (suitable if figure has enough bottom margin).
        cex (float, optional):
            Font size multiplier relative to Matplotlib's default font size for the
            legend item labels. Defaults to 1.0.
        frameon (bool, optional):
            Whether to draw a frame (border) around the legend box.
            Defaults to False.
        title (str, optional):
            An optional title for the legend box. Defaults to None.
        title_fontsize (str | float, optional):
             Font size for the legend title. Can be numeric (points) or descriptive
             string ('small', 'large', etc.). Requires Matplotlib >= 3.7.
             Defaults to None (uses default legend title size).
        labelspacing (float, optional):
             Vertical space between legend entries, in font-size units.
             Defaults to 0.5.
        columnspacing (float, optional):
             Horizontal space between legend columns, in font-size units.
             Defaults to 1.0.
        handletextpad (float, optional):
             Horizontal space between the legend handle (color patch) and the label text,
             in font-size units. Defaults to 0.8.
        handlelength (float, optional):
             Length of the legend handles (color patches), in font-size units.
             Defaults to 2.0.


    Returns:
        matplotlib.legend.Legend | None:
            The created Legend object, or None if no clones were found.

    Raises:
        AttributeError: If Matplotlib version is < 3.7 and `title_fontsize` is set.
    """
    if fig is None:
        if ax is not None:
            fig = ax.get_figure()
        else:
            fig = plt.gcf()  # Get current figure if no context provided

    n_clones = fish_data.n_clones
    if n_clones == 0:
        warnings.warn("No clones found in fish_data. Skipping legend.")
        return None

    # R default ncol logic: ceiling(n_clones / 8) rows -> ncol = 8 approx
    if ncol is None:
        if n_clones <= 6:
            ncol = n_clones
        elif n_clones <= 12:
            ncol = (n_clones + 1) // 2
        elif n_clones <= 24:  # Allow more cols for more clones
            ncol = (n_clones + 2) // 3
        else:
            ncol = (n_clones + 3) // 4
        # Cap reasonably
        ncol = min(ncol, 8)

    # Create legend handles (colored rectangles) and labels
    handles = [patches.Patch(color=color, label=label)
               for color, label in zip(fish_data.colors, fish_data.clone_labels)]

    fontsize = plt.rcParams['font.size'] * cex * 0.8  # Match R's 0.8 factor

    # Prepare legend kwargs, handle title_fontsize conditionally for compatibility
    legend_kwargs = {
        "handles": handles,
        "ncol": ncol,
        "loc": loc,
        "bbox_to_anchor": bbox_to_anchor,
        "frameon": frameon,
        "fontsize": fontsize,
        "title": title,
        "labelspacing": labelspacing,
        "columnspacing": columnspacing,
        "handletextpad": handletextpad,
        "handlelength": handlelength
    }
    # Check Matplotlib version before adding title_fontsize
    import matplotlib
    from packaging import version
    if version.parse(matplotlib.__version__) >= version.parse("3.7"):
        if title_fontsize is not None:
            legend_kwargs["title_fontsize"] = title_fontsize
    elif title_fontsize is not None:
        warnings.warn("legend 'title_fontsize' requires Matplotlib 3.7 or newer. Ignoring.", UserWarning)

    # Add legend to the figure
    legend = fig.legend(**legend_kwargs)

    return legend
