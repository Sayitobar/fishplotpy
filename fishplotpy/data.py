import pandas as pd
import numpy as np
import warnings
from typing import List, Optional, Union, Sequence, Dict, Tuple

# Default color palette
DEFAULT_COLORS = [
    "#888888", "#EF0000", "#FF6000", "#FFCF00", "#50FFAF",
    "#00DFFF", "#0070FF", "#0000FF", "#00008F"
][::-1]  # to match rev() in R


class FishPlotData:
    """
    Holds, validates, and preprocesses input data for creating a fishplot.

    This class is the primary data container for `fishplotpy`. It takes raw clonal
    fraction data, parental relationships, and timepoints, performs validation
    checks (e.g., ensuring child clone fractions do not exceed parent fractions),
    calculates necessary metadata like nesting levels, handles defaults, and stores
    the information required for layout calculation and plotting.

    The layout coordinates themselves are calculated by calling the `layout_clones`
    method on an instance of this class.

    Args:
        frac_table (pd.DataFrame | np.ndarray | List[List[float]]):
            A table where rows represent clones and columns represent timepoints.
            Values should be the fraction (typically 0-100, but can be 0-1 if
            consistent) of each clone at each timepoint. If not a DataFrame,
            it will be converted. Clones should be ordered consistently with
            the `parents` list.
        parents (Sequence[int]):
            A list or array specifying the parent of each clone. Must have the
            same length as the number of rows (clones) in `frac_table`.
            Use 0 for founder clones with no parent.
            For clones with a parent, use the 1-based index corresponding to the
            row number of the parent clone in the *original* `frac_table`.
            E.g., `[0, 1, 1]` means Clone 1 (row 0) is a founder, Clone 2 (row 1)
            has Clone 1 as parent, Clone 3 (row 2) also has Clone 1 as parent.
        timepoints (Sequence[Union[int, float]], optional):
            A list or array of numerical values corresponding to the timepoints
            (columns) in `frac_table`. Must have the same length as the number
            of columns. If None, timepoints will be assigned sequential integers
            0, 1, 2, ... Defaults to None.
        colors (List[str], optional):
            A list of color strings (e.g., hex codes '#RRGGBB', standard names
            like 'red', 'blue') for each clone. Length must match the number
            of clones. If None, a default color palette is used (can be recycled
            if more clones than colors). Defaults to None.
        clone_labels (List[str], optional):
            A list of names for each clone, used in the legend. Length must
            match the number of clones. If None, clones are labeled numerically
            '1', '2', ... Defaults to None.
        clone_annots (List[str], optional):
            A list of annotation strings (e.g., mutation names) to display near
            the origin of each corresponding clone shape. Use empty strings ""
            for clones without annotations. Length must match the number of clones.
            Defaults to None (no annotations displayed).
        clone_annots_angle (float, optional):
            Angle in degrees (0-360) for displaying clone annotations. Defaults to 0.
        clone_annots_col (str, optional):
            Color string for the clone annotation text. Defaults to 'black'.
        clone_annots_pos (int, optional):
            Position code for annotation relative to the clone origin point:
            1=below (matplotlib va='top'), 2=left (ha='right'),
            3=above (va='bottom'), 4=right (ha='left'). Affects text alignment.
            Defaults to 2 (left).
        clone_annots_cex (float, optional):
            Size factor (multiplier) for clone annotation text relative to the
            default matplotlib font size. Defaults to 0.7.
        clone_annots_offset (float, optional):
            Approximate offset distance for the annotation text from the origin point,
            scaled roughly relative to plot axes ranges. Larger values push text
            further away. May require tuning. Defaults to 0.2.
        fix_missing_clones (bool, optional):
            If True, attempts to fix cases where a clone's fraction is 0 at a
            timepoint *between* timepoints where it was present (> 0). Replaces
            such intermediate zeros with a very small value (0.01 ^ nest_level)
            to maintain visual continuity.
            Warning: Use with caution, as this modifies input data and may cause
            subsequent validation failures if fractions sum very close to 100.
            Defaults to False.

    Attributes:
        frac_table (pd.DataFrame): Validated (and potentially modified by
            `fix_missing_clones`) fraction table (clones x timepoints).
            Columns are numeric timepoints. Index is 0 to n_clones-1.
        parents (np.ndarray): Array of 1-based parent indices (0 for root).
        timepoints (np.ndarray): Numeric timepoint values corresponding to columns.
        colors (List[str]): Colors assigned to each clone.
        clone_labels (List[str]): Labels assigned to each clone.
        clone_annots (List[str]): Annotations assigned to each clone.
        clone_annots_angle (float): Angle for annotations.
        clone_annots_col (str): Color for annotations.
        clone_annots_pos (int): Position code (1-4) for annotations.
        clone_annots_cex (float): Size factor for annotations.
        clone_annots_offset (float): Offset value for annotations.
        n_clones (int): Number of clones (rows).
        n_timepoints (int): Number of timepoints (columns).
        nest_level (np.ndarray): Calculated nesting level (depth in hierarchy)
            for each clone (0=root).
        inner_space (pd.DataFrame | None): Calculated space fraction unique to each
            clone (not occupied by children) per timepoint. Populated by
            `layout_clones`. Initialized to None or empty DataFrame.
        outer_space (np.ndarray | None): Calculated space fraction outside all
            root clones per timepoint. Populated by `layout_clones`.
            Initialized to None or empty array.
        xpos (List[np.ndarray] | None): Calculated x-coordinates for plotting
            each clone shape's vertices. List length = n_clones. Populated
            by `layout_clones`.
        ytop (List[np.ndarray] | None): Calculated top y-coordinates for plotting
            each clone shape's vertices. Populated by `layout_clones`.
        ybtm (List[np.ndarray] | None): Calculated bottom y-coordinates for plotting
            each clone shape's vertices. Populated by `layout_clones`.
        xst_yst (List[Optional[Tuple[float, float]]]): Stores the calculated origin
             point `(xst, yst)` for each clone, used for annotation placement.
             Populated by plotting functions. Initialized with Nones.
        has_layout (bool): Flag indicating if `layout_clones` has been run
            successfully.
    """

    def __init__(self,
                 frac_table: Union[pd.DataFrame, np.ndarray, List[List[float]]],
                 parents: Sequence[int],
                 timepoints: Optional[Sequence[Union[int, float]]] = None,
                 colors: Optional[List[str]] = None,
                 clone_labels: Optional[List[str]] = None,
                 clone_annots: Optional[List[str]] = None,
                 clone_annots_angle: float = 0,
                 clone_annots_col: str = 'black',
                 clone_annots_pos: int = 2,
                 clone_annots_cex: float = 0.7,
                 clone_annots_offset: float = 0.2,
                 fix_missing_clones: bool = False):

        # --- Store annotation params ---
        self.clone_annots_angle = clone_annots_angle
        self.clone_annots_col = clone_annots_col
        self.clone_annots_pos = clone_annots_pos
        self.clone_annots_cex = clone_annots_cex
        self.clone_annots_offset = clone_annots_offset

        # --- Initialize Layout Flag EARLY ---
        self.has_layout = False  # Flag to check if layout is computed

        # --- Process frac_table ---
        if isinstance(frac_table, pd.DataFrame):
            self.frac_table = frac_table.copy()
        else:
            self.frac_table = pd.DataFrame(frac_table)
        # Ensure numeric data
        try:
            self.frac_table = self.frac_table.astype(float)
        except ValueError as e:
            raise ValueError(f"frac_table must contain numeric values. Error: {e}")

        self.n_clones = self.frac_table.shape[0]
        self.n_timepoints = self.frac_table.shape[1]
        # Use 0-based index internally for rows
        self.frac_table.index = range(self.n_clones)

        # --- Process parents ---
        if len(parents) != self.n_clones:
            raise ValueError("Length of parents must equal the number of rows in frac_table.")
        self.parents = np.array(parents, dtype=int)
        if np.any(self.parents < 0):
            raise ValueError("Parent indices cannot be negative.")
        max_parent = np.max(self.parents)
        if max_parent > self.n_clones:
            raise ValueError(f"Parent index {max_parent} is greater than the number of clones ({self.n_clones}). Parent indices must be 0 or between 1 and {self.n_clones}.")

        # --- Calculate Nesting Level ---
        self.nest_level = self._get_all_nest_levels()

        # --- Set Defaults ---
        # Store _fix_missing_clones for later check in validation
        self._fix_missing_clones_flag = fix_missing_clones
        self._set_defaults(timepoints, clone_labels, clone_annots)

        # --- Fix Missing Clones (Optional) ---
        if fix_missing_clones:
            self._fix_disappearing_clones()

        # --- Validate Inputs ---
        self._validate_inputs()

        # --- Set Colors ---
        self._set_colors(colors)

        # --- Initialize Layout Attributes ---
        self.inner_space: pd.DataFrame = pd.DataFrame(np.nan, index=self.frac_table.index, columns=self.frac_table.columns)
        self.outer_space: np.ndarray = np.full(self.n_timepoints, np.nan)
        self.xpos: List[np.ndarray] = [np.array([]) for _ in range(self.n_clones)]
        self.ytop: List[np.ndarray] = [np.array([]) for _ in range(self.n_clones)]
        self.ybtm: List[np.ndarray] = [np.array([]) for _ in range(self.n_clones)]
        self.xst_yst: List[Optional[Tuple[float, float]]] = [None] * self.n_clones  # storage for calculated origin points


    def _get_nest_level(self, clone_idx: int, _visited: Optional[set] = None) -> int:
        """Calculates the nesting level for a single clone (0-based index)."""
        if _visited is None:
            _visited = set()

        if clone_idx in _visited:
            raise ValueError(f"Circular dependency detected in parents involving clone {clone_idx + 1}")
        _visited.add(clone_idx)

        parent_one_based = self.parents[clone_idx]

        if parent_one_based == 0:
            return 0
        else:
            parent_zero_based = parent_one_based - 1
            # Basic check, more robust check in __init__
            if not (0 <= parent_zero_based < self.n_clones):
                raise ValueError(
                    f"Parent {parent_one_based} for clone {clone_idx + 1} is out of bounds [1, {self.n_clones}].")

            # Recursively find parent's level. Pass visited set copy.
            level = self._get_nest_level(parent_zero_based, _visited.copy()) + 1
            return level

    def _get_all_nest_levels(self) -> np.ndarray:
        """Calculates nesting levels for all clones."""
        levels = np.zeros(self.n_clones, dtype=int)
        for i in range(self.n_clones):
            try:
                levels[i] = self._get_nest_level(i)
            except ValueError as e:
                raise ValueError(f"Error calculating nest level for clone {i + 1}: {e}") from e
            except RecursionError:
                raise ValueError(f"Maximum recursion depth exceeded while calculating nest levels. Check for circular dependencies or excessively deep nesting in 'parents'. Clone index was {i + 1}.")
        return levels

    def _set_defaults(self, timepoints, clone_labels, clone_annots):
        """Sets default values for timepoints, labels, and annotations."""
        # Timepoints
        if timepoints is None:
            self.timepoints = np.arange(self.n_timepoints, dtype=float)
            self.frac_table.columns = self.timepoints
        else:
            if len(timepoints) != self.n_timepoints:
                raise ValueError("Length of timepoints must equal the number of columns in frac_table.")
            try:
                self.timepoints = np.array(timepoints, dtype=float)
            except ValueError:
                raise ValueError("timepoints must be a numeric vector.")
            self.frac_table.columns = self.timepoints  # Use timepoints as column names

        # Clone Labels
        if clone_labels is None:
            self.clone_labels = [str(i + 1) for i in range(self.n_clones)]
        else:
            if len(clone_labels) != self.n_clones:
                raise ValueError("Length of clone_labels must equal the number of rows in frac_table.")
            self.clone_labels = list(clone_labels)

        # Clone Annotations
        if clone_annots is None:
            self.clone_annots = [""] * self.n_clones
        else:
            if len(clone_annots) != self.n_clones:
                raise ValueError("Length of clone_annots must equal the number of rows in frac_table.")
            self.clone_annots = list(clone_annots)

    def _fix_disappearing_clones(self):
        """Replaces intermediate zeros with small values for continuity."""
        modified_fracs = self.frac_table.values.copy()  # numpy array more efficient

        for clone_idx in range(self.n_clones):
            row = modified_fracs[clone_idx, :]
            non_zero_indices = np.where(row > 0)[0]

            if len(non_zero_indices) > 1:
                min_nz_idx = non_zero_indices[0]
                max_nz_idx = non_zero_indices[-1]

                if max_nz_idx > min_nz_idx:
                    # Check for zeros between the first and last non-zero points
                    for i in range(min_nz_idx + 1, max_nz_idx):
                        if row[i] == 0:
                            # R uses 0.01^nest_level. Use max(1, level) to avoid 0.01^0 = 1
                            level = max(1, self.nest_level[clone_idx])
                            small_val = 0.01 ** level
                            modified_fracs[clone_idx, i] = small_val
                            warnings.warn(f"Clone {clone_idx + 1} has zero fraction at timepoint {i} "
                                          f"(value {self.timepoints[i]}) between non-zero values. "
                                          f"Setting fraction to {small_val:.2e} due to fix_missing_clones=True.",
                                          UserWarning)
        # Update the DataFrame
        self.frac_table = pd.DataFrame(modified_fracs, index=self.frac_table.index, columns=self.frac_table.columns)

    def _validate_inputs(self):
        """Performs sanity checks on the input data."""
        # Check clone reappearance (only if fix_missing_clones is False)
        if not getattr(self, '_fix_disappearing_clones_called', False):  # Check if fix was called
            for clone_idx in range(self.n_clones):
                row = self.frac_table.iloc[clone_idx].values
                started = False
                ended = False
                for time_idx in range(self.n_timepoints):
                    if row[time_idx] > 0:
                        if started and ended:
                            raise ValueError(
                                f"Clone {clone_idx + 1} goes from present to absent (fraction=0) and then back "
                                f"to present at timepoint {time_idx} (value {self.timepoints[time_idx]}). "
                                f"Values must be non-zero while clone is present. "
                                f"Either fix inputs or set fix_missing_clones=True."
                            )
                        started = True
                        ended = False  # Reset ended if it becomes present again
                    elif started:
                        ended = True

        # Check for all-zero clones
        row_sums = self.frac_table.sum(axis=1)
        zero_clones = row_sums == 0
        if zero_clones.any():
            zero_clone_indices = np.where(zero_clones)[0] + 1
            warnings.warn(f"Clones {list(zero_clone_indices)} have fraction zero at all timepoints "
                          "and will not be displayed.", UserWarning)

        # Check sums within nest levels and parent groups (use tolerance)
        tolerance = 1e-6
        for time_idx in range(self.n_timepoints):
            current_time_fracs = self.frac_table.iloc[:, time_idx].values

            calculated_nest_levels = self.nest_level  # Get the calculated levels
            # print(f"DEBUG: Time {time_idx}, Nest Levels: {calculated_nest_levels}"
            for level in np.unique(calculated_nest_levels):
                level_mask = calculated_nest_levels == level
                level_clones_indices = np.where(level_mask)[0]  # 0-based indices
                level_sum = current_time_fracs[level_mask].sum()
                # print(f"DEBUG: Time {time_idx}, Level {level}, Clones(0-based): {level_clones_indices}, Sum: {level_sum}")
                if level_sum > 100.0 + tolerance:
                    level_clones_one_based = level_clones_indices + 1
                    # print(f"ERROR Triggered: Clones {level_clones_one_based}, Level {level}, Sum {level_sum}")
                    raise ValueError(
                        f"Clones {list(level_clones_one_based)} with nest level {level} sum to "
                        f"{level_sum:.2f} (> 100) at timepoint {time_idx} (value {self.timepoints[time_idx]})."
                    )

            # Check sibling sums (should not exceed parent fraction)
            unique_parents = np.unique(self.parents)
            for parent_one_based in unique_parents:
                if parent_one_based > 0:  # Skip root clones
                    parent_zero_based = parent_one_based - 1
                    parent_fraction = current_time_fracs[parent_zero_based]

                    children_mask = self.parents == parent_one_based
                    children_sum = current_time_fracs[children_mask].sum()

                    if children_sum > parent_fraction + tolerance:
                        children_clones = np.where(children_mask)[0] + 1
                        raise ValueError(
                            f"Children clones {list(children_clones)} of parent {parent_one_based} sum to "
                            f"{children_sum:.2f}, which is greater than the parent's fraction "
                            f"({parent_fraction:.2f}) at timepoint {time_idx} (value {self.timepoints[time_idx]})."
                        )

    def _set_colors(self, colors: Optional[List[str]]):
        """Sets and validates the color list for clones."""
        if colors is None:
            n_defaults = len(DEFAULT_COLORS)
            if self.n_clones > n_defaults:
                warnings.warn(
                    f"Number of clones ({self.n_clones}) exceeds the number of default "
                    f"colors ({n_defaults}). Colors will be recycled. Provide a 'colors' list "
                    "for unique colors.", UserWarning
                )
                # Recycle colors
                self.colors = [DEFAULT_COLORS[i % n_defaults] for i in range(self.n_clones)]
            else:
                # Use first n_clones colors from default
                self.colors = DEFAULT_COLORS[:self.n_clones]
            print(f"Using default color scheme for {self.n_clones} clones.")  # Mimic R message location
        else:
            if len(colors) != self.n_clones:
                raise ValueError(
                    f"Number of colors provided ({len(colors)}) must equal the number of clones ({self.n_clones}).")
            # Basic validation: ensure it's a list of strings? Add more checks later if needed.
            if not all(isinstance(c, str) for c in colors):
                raise ValueError("colors must be a list of strings.")
            self.colors = list(colors)


    def _get_inner_space(self) -> pd.DataFrame:
        """
        Calculates the fraction of each clone NOT occupied by its direct children.
        Equivalent to R's getInnerSpace, but calculates for all clones at once.

        Returns:
            pd.DataFrame: DataFrame (clones x timepoints) of inner space fractions.
        """
        inner_space_df = self.frac_table.copy()
        for parent_1based in range(1, self.n_clones + 1):
            children_mask = (self.parents == parent_1based)
            if np.any(children_mask):
                parent_0based = parent_1based - 1
                # Subtract sum of children fractions from parent fraction
                children_sum = self.frac_table.loc[children_mask, :].sum(axis=0)
                inner_space_df.iloc[parent_0based, :] -= children_sum

        # Ensure no negative values due to floating point errors / slight validation misses
        inner_space_df[inner_space_df < 0] = 0
        return inner_space_df

    def _get_outer_space(self) -> np.ndarray:
        """
        Calculates the fraction NOT occupied by any top-level clones (parent=0).
        Equivalent to R's getOuterSpace.

        Returns:
            np.ndarray: Array of outer space fractions for each timepoint.
        """
        root_clones_mask = (self.parents == 0)
        if not np.any(root_clones_mask):
            warnings.warn("No root clones (parent=0) found. Outer space will be 100% at all timepoints.", UserWarning)
            return np.full(self.n_timepoints, 100.0)

        root_clones_sum = self.frac_table.loc[root_clones_mask, :].sum(axis=0)
        outer_space_array = 100.0 - root_clones_sum
        # Ensure no negative values
        outer_space_array[outer_space_array < 0] = 0
        return outer_space_array.values

    def _num_active_children(self, parent_1based: int, time_idx: int) -> int:
        """
        Counts the number of direct children of a parent clone that have
        a non-zero fraction at a specific timepoint index.
        Equivalent to R's numChildren.

        Args:
            parent_1based (int): The 1-based index of the parent clone.
            time_idx (int): The 0-based index of the timepoint column.

        Returns:
            int: The number of active children.
        """
        if parent_1based <= 0:  # Root or invalid parent index
            return 0
        children_mask = (self.parents == parent_1based)
        if not np.any(children_mask):
            return 0

        # Get fractions of children at the specific timepoint
        children_fracs = self.frac_table.iloc[children_mask, time_idx]
        num_active = np.sum(children_fracs > 0)  # Count non-zero fractions
        return int(num_active)

    # --- Methods to be added later ---
    def layout_clones(self, separate_independent_clones: bool = False):
        """
        Calculates the vertical layout coordinates (ybtm, ytop) for each
        clone at each timepoint, based on clonal fractions and parentage.
        Populates self.xpos, self.ybtm, self.ytop attributes.
        Equivalent to R's layoutClones function.

        Args:
            separate_independent_clones (bool, optional): If True, add equal
                vertical spacing between unrelated clones (those with parent=0).
                If False, stack all clones starting from y=0. Defaults to False.
        """
        # --- Pre-calculate inner and outer space ---
        self.inner_space = self._get_inner_space()
        self.outer_space = self._get_outer_space()

        # --- Initialize temporary coordinate storage ---
        # Use NaN to indicate points where clone is absent
        ybtm_matrix = np.full((self.n_clones, self.n_timepoints), np.nan)
        ytop_matrix = np.full((self.n_clones, self.n_timepoints), np.nan)
        xpos_matrix = np.full((self.n_clones, self.n_timepoints), np.nan)

        # Store timepoints in xpos matrix where clone potentially exists
        timepoints_expanded = np.tile(self.timepoints, (self.n_clones, 1))
        xpos_matrix = timepoints_expanded  # Start with all timepoints

        # --- Build layout timepoint by timepoint ---
        for time_idx in range(self.n_timepoints):
            timepoint_val = self.timepoints[time_idx]

            # Use a queue for breadth-first traversal (parent -> children)
            # Start with the virtual root parent 0
            parents_queue = [0]
            processed_parents = set()

            while parents_queue:
                parent_1based = parents_queue.pop(0)

                # Avoid reprocessing if a node is reached via different paths (shouldn't happen with tree)
                if parent_1based in processed_parents:
                    continue
                processed_parents.add(parent_1based)

                children_mask = (self.parents == parent_1based)
                children_indices = np.where(children_mask)[0]

                # Add children to the queue for processing later
                # Convert 0-based index to 1-based ID for queue
                parents_queue.extend([idx + 1 for idx in children_indices])

                # Calculate spacing and starting y
                num_active_children_here = self._num_active_children(parent_1based, time_idx)
                current_y = 0.0
                spacing = 0.0

                if parent_1based == 0:  # Handling root clones
                    # Start at bottom, considering outer space
                    root_outer_space = self.outer_space[time_idx]
                    if separate_independent_clones:
                        num_roots = np.sum(self.parents == 0)
                        if num_roots > 0 and root_outer_space > 1e-9:  # Add tolerance
                            # Divide space evenly between and around root clones
                            spacing = root_outer_space / (num_roots + 1)
                        current_y = 0.0  # Start strictly at the bottom
                    else:
                        # Start half the outer space up from the bottom
                        current_y = root_outer_space / 2.0
                        spacing = 0  # No extra space between clones

                else:  # Handling subclones (parent_1based > 0)
                    parent_0based = parent_1based - 1
                    # Start at the bottom coordinate of the parent
                    # If parent ybtm[parent_0based, time_idx] is NaN, this will propagate NaN.
                    current_y = ybtm_matrix[parent_0based, time_idx]

                    # Check if parent is actually present (y is not NaN)
                    if not np.isnan(current_y):
                        parent_inner_space = self.inner_space.iloc[parent_0based, time_idx]
                        if num_active_children_here > 0 and parent_inner_space > 1e-9:
                            # Divide parent's inner space among children
                            spacing = parent_inner_space / (num_active_children_here + 1)
                        else:
                            spacing = 0  # No space if no active children or no inner space
                    else:
                        # if parent is absent at this timepoint, children cannot be plotted either
                        # their fractions should also be 0 due to validation, but enforce here
                        spacing = 0
                        # Set children y values to NaN if parent is NaN
                        for child_idx in children_indices:
                            ybtm_matrix[child_idx, time_idx] = np.nan
                            ytop_matrix[child_idx, time_idx] = np.nan
                        continue  # Skip processing children if parent absent

                # Layout children vertically
                for clone_idx in children_indices:  # clone_idx is 0-based
                    clone_fraction = self.frac_table.iloc[clone_idx, time_idx]

                    if clone_fraction > 0:
                        # Clone is present
                        current_y += spacing  # Add space below the clone
                        y_bottom = current_y
                        current_y += clone_fraction  # Add clone height
                        y_top = current_y

                        ybtm_matrix[clone_idx, time_idx] = y_bottom
                        ytop_matrix[clone_idx, time_idx] = y_top
                        # xpos already set to timepoint_val
                    else:
                        # Clone is absent (fraction is 0)
                        ybtm_matrix[clone_idx, time_idx] = np.nan
                        ytop_matrix[clone_idx, time_idx] = np.nan
                        xpos_matrix[clone_idx, time_idx] = np.nan  # mark x as NaN too

        # --- Finalize Coordinates ---
        final_xpos = []
        final_ytop = []
        final_ybtm = []

        for i in range(self.n_clones):
            valid_mask = ~np.isnan(ybtm_matrix[i, :]) & ~np.isnan(ytop_matrix[i, :]) & ~np.isnan(xpos_matrix[i, :])

            final_xpos.append(xpos_matrix[i, valid_mask])
            final_ytop.append(ytop_matrix[i, valid_mask])
            final_ybtm.append(ybtm_matrix[i, valid_mask])

        self.xpos = final_xpos
        self.ytop = final_ytop
        self.ybtm = final_ybtm
        self.has_layout = True  # Mark layout as computed

        # Return self for potential chaining, though not typical in Python init/layout
        return self

    # --- Method added in previous step ---
    def get_plot_data(self, clone_idx: int) -> Dict[str, Union[np.ndarray, str, float, int]]:
        """
        Helper to get calculated layout and style data for a specific clone.
        Requires layout_clones() to have been run.

        Args:
            clone_idx (int): The 0-based index of the clone.

        Returns:
            dict: A dictionary containing 'x', 'ytop', 'ybtm' coordinate arrays,
                  'color', 'nest_level', and 'annot' string for the clone.

        Raises:
            RuntimeError: If layout_clones() has not been run.
            IndexError: If clone_idx is out of range.
        """
        if not self.has_layout:
            raise RuntimeError("Layout must be calculated first using layout_clones().")
        if not (0 <= clone_idx < self.n_clones):
            raise IndexError(f"clone_idx {clone_idx} out of range for {self.n_clones} clones.")

        return {
            "x": self.xpos[clone_idx],
            "ytop": self.ytop[clone_idx],
            "ybtm": self.ybtm[clone_idx],
            "color": self.colors[clone_idx],
            "nest_level": int(self.nest_level[clone_idx]),  # Ensure integer
            "annot": self.clone_annots[clone_idx]
        }

    def __repr__(self):
        status = "Layout NOT computed"
        if self.has_layout:
            status = "Layout computed"
        return (f"<FishPlotData object with {self.n_clones} clones, "
                f"{self.n_timepoints} timepoints. {status}>")
