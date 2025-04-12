import pytest
import numpy as np
import pandas as pd
import warnings

# Import the class we are testing
from fishplotpy.data import FishPlotData

# --- Basic Valid Data ---
# Use a simple, known-good dataset for baseline checks
VALID_TIMEPOINTS = [0.0, 50.0, 100.0]
VALID_FRACS = pd.DataFrame([
    [100.0, 60.0, 95.0],  # P0, Clone 1
    [  0.0, 40.0, 80.0],  # P1, Clone 2
    [  0.0,  0.0, 10.0]   # P1, Clone 3
], columns=VALID_TIMEPOINTS)
VALID_PARENTS = [0, 1, 1]

# --- Test Fixture for Valid Data (Optional but good practice) ---
@pytest.fixture
def valid_input_data():
    """Provides a copy of basic valid input data."""
    return {
        "frac_table": VALID_FRACS.copy(),
        "parents": VALID_PARENTS[:],
        "timepoints": VALID_TIMEPOINTS[:]
    }

# --- Initialization Tests ---

def test_init_valid_data(valid_input_data):
    """Test successful initialization with minimal valid data."""
    fp_data = FishPlotData(**valid_input_data)
    assert fp_data.n_clones == 3
    assert fp_data.n_timepoints == 3
    assert np.array_equal(fp_data.parents, np.array(VALID_PARENTS))
    assert np.array_equal(fp_data.timepoints, np.array(VALID_TIMEPOINTS))
    pd.testing.assert_frame_equal(fp_data.frac_table, VALID_FRACS)
    # Check defaults
    assert fp_data.clone_labels == ['1', '2', '3']
    assert fp_data.clone_annots == ['', '', '']
    assert len(fp_data.colors) == 3  # Should have assigned default colors


def test_init_mismatched_parents_length():
    """Test error if len(parents) != num_clones."""
    with pytest.raises(ValueError, match="Length of parents must equal"):
        FishPlotData(frac_table=VALID_FRACS, parents=[0, 1])


def test_init_mismatched_timepoints_length():
    """Test error if len(timepoints) != num_timepoints."""
    with pytest.raises(ValueError, match="Length of timepoints must equal"):
        FishPlotData(frac_table=VALID_FRACS, parents=VALID_PARENTS, timepoints=[0, 50])


def test_init_invalid_parent_index_negative():
    """Test error if parents index is negative."""
    with pytest.raises(ValueError, match="Parent indices cannot be negative"):
        FishPlotData(frac_table=VALID_FRACS, parents=[0, -1, 1])


def test_init_invalid_parent_index_too_large():
    """Test error if parents index > num_clones."""
    with pytest.raises(ValueError, match="greater than the number of clones"):
        FishPlotData(frac_table=VALID_FRACS, parents=[0, 1, 4])  # Parent 4 invalid


def test_init_non_numeric_fracs():
    """Test error if frac_table contains non-numeric data."""
    invalid_fracs = VALID_FRACS.copy()
    invalid_fracs.iloc[0, 0] = "not a number"
    with pytest.raises(ValueError, match="frac_table must contain numeric values"):
        FishPlotData(frac_table=invalid_fracs, parents=VALID_PARENTS)


# --- Validation Logic Tests (_validate_inputs) ---

def test_validation_clone_reappearance_error():
    """Test error when a clone reappears after being zero."""
    fracs = pd.DataFrame([
        [10, 0, 10],  # Clone 1 reappears
        [90, 100, 90]
    ])
    parents = [0, 0]
    with pytest.raises(ValueError, match="goes from present to absent"):
        FishPlotData(frac_table=fracs, parents=parents)


def test_validation_clone_reappearance_ok_with_fix():
    """Test that clone reappearance doesn't error if fix_missing_clones=True."""
    fracs = pd.DataFrame([
        [10, 0, 10],
        [90, 90, 90]
    ])
    parents = [0, 0]
    # Expect a warning about the fix, but no ValueError
    with pytest.warns(UserWarning, match="Setting fraction to 1.00e-02"):
        fp_data = FishPlotData(frac_table=fracs, parents=parents, fix_missing_clones=True)

    # Check if the zero was replaced
    assert fp_data.frac_table.iloc[0, 1] == pytest.approx(0.01)
    # Implicitly checks that no ValueError was raised during init


def test_validation_all_zero_clone_warning(valid_input_data):
    """Test warning for clones that are always zero."""
    fracs = valid_input_data["frac_table"].copy()
    fracs.iloc[2, :] = 0 # Make clone 3 all zero
    with pytest.warns(UserWarning, match="fraction zero at all timepoints"):
        FishPlotData(frac_table=fracs, parents=valid_input_data["parents"])


def test_validation_nest_level_sum_error():
    """Test error if clones at same nest level sum > 100."""
    # Clones 2 and 3 are level 1, parent is Clone 1 (level 0)
    fracs = pd.DataFrame([
        [100, 100, 100],  # P0, Clone 1
        [  0,  60,  50],  # P1, Clone 2
        [  0,  50,  50]   # P1, Clone 3
    ], columns=VALID_TIMEPOINTS)
    parents = [0, 1, 1]
    # At time 50, level 1 clones sum to 60+50=110
    with pytest.raises(ValueError, match="nest level 1 sum to 110.00"):
        FishPlotData(frac_table=fracs, parents=parents, timepoints=VALID_TIMEPOINTS)


def test_validation_sibling_sum_error():
    """Test error if sibling clones sum > parent fraction."""
    # Clones 2 and 3 are siblings (parent 1)
    fracs = pd.DataFrame([
        [100, 80, 100], # P0, Clone 1 (parent)
        [  0, 50,  50], # P1, Clone 2 (child)
        [  0, 40,  50]  # P1, Clone 3 (child)
    ], columns=VALID_TIMEPOINTS)
    parents = [0, 1, 1]
    # At time 50, children sum to 50+40=90, parent is 80
    with pytest.raises(ValueError, match="Children clones .* of parent 1 sum to 90.00"):
        FishPlotData(frac_table=fracs, parents=parents, timepoints=VALID_TIMEPOINTS)

# --- Nest Level Calculation Tests ---

def test_nest_level_simple():
    """Test nest level calculation for a simple hierarchy."""
    parents = [0, 1, 1, 2, 0]
    fracs = pd.DataFrame(np.zeros((5, 2))) # Fractions don't matter for this test
    fp_data = FishPlotData(frac_table=fracs, parents=parents)
    expected_levels = [0, 1, 1, 2, 0]
    assert np.array_equal(fp_data.nest_level, expected_levels)

def test_nest_level_deeper():
    """Test nest level calculation for a deeper hierarchy."""
    parents = [0, 1, 2, 3, 1, 5]
    fracs = pd.DataFrame(np.zeros((6, 2)))
    fp_data = FishPlotData(frac_table=fracs, parents=parents)
    expected_levels = [0, 1, 2, 3, 1, 2]
    assert np.array_equal(fp_data.nest_level, expected_levels)

def test_nest_level_circular_dependency():
    """Test error on circular parent definitions."""
    parents = [0, 1, 3, 2] # 3->2, 2->3
    fracs = pd.DataFrame(np.zeros((4, 2)))
    with pytest.raises(ValueError, match="Circular dependency detected"):
        FishPlotData(frac_table=fracs, parents=parents)

# --- Tests for _fix_disappearing_clones ---

def test_fix_clones_basic_case():
    """Test basic replacement of intermediate zero."""
    fracs = pd.DataFrame([[10, 0, 10]])
    parents = [0] # nest_level 0 -> max(1, 0) = 1 -> small value = 0.01**1 = 0.01
    fp_data = FishPlotData(frac_table=fracs, parents=parents, fix_missing_clones=True)
    assert fp_data.frac_table.iloc[0, 1] == pytest.approx(0.01)


def test_fix_clones_multiple_zeros():
    """Test replacement of multiple consecutive intermediate zeros."""
    fracs = pd.DataFrame([[10, 0, 0, 0, 10]])
    parents = [0] # small value = 0.01
    fp_data = FishPlotData(frac_table=fracs, parents=parents, fix_missing_clones=True)
    assert fp_data.frac_table.iloc[0, 1] == pytest.approx(0.01)
    assert fp_data.frac_table.iloc[0, 2] == pytest.approx(0.01)
    assert fp_data.frac_table.iloc[0, 3] == pytest.approx(0.01)


def test_fix_clones_different_nest_levels():
    """Test small value depends on nest level."""
    fracs = pd.DataFrame([
        [50, 0, 50],  # Clone 1, P0 -> Level 0 -> fix = 0.01**1 = 0.01
        [30, 0, 30],   # Clone 2, P1 -> Level 1 -> fix = 0.01**1 = 0.01 (nest level calculated dynamically)
        [10, 0, 10]   # Clone 3, P2 -> Level 2 -> fix = 0.01**2 = 0.0001
    ])
    parents = [0, 1, 2]
    with pytest.warns(UserWarning):  # Expect warnings due to fixes
        fp_data = FishPlotData(frac_table=fracs, parents=parents, fix_missing_clones=True)

    # Assert fixes based on nest levels
    assert fp_data.frac_table.iloc[0, 1] == pytest.approx(0.01 ** 1)  # Lev 0 fix
    assert fp_data.frac_table.iloc[1, 1] == pytest.approx(0.01 ** 1)  # Lev 1 fix
    assert fp_data.frac_table.iloc[2, 1] == pytest.approx(0.01 ** 2)  # Lev 2 fix


def test_fix_clones_no_reappearance():
    """Test it doesn't change trailing or leading zeros."""
    fracs = pd.DataFrame([[0, 0, 10, 5, 0, 0]])
    parents = [0]
    fp_data = FishPlotData(frac_table=fracs, parents=parents, fix_missing_clones=True)
    assert fp_data.frac_table.iloc[0, 0] == 0
    assert fp_data.frac_table.iloc[0, 1] == 0
    assert fp_data.frac_table.iloc[0, 4] == 0 # Original zero was > 0 before fix
    assert fp_data.frac_table.iloc[0, 5] == 0
    # Check original values are kept
    assert fp_data.frac_table.iloc[0, 2] == 10
    assert fp_data.frac_table.iloc[0, 3] == 5


# --- Tests for _get_inner_space and _get_outer_space ---
@pytest.fixture
def fp_data_for_space_tests():
    """Fixture providing a valid FishPlotData instance."""
    # Use corrected VALID_FRACS data
    return FishPlotData(frac_table=VALID_FRACS.copy(), parents=VALID_PARENTS, timepoints=VALID_TIMEPOINTS)


def test_get_outer_space(fp_data_for_space_tests):
    """Test calculation of space outside root clones."""
    # Only Clone 1 is root (parent=0).
    # Outer space = 100 - fraction(Clone 1)
    expected_outer = 100.0 - np.array([100.0, 60.0, 95.0]) # -> [0, 40, 5]
    # Calculation happens within layout_clones, but we can call the private method directly for testing
    # or check the attribute after layout_clones runs. Let's call layout_clones.
    fp_data_for_space_tests.layout_clones()
    np.testing.assert_allclose(fp_data_for_space_tests.outer_space, expected_outer, atol=1e-9)


def test_get_inner_space(fp_data_for_space_tests):
    """Test calculation of space inside clones not occupied by children."""
    # Clone 1 is parent of 2, 3. Inner = Frac(1) - (Frac(2) + Frac(3))
    # Clone 2 has no children. Inner = Frac(2)
    # Clone 3 has no children. Inner = Frac(3)
    expected_inner_c1 = np.array([100, 60, 95]) - (np.array([0, 40, 80]) + np.array([0, 0, 10])) # -> [100, 20, 5]
    expected_inner_c2 = np.array([0, 40, 80])
    expected_inner_c3 = np.array([0, 0, 10])
    expected_inner_df = pd.DataFrame([expected_inner_c1, expected_inner_c2, expected_inner_c3],
                                     columns=VALID_TIMEPOINTS, dtype=float) # Ensure float comparison

    fp_data_for_space_tests.layout_clones() # Calculate inner space during layout

    # Ensure column types match for comparison
    inner_space_calculated = fp_data_for_space_tests.inner_space.astype(float)

    pd.testing.assert_frame_equal(inner_space_calculated, expected_inner_df, check_dtype=False, atol=1e-9)

def test_get_inner_space_no_children():
    """Test inner space when a clone has no children."""
    fracs = pd.DataFrame([[100.0], [0.0]])  # Use floats
    parents = [0, 0]  # Both root
    # If timepoints aren't specified, default is [0.0]
    fp_data = FishPlotData(fracs, parents)
    fp_data.layout_clones()

    # Create expected df with matching float column
    expected_inner = pd.DataFrame([[100.0], [0.0]], columns=[0.0], dtype=float)

    inner_space_calculated = fp_data.inner_space.astype(float)

    # Now compare, check_column_type might not even be needed now
    pd.testing.assert_frame_equal(inner_space_calculated, expected_inner)
