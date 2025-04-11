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
    assert len(fp_data.colors) == 3 # Should have assigned default colors


def test_init_mismatched_parents_length():
    """Test error if len(parents) != num_clones."""
    with pytest.raises(ValueError, match="Length of parents must equal"):
        FishPlotData(frac_table=VALID_FRACS, parents=[0, 1]) # Too short


def test_init_mismatched_timepoints_length():
    """Test error if len(timepoints) != num_timepoints."""
    with pytest.raises(ValueError, match="Length of timepoints must equal"):
        FishPlotData(frac_table=VALID_FRACS, parents=VALID_PARENTS, timepoints=[0, 50]) # Too short


def test_init_invalid_parent_index_negative():
    """Test error if parents index is negative."""
    with pytest.raises(ValueError, match="Parent indices cannot be negative"):
        FishPlotData(frac_table=VALID_FRACS, parents=[0, -1, 1])


def test_init_invalid_parent_index_too_large():
    """Test error if parents index > num_clones."""
    with pytest.raises(ValueError, match="greater than the number of clones"):
        FishPlotData(frac_table=VALID_FRACS, parents=[0, 1, 4]) # Parent 4 invalid


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
        [10, 0, 10], # Clone 1 reappears
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

# --- TODO: Add tests for _fix_disappearing_clones logic ---
# --- TODO: Add tests for _get_inner_space / _get_outer_space ---
# --- TODO: Add tests for layout_clones coordinate calculations ---
