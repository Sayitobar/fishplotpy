import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# Import necessary components
from fishplotpy import FishPlotData, fishplot, draw_legend

# 1. Define Input Data & Styling
timepoints = [0.0, 30.0, 75.0, 150.0]  # Use floats
# Use the corrected fraction table
frac_table = pd.DataFrame([
    [100,  2,  2, 98],  # Clone 1 (P0)
    [ 45,  0,  0,  0],  # Clone 2 (P1)
    [  0,  0,  2, 95],  # Clone 3 (P2)
    [  0,  0,  1, 40]   # Clone 4 (P3)
], columns=timepoints)
parents = [0, 1, 1, 3]  # 0-based parent indices would be [-1, 0, 0, 2] - USE 1-based!

# --- Custom Styling ---
custom_labels = ["Founder (WT)", "Gain(Chr3)", "Mut(GeneA)", "DoubleHit+Resist"]
custom_colors = ['#808080', '#4682B4', '#FF4500', '#2E8B57']

# Provide annotations, but leave overlapping ones empty for now
custom_annots_base = ["Start", "+Chr3", "", ""]
# Define the text for the ones we'll place manually
annot_clone3 = "GeneA\nmissense"
annot_clone4 = "Drug\nResistant"

# 2. Create FishPlotData object with more options
fish_data = FishPlotData(
    frac_table=frac_table,
    parents=parents,
    timepoints=timepoints,
    clone_labels=custom_labels,
    colors=custom_colors,
    clone_annots=custom_annots_base,
    # Style annotations for clarity
    clone_annots_angle=45,
    clone_annots_pos=5,          # Place to the right of origin
    clone_annots_offset=0.6,     # Push slightly further out
    clone_annots_col='#111111',  # Dark grey text
    clone_annots_cex=0.9         # Annotation text size factor
)

# 3. Calculate Layout Coordinates
fish_data.layout_clones()

# 4. Generate the Plot with more styling
fig, ax = plt.subplots(figsize=(9, 6.5))
# Reserve space at bottom (20%) and top (10%) for legend/title
fig.subplots_adjust(left=0.05, right=0.95, bottom=0.22, top=0.88)

# Generate fishplot with more arguments
fishplot(
    fish_data,
    ax=ax,
    shape="spline",               # Smooth shapes
    pad_left_frac=0.1,            # Less padding on left
    # Vertical lines for all timepoints
    vlines=timepoints,
    vlab=[f"T={int(t)}" for t in timepoints],  # Simple time labels
    cex_vlab=0.75,                 # Vlab text size
    col_vline="#cccccc",          # Lighter grey vlines
    border=0.6,                   # Clone border thickness
    col_border="#666666",         # Mid-grey clone border
    # Background
    bg_type='solid',              # Clean solid background
    bg_col='#f0f0f0',             # Light grey background
    # Annotations
    use_annot_outline=True        # Add white outline to annotations
)

# Set the main title using ax.set_title positioned by subplots_adjust
ax.set_title("Example Clonal Evolution Dynamics", fontsize=14, weight='bold', pad=25)

# Add a subtle y-axis label substitute for context
ax.text(ax.get_xlim()[0] - fish_data.timepoints[-1]*0.02, 50, 'Clonal Fraction (%)',
        ha='right', va='center', fontsize=9, rotation=90, color='#444444')


# 5. Manual Annotation Placement for Clones 3 & 4
base_fontsize = plt.rcParams['font.size']
annot_effects = [pe.withStroke(linewidth=1.5, foreground='white')]  # outline effect

# Get stored origins (make sure modification in step 1 was done)
origin_c3 = fish_data.xst_yst[2]  # Clone 3 is at index 2
origin_c4 = fish_data.xst_yst[3]  # Clone 4 is at index 3

ax.text(
    origin_c4[0], origin_c4[1] + 4, annot_clone3,
    ha='left', va='center', rotation=45,
    color='#111111', fontsize=base_fontsize * 0.9,
    path_effects=annot_effects,
    transform=ax.transData
)

ax.text(
    origin_c4[0], origin_c4[1] - 4, annot_clone4,  # Add small negative Y shift
    ha='left', va='center', rotation=45,  # Keep pos=4 alignment/angle
    color='#111111', fontsize=base_fontsize * 0.9,
    path_effects=annot_effects,
    transform=ax.transData
)


# 6. Add a styled and well-positioned Legend
draw_legend(
    fish_data,
    fig=fig,                     # Pass figure for figure coords
    ncol=2,                      # Use 2 columns for better fit
    frameon=True,                # Draw frame around legend
    title="Clone Identity",      # Add title to legend
    loc='upper center',          # Anchor legend at its upper center edge
    bbox_to_anchor=(0.5, 0.15),  # X=0.5 (fig center), Y=0.15 (low in figure, above very bottom)
    cex=0.9                      # legend text size factor
)

# 6. Display or Save
plt.show()

# To save in higher resolution (e.g., 300 DPI):
# Use bbox_inches='tight' to minimize extra whitespace and ensure legend is included
# fig.savefig("fishplot_showcase.png", dpi=300, bbox_inches='tight')
