import pandas as pd
import matplotlib.pyplot as plt
from fishplotpy import FishPlotData, fishplot, draw_legend

timepoints = [0.0, 2.0, 3.0, 4.0]
frac_table = pd.DataFrame([
    [15,  40,  50, 100],
    [1, 15,  20,  60],
    [  1,  5,  20, 20],
    [  0,  0,  3, 20]
], columns=timepoints)
parents = [0, 1, 1, 3]  # first root (0), second and third inside the first one, fourth is inside the third one.

custom_labels = ["Founder (WT)", "Gain(Chr3)", "Mut(GeneA)", "DoubleHit+Resist"]
custom_colors = ['#FFFFFF', '#4682B4', '#FF4500', '#2E8B57']

custom_annots_base = ["First", "Second", "Third", "Fourth"]

fish_data = FishPlotData(
    frac_table=frac_table,
    parents=parents,
    timepoints=timepoints,
    clone_labels=custom_labels,
    colors=custom_colors,
    clone_annots=custom_annots_base,
    clone_annots_angle=45,
    clone_annots_pos=5,
    clone_annots_offset=0.6,
    clone_annots_col='#111111',
    clone_annots_cex=0.9
)

fish_data.layout_clones()

fig, ax = plt.subplots(figsize=(9, 6.5))
fig.subplots_adjust(left=0.05, right=0.95, bottom=0.22, top=0.88)

fishplot(
    fish_data,
    ax=ax,
    shape="spline",
    pad_left_frac=0.1,
    vlines=timepoints,
    vlab=[f"T={int(t)}" for t in timepoints],
    cex_vlab=0.75,
    col_vline="#cccccc",
    border=0.6,
    col_border="#666666",
    bg_type='solid',
    bg_col='#f0f0f0',
    use_annot_outline=True
)

ax.set_title("Muller Graphs Testing", fontsize=14, weight='bold', pad=25)

ax.text(ax.get_xlim()[0] - fish_data.timepoints[-1]*0.02, 50, 'Clonal Fraction (%)',
        ha='right', va='center', fontsize=9, rotation=90, color='#444444')

draw_legend(
    fish_data,
    fig=fig,
    ncol=4,
    frameon=True,
    title="Clone Identity",
    loc='upper center',
    bbox_to_anchor=(0.5, 0.15),
    cex=0.9
)

plt.show()
