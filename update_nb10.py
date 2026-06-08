import nbformat
import json

nb_path = '10-11-selection-centre-clipped.ipynb'
nb = nbformat.read(nb_path, as_version=4)

plot_code = """
# Plotting quantising error vs bits for different rise1 ratios
print("Plotting quantising error vs bits for different rise1 ratios...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
fig.suptitle('Quantising Error vs Bits/Pixel for varying rise1_ratio')

steps = np.arange(5, 60, 5)

for idx, scheme in enumerate(schemes):
    Y = scheme.encode(lighthouse)
    ax = axes[idx]
    
    for ratio in ratios:
        errs = []
        bits_list = []
        for step in steps:
            Yq = scheme.quant(Y, step, rise1_ratio=ratio)
            Z = scheme.decode(Yq)
            errs.append(np.std(lighthouse - Z))
            bits_list.append(scheme.get_bits(Yq))
        
        ax.plot(bits_list, errs, marker='o', label=f'rise1_ratio={ratio}')
    
    ax.set_title(scheme.name)
    ax.set_xlabel('Bits/Pixel')
    if idx == 0:
        ax.set_ylabel('RMS Error')
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()
"""

new_cell = nbformat.v4.new_code_cell(source=plot_code.strip())

# We want to insert this after the current RISE1 RATIO INVESTIGATION cell (cell 6 in 0-indexed terms)
# But it's safer to just append it before the discussion markdown cell, or just at the end of the notebook.
# Let's insert it at index 7 (after cell 6). Wait, in scratch_update_nb10.py, the script was putting the discussion in cell 10.
# So let's insert the plot code right before the discussion for rise1.
idx_insert = None
for i, c in enumerate(nb.cells):
    if c.cell_type == 'markdown' and 'Is `rise1 = step` a reasonable compromise?' in c.source:
        idx_insert = i
        break

if idx_insert is not None:
    nb.cells.insert(idx_insert, new_cell)
else:
    nb.cells.append(new_cell)

nbformat.write(nb, nb_path)
print("Updated 10-11-selection-centre-clipped.ipynb successfully with plotting cell.")
