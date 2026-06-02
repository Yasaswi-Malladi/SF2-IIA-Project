import json
nb = json.load(open('c:/SF2 IIA Project/SF2-IIA-Project/10-11-selection-centre-clipped.ipynb', encoding='utf-8'))
for i, c in enumerate(nb['cells']):
    source = "".join(c['source']).replace('\n', ' ')
    print(f"Cell {i} ({c['cell_type']}): {source[:100]}")
