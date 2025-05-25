import os
import json
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import re

# Directory containing results
RESULTS_DIR = "results"

# Mapping of display labels
DISPLAY_ATTRS = {
    "compile": "compile",
    "quantization_torchao": "quant",
    "enable_fsdp_float8_all_gather": "f8_allg",
    "force_recompute_fp8_weight_in_bwd": "f8_bwd",
    "quantize_optimizer": "quant_opt",
    "fused_optimizer": "fused_opt",
    "world_size": "world_size",
}

# Regex to extract world_size from filename
world_size_pattern = re.compile(r"world_size_(\d+)")

json_files = sorted(glob(os.path.join(RESULTS_DIR, "*.json")))

results = []

for file_path in json_files:
    with open(file_path, "r") as f:
        data = json.load(f)

    # Extract world_size from filename
    match = world_size_pattern.search(file_path)
    world_size = int(match.group(1)) if match else "?"

    # Compose label with renamed attributes
    label_lines = []
    for attr, display in DISPLAY_ATTRS.items():
        value = data.get(attr) if attr != "world_size" else world_size
        label_lines.append(f"{display}: {value}")
    label = "\n".join(label_lines)

    memory_summaries = data.get("memory_summaries", [])
    if memory_summaries:
        median_memory = np.median(memory_summaries) / (1024 ** 3)  # bytes to GB
        results.append((label, median_memory))

# Sort by memory usage (descending)
results.sort(key=lambda x: x[1], reverse=True)

# Plotting
labels, medians = zip(*results)
plt.figure(figsize=(max(12, len(labels) * 1.2), 8))
plt.bar(range(len(medians)), medians, color="skyblue")
plt.ylabel("Median VRAM Usage (GB)")
plt.xticks(range(len(labels)), labels, rotation=0, ha="center", fontsize=9)
plt.title("VRAM Usage Across Experiments")
plt.tight_layout()
plt.grid(axis="y", linestyle="--", alpha=0.6)

plt.savefig("vram_usage_plot.png")
