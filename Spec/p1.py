# High-resolution spectrograph wavelength coverage plot
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Data extracted from the provided table
spectrographs = [
	# Optical
	{"telescope": "VLT (8.2)", "spec": "ESPRESSO (f)", "R": "140k", "wl_min": 0.38, "wl_max": 0.79, "band": "Optical"},
	{"telescope": "VLT (8.2)", "spec": "UVES (s)", "R": "100k", "wl_min": 0.30, "wl_max": 1.10, "band": "Optical"},
	{"telescope": "Keck (10)", "spec": "HIRES (s)", "R": "70k", "wl_min": 0.30, "wl_max": 1.00, "band": "Optical"},
	{"telescope": "LBT (2 × 8.4)", "spec": "PEPSI (f)", "R": "130k", "wl_min": 0.38, "wl_max": 0.91, "band": "Optical"},
	{"telescope": "Gemini (8.1)", "spec": "MAROON-X (f)", "R": "80k", "wl_min": 0.50, "wl_max": 0.90, "band": "Optical"},
	{"telescope": "Subaru (8.2)", "spec": "HDS (s)", "R": "45k", "wl_min": 0.30, "wl_max": 1.00, "band": "Optical"},
	{"telescope": "DCT (4.3)", "spec": "EXPRES (f)", "R": "150k", "wl_min": 0.38, "wl_max": 0.68, "band": "Optical"},
	{"telescope": "ESO (3.6)", "spec": "HARPS (f)", "R": "120k", "wl_min": 0.38, "wl_max": 0.69, "band": "Optical"},
	{"telescope": "CFHT (3.5)", "spec": "ESPaDOnS (f)", "R": "81k", "wl_min": 0.37, "wl_max": 1.05, "band": "Optical"},
	{"telescope": "Calar A. (3.5)", "spec": "CARMENES (f)", "R": "95k", "wl_min": 0.52, "wl_max": 0.96, "band": "Optical"},
	{"telescope": "TNG (3.6)", "spec": "HARPS-N (f)", "R": "120k", "wl_min": 0.38, "wl_max": 0.68, "band": "Optical"},
	{"telescope": "ELT (39)", "spec": "ANDES (f)", "R": "100k", "wl_min": 0.40, "wl_max": 1.00, "band": "Optical"},
	{"telescope": "GMT (25)", "spec": "G-CLEF (f)", "R": "100k", "wl_min": 0.35, "wl_max": 0.90, "band": "Optical"},
	# Near-infrared
	{"telescope": "VLT (8.2)", "spec": "CRIRES+ (s)", "R": "100k", "wl_min": 0.95, "wl_max": 5.30, "band": "Near-IR"},
	{"telescope": "Keck (10)", "spec": "NIRSpec (s)", "R": "35k", "wl_min": 0.95, "wl_max": 5.50, "band": "Near-IR"},
	{"telescope": "Gemini (8.1)", "spec": "IGRINS (s)", "R": "45k", "wl_min": 1.45, "wl_max": 2.50, "band": "Near-IR"},
	{"telescope": "Calar A. (3.5)", "spec": "CARMENES (f)", "R": "80k", "wl_min": 0.96, "wl_max": 1.71, "band": "Near-IR"},
	{"telescope": "ESO (3.6)", "spec": "NIRPS (f)", "R": "82k", "wl_min": 0.95, "wl_max": 1.80, "band": "Near-IR"},
	{"telescope": "CFHT (3.5)", "spec": "Spirou (f)", "R": "75k", "wl_min": 0.95, "wl_max": 2.35, "band": "Near-IR"},
	{"telescope": "TNG (3.6)", "spec": "GIANO (f)", "R": "50k", "wl_min": 0.90, "wl_max": 2.50, "band": "Near-IR"},
	{"telescope": "ELT (39)", "spec": "ANDES (f)", "R": "100k", "wl_min": 1.00, "wl_max": 1.80, "band": "Near-IR"},
	{"telescope": "TMT (30)", "spec": "MODHIS (f)", "R": "100k", "wl_min": 0.95, "wl_max": 2.40, "band": "Near-IR"},
	{"telescope": "GMT (25)", "spec": "GMTNIRS (f)", "R": "50k", "wl_min": 1.07, "wl_max": 2.45, "band": "Near-IR"},
	# Mid-infrared
	{"telescope": "VLT (8.2)", "spec": "CRIRES+ (s)", "R": "100k", "wl_min": 0.95, "wl_max": 5.30, "band": "Mid-IR"},
	{"telescope": "Keck (10)", "spec": "NIRSpec (s)", "R": "25k", "wl_min": 0.95, "wl_max": 5.50, "band": "Mid-IR"},
	{"telescope": "ELT (39)", "spec": "METIS (s)", "R": "100k", "wl_min": 3.00, "wl_max": 5.00, "band": "Mid-IR"},
	{"telescope": "GMT (25)", "spec": "GMTNIRS (f)", "R": "100k", "wl_min": 2.90, "wl_max": 5.30, "band": "Mid-IR"},
]

# Assign colors for each band (colorblind-friendly, modern palette)
band_colors = {
	"Optical": "#4E79A7",   # blue
	"Near-IR": "#F28E2B",  # orange
	"Mid-IR": "#59A14F"    # green
}

# Sort for better grouping in the plot
spectrographs = sorted(spectrographs, key=lambda x: (x["band"], -float(x["R"].replace('k',''))))


# Prepare y-axis positions and labels
y_telescopes = []
y_pos = []
for i, spec in enumerate(spectrographs):
	y = len(spectrographs) - i
	y_pos.append(y)
	y_telescopes.append(spec['telescope'])

fig, ax = plt.subplots(figsize=(12, 10))

# Plot bars and annotate spectrograph name inside each bar

# Smart label placement: inside if bar is wide enough, else just outside

for i, spec in enumerate(spectrographs):
	y = y_pos[i]
	color = band_colors[spec["band"]]
	bar_width = spec["wl_max"] - spec["wl_min"]
	bar = ax.barh(y, bar_width, left=spec["wl_min"], height=0.7, color=color, edgecolor='black', alpha=0.8)
	label = f"{spec['spec']} R={spec['R']}"
	# Draw connector line from left end of bar to y-axis label
	ax.plot([0.25, spec["wl_min"]], [y, y], color='gray', lw=0.3, ls='-', zorder=1, alpha=0.3, solid_capstyle='round')
	# Estimate if label fits: assume ~0.11 μm per character (empirical, adjust as needed)
	char_width = 0.11
	label_length = len(label)
	needed_width = char_width * label_length
	if bar_width > needed_width:
		# Center inside bar
		x_center = (spec["wl_min"] + spec["wl_max"]) / 2
		ax.text(x_center, y, label, ha='center', va='center', color='white', fontsize=9, fontweight='bold', zorder=10, clip_on=True)
	else:
		# Place just outside right edge, smaller font, black text
		ax.text(spec["wl_max"] + 0.03, y, label, ha='left', va='center', color='black', fontsize=8, fontweight='bold', zorder=10, clip_on=False)

# Set left y-ticks and labels (telescope name)
ax.set_yticks(y_pos)
ax.set_yticklabels(y_telescopes, fontsize=11)


# Axis labels and limits
ax.set_xlabel("Wavelength (μm)", fontsize=14)
ax.set_xlim(0.27, 5.6)
ax.set_title("Wavelength Coverage of High-Resolution Spectrographs", fontsize=16, weight='bold')

# Add legend
patches = [mpatches.Patch(color=col, label=band) for band, col in band_colors.items()]
ax.legend(handles=patches, title="Band", loc="lower right")

# Grid and layout
ax.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()

ax.set_xscale('log')
ax.set_xticks(ticks=np.array([0.3, 0.5, 1., 2., 3., 4., 5.]),\
              labels=np.array(['0.3', '0.5', '1', '2', '3', '4', '5']))

# Show or save the plot
#plt.show()
plt.savefig("Spec/spectrograph_wavelength_coverage.png", dpi=300)