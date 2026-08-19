import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# INPUT: INDEPENDENT TOP 10 SHAP FEATURES
# ============================================================

# Each row represents:
# 0 = Server
# 1 = Client 1
# 2 = Client 2
# 3 = Client 3
# 4 = Client 4
# 5 = Client 5

features = np.array([
    # Server
    [
        "Number",
        "ack_flag_number",
        "IAT",
        "Min",
        "Std",
        "TCP",
        "Tot sum",
        "Header_Length",
        "Max",
        "Variance",
    ],

    # Client 1
    [
        "Number",
        "ack_flag_number",
        "IAT",
        "Header_Length",
        "Protocol Type",
        "Max",
        "Std",
        "Min",
        "TCP",
        "UDP",
    ],

    # Client 2
    [
        "Number",
        "IAT",
        "Header_Length",
        "ack_flag_number",
        "Std",
        "Max",
        "Tot size",
        "Min",
        "TCP",
        "Protocol Type",
    ],

    # Client 3
    [
        "Tot sum",
        "TCP",
        "ack_flag_number",
        "Header_Length",
        "Min",
        "Number",
        "Protocol Type",
        "Tot size",
        "AVG",
        "IAT",
    ],

    # Client 4
    [
        "Number",
        "ack_flag_number",
        "IAT",
        "TCP",
        "Tot sum",
        "Std",
        "Variance",
        "Max",
        "Min",
        "Header_Length",
    ],

    # Client 5
    [
        "Number",
        "Std",
        "Variance",
        "Min",
        "ack_flag_number",
        "IAT",
        "Max",
        "Header_Length",
        "TCP",
        "Tot sum",
    ],
], dtype=object)


# Normalized mean absolute SHAP importance (%)
scores = np.array([
    # Server
    [
        13.5982,
        7.9741,
        6.7399,
        6.5076,
        5.9842,
        5.9447,
        5.8377,
        5.1265,
        4.8096,
        4.6912,
    ],

    # Client 1
    [
        15.1720,
        7.6514,
        7.4913,
        6.9403,
        5.8443,
        5.3435,
        5.3329,
        5.0661,
        4.7554,
        3.7331,
    ],

    # Client 2
    [
        19.2028,
        10.4584,
        6.2360,
        6.1757,
        5.1785,
        4.8208,
        4.2147,
        4.1989,
        4.1707,
        3.9729,
    ],

    # Client 3
    [
        10.5658,
        10.0067,
        8.8150,
        7.2229,
        6.7626,
        6.5686,
        6.0450,
        5.9540,
        4.6234,
        4.4769,
    ],

    # Client 4
    [
        13.9698,
        7.6849,
        6.9117,
        6.3794,
        6.1796,
        5.6564,
        5.1651,
        5.0451,
        4.7717,
        4.5901,
    ],

    # Client 5
    [
        15.6319,
        8.0825,
        7.2541,
        6.8827,
        6.7210,
        6.2807,
        5.5540,
        4.9911,
        4.6927,
        3.6814,
    ],
], dtype=float)


panel_labels = [
    "(a) Server",
    "(b) Client 1",
    "(c) Client 2",
    "(d) Client 3",
    "(e) Client 4",
    "(f) Client 5",
]


# ============================================================
# INPUT VALIDATION
# ============================================================

if features.ndim != 2 or scores.ndim != 2:
    raise ValueError(
        "features and scores must both be two-dimensional arrays."
    )

if features.shape != scores.shape:
    raise ValueError(
        "features and scores must have identical dimensions."
    )

if features.shape != (6, 10):
    raise ValueError(
        "Expected arrays with shape (6, 10): "
        "one server and five clients."
    )

if np.any(scores < 0):
    raise ValueError(
        "SHAP importance scores cannot be negative."
    )


# ============================================================
# FIGURE SETTINGS
# ============================================================

fig, axes = plt.subplots(
    nrows=1,
    ncols=6,
    figsize=(32, 5.8),
    sharex=True,
    sharey=False,
)

server_colour = "#2878B5"
client_colour = "#F28E2B"

# Use one common X-axis range for all panels
x_max = np.ceil((scores.max() * 1.12) / 5) * 5
x_ticks = np.arange(0, x_max + 0.1, 5)

y_positions = np.arange(10)


# ============================================================
# DRAW PANELS
# ============================================================

for model_index, axis in enumerate(axes):

    # Reverse values so the highest-ranked feature appears at the top
    model_features = features[model_index][::-1]
    model_scores = scores[model_index][::-1]

    if model_index == 0:
        bar_colour = server_colour
    else:
        bar_colour = client_colour

    bars = axis.barh(
        y_positions,
        model_scores,
        height=0.68,
        color=bar_colour,
        edgecolor="none",
    )

    # --------------------------------------------------------
    # Y-axis feature names
    # --------------------------------------------------------

    axis.set_yticks(y_positions)

    axis.set_yticklabels(
        model_features,
        fontsize=7.5,
    )

    axis.tick_params(
        axis="y",
        labelleft=True,
        length=3,
        pad=5,
    )

    # --------------------------------------------------------
    # X-axis
    # --------------------------------------------------------

    axis.set_xlim(0, x_max)
    axis.set_xticks(x_ticks)

    axis.tick_params(
        axis="x",
        labelsize=8,
    )

    # --------------------------------------------------------
    # SHAP value labels
    # --------------------------------------------------------

    for bar, value in zip(bars, model_scores):
        axis.text(
            value + 0.18,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            ha="left",
            va="center",
            fontsize=7,
        )

    # --------------------------------------------------------
    # Panel identification below each plot
    # --------------------------------------------------------

    axis.set_xlabel(
        panel_labels[model_index],
        fontsize=9,
        labelpad=11,
    )

    # No subplot titles
    axis.set_title("")

    # --------------------------------------------------------
    # Retain only the X-axis and Y-axis
    # --------------------------------------------------------

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    axis.spines["left"].set_linewidth(0.8)
    axis.spines["bottom"].set_linewidth(0.8)

    # Remove gridlines
    axis.grid(False)


# ============================================================
# COMMON AXIS LABELS
# ============================================================

axes[0].set_ylabel(
    "SHAP features",
    fontsize=10,
    labelpad=12,
)

fig.text(
    0.5,
    0.025,
    "Normalized mean absolute SHAP importance (%)",
    ha="center",
    va="center",
    fontsize=10,
)


# ============================================================
# LAYOUT
# ============================================================

# The increased left margin prevents the server feature names
# and Y-axis label from being cut.
plt.subplots_adjust(
    left=0.11,
    right=0.99,
    top=0.96,
    bottom=0.23,
    wspace=0.88,
)


# ============================================================
# SAVE FIGURE
# ============================================================

plt.savefig(
    "independent_top10_shap_horizontal.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.35,
)

plt.savefig(
    "independent_top10_shap_horizontal.pdf",
    bbox_inches="tight",
    pad_inches=0.35,
)

plt.show()