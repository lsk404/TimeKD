import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FormatStrFormatter
## == ETTh1 ==
# ili_ave_data = { 
#     "Agent_Participation_Ratio": [0.2,0.4,0.6,0.8,1.0],
#     "MSE": [0.2994,0.2970,0.2974,0.2973,0.2997],
#     "MAE": [0.3543, 0.3512, 0.3510, 0.3517,0.3526]
# }
# xlabel = 'Agent Participation Ratio on ETTh1'
# MSE_range = (0.29,0.31)
# MAE_range = (0.34,0.36)
## == ETTh2 ==
ili_ave_data = {
    "Agent_Participation_Ratio": [0.2,0.4,0.6,0.8,1.0],
    "MSE": [0.1710,0.1715,0.1725,0.1725,0.1725],
    "MAE": [0.2622, 0.2624, 0.2630, 0.2626,0.2627]
}
MSE_range = (0.16,0.18)
MAE_range = (0.255,0.275)
xlabel = 'Agent Participation Ratio on ETTh2'

n_ticks = 5
# Creating DataFrame
df_ili_ave = pd.DataFrame(ili_ave_data)

plt.rcParams.update({
    'font.size': 20,          # Set font size
    'axes.titlesize': 20,     # Title font size
    'axes.labelsize': 20,     # Label font size
    'xtick.labelsize': 20,    # X-axis tick label size
    'ytick.labelsize': 20,    # Y-axis tick label size
    'legend.fontsize': 20,    # Legend font size
    'font.family': 'Times New Roman',  # Set font family
})


fig, ax1 = plt.subplots(figsize=(8, 6))  # Increase figure size for better visibility

# Define positions
x = range(len(df_ili_ave['Agent_Participation_Ratio']))

# Plot MSE
ax1.plot(x, df_ili_ave['MSE'], color='#2D69A8', label='MSE', marker='o', markersize=8, linewidth=2, zorder=3)
ax1.set_ylabel('MSE', color='#2D69A8', fontsize=18)
ax1.tick_params(axis='y', labelcolor='#2D69A8', labelsize=20)
ax1.set_ylim(MSE_range)  # Adjust MSE y-axis range for better visualization
plt.xlabel(xlabel, fontsize=20, labelpad=15)  # 添加横轴标题

# Rotate x-axis labels
plt.xticks(x, df_ili_ave['Agent_Participation_Ratio'], ha='center', fontsize=20)

# Create a second y-axis for MAE
ax2 = ax1.twinx()
ax2.plot(x, df_ili_ave['MAE'], color='#CF2E2B', label='MAE', marker='s', markersize=8, linewidth=2, zorder=3)
ax2.set_ylabel('MAE', color='#CF2E2B', fontsize=20)
ax2.tick_params(axis='y', labelcolor='#CF2E2B', labelsize=20)
ax2.set_ylim(MAE_range)  # Adjust MAE y-axis range to match the visualization better
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

# ax1.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True,min_n_ticks=5))  # Limit MSE ticks to 5
# ax2.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True,min_n_ticks=5))  # Limit MAE ticks to 5
ax1.set_yticks(np.linspace(*ax1.get_ylim(), n_ticks))
ax2.set_yticks(np.linspace(*ax2.get_ylim(), n_ticks))
# Add grid for MSE axis
ax1.grid(True, which='major', linestyle='--', linewidth=0.7, zorder=0, alpha=0.6)

# Add legend
fig.legend(
    ['MSE', 'MAE'],
    loc='upper center',
    bbox_to_anchor=(0.5, 0.87),  # 控制图例的位置，(x, y) 表示相对画布的位置
    fontsize=18,
    ncol=2
)

# Save to PDF
ili_ave_non_overlapping_pdf_path = f"Results/plot_res/{xlabel}.pdf"
with PdfPages(ili_ave_non_overlapping_pdf_path) as pdf:
    pdf.savefig(fig, bbox_inches='tight')  # Use bbox_inches='tight' to fit all elements
plt.tight_layout()
plt.show()