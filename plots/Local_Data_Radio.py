import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FormatStrFormatter



# Scalability ETTm1-24
ili_ave_data = {
    "Local_Data_Radio": [0.2,0.4,0.6,0.8,1.0],
    "MSE": [0.4078,0.3147,0.3033,0.3003, 0.3008],
    "MAE": [0.4240, 0.3649, 0.3560, 0.3535,0.3537]
}

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
x = range(len(df_ili_ave['Local_Data_Radio']))

# Plot MSE
ax1.plot(x, df_ili_ave['MSE'], color='#2D69A8', label='MSE', marker='o', markersize=8, linewidth=2, zorder=3)
ax1.set_ylabel('MSE', color='#2D69A8', fontsize=18)
ax1.tick_params(axis='y', labelcolor='#2D69A8', labelsize=20)
ax1.set_ylim(0.29, 0.41)  # Adjust MSE y-axis range for better visualization
plt.xlabel('Local Data Radio', fontsize=20, labelpad=15)  # 添加横轴标题

# Rotate x-axis labels
plt.xticks(x, df_ili_ave['Local_Data_Radio'], ha='center', fontsize=20)

# Create a second y-axis for MAE
ax2 = ax1.twinx()
ax2.plot(x, df_ili_ave['MAE'], color='#CF2E2B', label='MAE', marker='s', markersize=8, linewidth=2, zorder=3)
ax2.set_ylabel('MAE', color='#CF2E2B', fontsize=20)
ax2.tick_params(axis='y', labelcolor='#CF2E2B', labelsize=20)
ax2.set_ylim(0.34, 0.43)  # Adjust MAE y-axis range to match the visualization better
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

ax1.yaxis.set_major_locator(MaxNLocator(nbins=5))  # Limit MSE ticks to 5
ax2.yaxis.set_major_locator(MaxNLocator(nbins=5))  # Limit MAE ticks to 5
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
ili_ave_non_overlapping_pdf_path = "Results/plot_res/Local_Data_Radio_ETTm1.pdf"
with PdfPages(ili_ave_non_overlapping_pdf_path) as pdf:
    pdf.savefig(fig, bbox_inches='tight')  # Use bbox_inches='tight' to fit all elements
plt.tight_layout()
plt.show()