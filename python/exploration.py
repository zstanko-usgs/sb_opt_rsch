import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

opj = os.path.join

cwd = os.getcwd()

fig_dir = opj(cwd, 'figures')
if not os.path.exists(fig_dir):
    os.mkdir(fig_dir)

plt.style.use('ggplot')
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.size'] = 8
mpl.rcParams['lines.linewidth'] = 1
mpl.rcParams['lines.color'] = 'gray'
mpl.rcParams['axes.facecolor'] = 'gainsboro'
mpl.rcParams['legend.facecolor'] = 'whitesmoke'
mpl.rcParams['patch.linewidth'] = 0.3
mpl.rcParams['xtick.labelsize'] = 6
mpl.rcParams['ytick.labelsize'] = 6
mpl.rcParams['lines.markersize'] = 5

data_dir = opj(cwd, 'sb_opt_rsch', 'source_data')
file_name = 'scenario1_result.csv'
s1_result = pd.read_csv(opj(data_dir, file_name))
s1_data = s1_result.iloc[:, :440]
s1_pca = PCA().fit(s1_data.values)
#
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(np.cumsum(s1_pca.explained_variance_ratio_))
ax.set_xlim(-10, 440)
ax.set_ylim(0, 1.05)
ax.set_xlabel('number of components')
ax.set_ylabel('cumulative explained variance')
ax.minorticks_on()
ax.grid(which='minor', axis='both', lw=0.5, ls='--', color='whitesmoke')
plt.title('Scenario 1 Decision Variable Space')
plt.savefig(opj(fig_dir, 'pca_explained_variance_s1.png'))
#
plt.close()

# repeat for scenario 2
data_dir = opj(cwd, 'SB_Archive', 'ancillary', 'optimization', 'output', 'output_scenario_2')
s2_result = pd.read_csv(opj(data_dir, 'result.csv'))
s2_data = s2_result.iloc[:, :440]
s2_pca = PCA().fit(s2_data.values)
#
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(np.cumsum(s2_pca.explained_variance_ratio_))
ax.set_xlim(-10, 440)
ax.set_ylim(0, 1.05)
ax.set_xlabel('number of components')
ax.set_ylabel('cumulative explained variance')
ax.minorticks_on()
ax.grid(which='minor', axis='both', lw=0.5, ls='--', color='whitesmoke')
plt.title('Scenario 2 Decision Variable Space')
plt.savefig(opj(fig_dir, 'pca_explained_variance_s2.png'))
#
plt.close()


