import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
"""
python 3

"""

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


data_dir = opj(cwd, 'SB_Archive', 'ancillary', 'optimization', 'output', 'output_scenario_1')

s1_result = pd.read_csv(opj(data_dir, 'result.csv'))

welnam = ['Alameda',
          'CityHal',
          'CorpYrd',
          'HopeAve',
          'Lwood',
          'Lrobles',
          'OrtegaP',
          'SanRoq2',
          'SBHighS',
          'ValVer',
          'VerCruz']

# TODO: try replacing with np.arange()
well_dv = {
'Alameda Park': list(range(0, 440, 11)),
'City Hall': list(range(1, 440, 11)),
'Corporation Yard': list(range(2, 440, 11)),
'Hope Avenue': list(range(3, 440, 11)),
'Lincolnwood': list(range(4, 440, 11)),
'Los Robles': list(range(5, 440, 11)),
'Ortega Park': list(range(6, 440, 11)),
'San Roque Park': list(range(7, 440, 11)),
'Santa Barbara HS': list(range(8, 440, 11)),
'Val Verde': list(range(9, 440, 11)),
'Vera Cruz': list(range(10, 440, 11))
}

plt.ioff()
for c in s1_result.columns[:440]:
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    s1_result[c].hist(ax=ax, bins=10)
    ax.set_xlabel('Pumping Rate ' + r'[$\frac{m^3}{day}$]')
    ax.set_ylabel('count')
    ax.set_title('Decision Variable {}'.format(c))
    plt.savefig(opj(fig_dir, 'hist_{}.png'.format(c)))
    plt.close()

for c in s1_result.columns[440:444]:
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    s1_result[c].hist(ax=ax, bins=10)
    ax.set_xlabel('Value')
    ax.set_ylabel('count')
    ax.set_title('Objective {}'.format(c))
    plt.savefig(opj(fig_dir, 'hist_{}.png'.format(c)))
    plt.close()

# this messes up the order of the columns. maybe because a dictionary is used?
# for key, value in well_dv.items():
#     s1_result.iloc[:, value].hist(figsize=(6, 12), bins=10, sharex=True, sharey=True, layout=(10, 4))
#     plt.suptitle('{}'.format(key))
#     plt.savefig(opj(fig_dir, 'hist_{}.png'.format(key)))
#     plt.close()

# no this is out of order too :-/
# for i, w in enumerate(welnam):
#     s1_result.iloc[:, i:440:11].hist(figsize=(6, 12), bins=10, sharex=True, sharey=True, layout=(10, 4))
#     plt.suptitle('{}'.format(w))
#     plt.savefig(opj(fig_dir, 'hist_{}.png'.format(w)))
#     plt.close()

# this does it. i suppose explicit is better
colnam = ['Quarter {}'.format(qtr) for qtr in range(1, 5)]
rownam = ['Year {}'.format(yr) for yr in range(1, 11)]

for i, w in enumerate(welnam):
    fig, axs = plt.subplots(nrows=10, ncols=4, sharex=True, sharey=True, figsize=(7, 12))
    for j, ax in enumerate(axs.flatten()):
        ax.hist(s1_result.iloc[:, i + j * 11])
        # ax.set_title('{}'.format(s1_result.columns[i + j * 11]), fontsize=8)
    for ax, col in zip(axs[0], colnam):
        ax.set_title(col, fontsize=12)
    for ax, row in zip(axs[:, 0], rownam):
        ax.set_ylabel(row, rotation=0, labelpad=25, fontsize=12)
        ax.tick_params(labelleft=False)
    plt.suptitle('{}'.format(w), y=1.01, fontsize=14)
    plt.tight_layout()
    plt.savefig(opj(fig_dir, 'hist_{}.png'.format(w)), dpi=300, bbox_inches='tight')
    plt.close()

plt.show()

