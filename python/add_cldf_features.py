import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import zipfile as zf

# because laziness
opj = os.path.join
cwd = os.getcwd()

scen = 1
ns = 912  # number of schedules

Clwelnam = ['22A002', '22A004', '22B009', '22B010', '22B011', '22G003', '22G004',
            '23E003', '23E005', '23F002', '23F004', '23F005', '23F007', '23F008']
ends = []
diffs = []
for n in range(ns):
    with zf.ZipFile(opj(cwd, 'results', 's{}_schd{}.zip'.format(scen, n))) as z:
        z.extract('cldf.txt')
    tdf = pd.read_csv('cldf.txt', header=None, skipfooter=15, engine='python',
                      names=['c_end', 'c_strt'])
    os.remove('cldf.txt')
    tdf.index = Clwelnam
    tdf['c_diff'] = tdf.c_end - tdf.c_strt
    ends.append(tdf.c_end.to_dict())
    diffs.append(tdf.c_diff.to_dict())

ends_df = pd.DataFrame(ends)
ends_df.columns = [i + '_end' for i in Clwelnam]
diffs_df = pd.DataFrame(diffs)
diffs_df.columns = [i + '_diff' for i in Clwelnam]

ends_df.to_csv(opj(cwd, 'scripts', '_tmp', 'cl_obs_end.csv'))
diffs_df.to_csv(opj(cwd, 'scripts', '_tmp', 'cl_obs_diff.csv'))
