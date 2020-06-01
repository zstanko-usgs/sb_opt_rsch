import sys
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import flopy as fp
import flopy.utils.binaryfile as bf
from flopy.utils.mflistfile import SwtListBudget

opj = os.path.join
scen = 32
sched = '2D'
model_nam = "SBModel"

model_dir = os.path.abspath('E:/SB_research/SB_Archive/ancillary/optimization/modelfiles')
in_dir = opj(model_dir, 'input')
out_dir = opj(model_dir, 'output')
out_pth = opj(out_dir, 'scenario_{}'.format(scen), 'Schedule_{}'.format(sched))
namefile = opj(model_dir, model_nam+'_s{}.nam'.format(scen))

exe_name = opj(model_dir, 'swt_v4x64')

fig_dir = opj(model_dir, '_fig')
if not os.path.exists(fig_dir):
    os.mkdir(fig_dir)
tmp_dir = opj(model_dir, '_tmp')
if not os.path.exists(tmp_dir):
    os.mkdir(tmp_dir)

swt = fp.seawat.Seawat(model_nam, exe_name=exe_name, model_ws=model_dir)
dis = fp.modflow.ModflowDis.load(opj(in_dir, model_nam+'.dis'), swt)

# create head object
head_obj = bf.HeadFile(opj(out_pth, 'SBModel.bhd'))
times = head_obj.get_times()
kskp = head_obj.get_kstpkper()

cl_obj = bf.UcnFile(opj(out_pth, 'SBModel_Cl.ucn'), text='CONCENTRATION', model=swt)
# tstp = cl_obj.get_kstpkper()
# times = cl_obj.get_times()

cbc_obj = bf.CellBudgetFile(opj(out_pth, 'SBModel.cbc'))
ts = cbc_obj.get_times()
sp = cbc_obj.get_kstpkper()
cbc_obj.get_unique_record_names()
nrec = cbc_obj.get_nrecords()

recharge = cbc_obj.get_data(kstpkper=sp[-1], text='RECHARGE', full3D=True)[0]
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1, aspect='equal')
ax.imshow(recharge[0, :, :], interpolation='nearest')
plt.savefig(os.path.join(fig_dir, 's{}_{}_recharge.png'.format(scen, sched)))
plt.show()

wells = cbc_obj.get_data(kstpkper=sp[-1], text='WELLS', full3D=True)[0]
fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(2, 1, 1, aspect='equal')
ax2 = fig.add_subplot(2, 1, 2, aspect='equal')
w1 = ax1.imshow(wells[1, :, :])
w2 = ax2.imshow(wells[1, :, :])
fig.colorbar(w1, ax=ax1)
fig.colorbar(w2, ax=ax2)
plt.savefig(os.path.join(fig_dir, 's{}_{}_wells.png'.format(scen, sched)))
plt.show()

# mountain front recharge will be the values in the WELL data that are positive
# negative values would be extraction wells
# assuming mtn front recharge is only in layer 0 and 1
mtn0 = wells[0, :, :] > 0
mtn_wells_lay0 = wells[0][mtn0]
mtn1 = wells[1, :, :] > 0
mtn_wells_lay1 = wells[1][mtn1]

# compute the total inflow from mountain front recharge in the last stress period
mtn_wells_sum = mtn_wells_lay0.sum() + mtn_wells_lay1.sum()

# compute the total recharge inflow from precipitation in the last stress period
recharge_sum = recharge.sum()

# load in the list file to get cumulative mass
lstfile = opj(out_pth, 'SBModel.lst')
lstbud = SwtListBudget(lstfile)
lst_flow, lst_vol = lstbud.get_dataframes()
lst_flow.to_csv(opj(tmp_dir, 's{}_{}_list_flow_budget.csv'.format(scen, sched)))
lst_vol.to_csv(opj(tmp_dir, 's{}_{}_list_vol_budget.csv'.format(scen, sched)))

density = 1000  # kg/m^3
cum2af = 1233.48183754752
# compute the total recharge IN volume [m^3] for entire simulation
recharge_vol = lst_vol['RECHARGE_IN'][-1] / density
rch_vol_af = recharge_vol / cum2af
# compute the total mountain front recharge (wells) IN volume [m^3] for entire simulation
mtnrch_vol = lst_vol['WELLS_IN'][-1] / density
mtnrch_vol_af = mtnrch_vol / cum2af
nat_rch_af = rch_vol_af + mtnrch_vol_af
