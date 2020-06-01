import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn import preprocessing
from sklearn.decomposition import PCA
from pandas.plotting import scatter_matrix
from sklearn.inspection import permutation_importance

opj = os.path.join

cwd = os.getcwd()

fig_dir = opj(cwd, 'scripts', '_fig', '2019-12-09')
if not os.path.exists(fig_dir):
    os.mkdir(fig_dir)

plt.style.use('ggplot')
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.size'] = 12
mpl.rcParams['lines.linewidth'] = 1
mpl.rcParams['lines.color'] = 'gray'
mpl.rcParams['axes.facecolor'] = 'gainsboro'
mpl.rcParams['legend.facecolor'] = 'whitesmoke'
mpl.rcParams['patch.linewidth'] = 0.3
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['lines.markersize'] = 5
# mpl.rcParams['axes.labelweight'] = 'bold'

plt.ioff()

# rename DVs
well_names = ['AL', 'CH', 'CY', 'HA', 'LW', 'LR', 'OP', 'SR', 'SBHS', 'VV', 'VC']
dv_nam = []
for q in range(1, 41):
    for s in well_names:
        dv_nam.append('{}_{}'.format(s, q))

scen = 2

if scen == 1:
    data_dir = opj(cwd, 'SB_Archive', 'ancillary', 'optimization', 'output', 'output_scenario_1')
    # s1_result = pd.read_csv(opj(data_dir, 'result.csv'))
    # converted units and additional objective targets
    s1_add = pd.read_csv(opj(data_dir, 's1_cnvrt_add.csv'))
    s1_data = s1_add.iloc[:, :440]
    s1_data.columns = dv_nam
    s1_pca = PCA().fit(s1_data.values)
    X_mat = s1_data.values
elif scen == 2:
    data_dir = opj(cwd, 'SB_Archive', 'ancillary', 'optimization', 'output', 'output_scenario_2')
    # s2_result = pd.read_csv(opj(data_dir, 'result.csv'))
    # converted units and additional objective targets
    s2_add = pd.read_csv(opj(data_dir, 's2_cnvrt_add.csv'))
    s2_data = s2_add.iloc[:, :440]
    s2_data.columns = dv_nam
    s2_pca = PCA().fit(s2_data.values)
    # rename columns to more descriptive well codes
    X_mat = s2_data.values

objs = ['obj 0', 'obj 1', 'obj 2', 'obj 3']
# targets = ['pumpage[ac-ft]', 'swi[1000mgL]', 'dd_tot[ft]', 'dd_max[ft]',
#            'sos-A', 'sos-B', 'sos-C', 'sos-C5', 'sos-D', 'sos-S', 'sos-G']
targets = ['swi[1000mgL]', 'dd_tot[ft]', 'sos-D']
cons = ['con 0', 'con 1', 'con 2']

# if additional features were created, add them here
# ends_df = pd.read_csv(opj(cwd, 'scripts', '_tmp', 'cl_obs_end.csv'))
# s1_data = pd.concat([s1_data, ends_df], axis=1)
# diffs_df = pd.read_csv(opj(cwd, 'scripts', '_tmp', 'cl_obs_diff.csv'))
# s1_data = pd.concat([s1_data, diffs_df], axis=1)

# obj = 'obj 2'
topn = 10
targ_name = 'sos-D'
for targ_name in targets:
    # TODO put in Dictionary
    # targ_name = targets[obj]
    if scen == 1:
        y_mat = s1_add[targ_name].values
    elif scen == 2:
        y_mat = s2_add[targ_name].values

    X_train, X_test, y_train, y_test = train_test_split(X_mat, y_mat,
                                                        test_size=0.2, random_state=57)

    # regr_1 = DecisionTreeRegressor(max_depth=4)
    # regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=300)
    trnsfrm = True
    # preprocessing for standardization (centering/scaling)
    X_scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_std = X_scaler.transform(X_train)
    X_test_std = X_scaler.transform(X_test)

    # TRAINING
    n = 500
    est = GradientBoostingRegressor(n_estimators=n, learning_rate=0.1,
                                    max_leaf_nodes=11, min_samples_leaf=5, random_state=0,
                                    loss='ls')
    print('Cross-val scores for {}, Scenario {}'.format(targ_name, scen))
    if ~trnsfrm:
        print('not transformed')
        est.fit(X_train, y_train)
        scores = cross_val_score(est, X_train, y_train, cv=10)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        # [print('{:8.5f}'.format(i)) for i in scores]
        print(', '.join('{:6.3f}'.format(s) for s in scores))
    else:
        print('transformed')
        est.fit(X_train_std, y_train)
        scores = cross_val_score(est, X_train_std, y_train, cv=10)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        # [print('{:8.5f}'.format(i)) for i in scores]
        print(', '.join('{:6.3f}'.format(s) for s in scores))

    # TESTING
    if ~trnsfrm:
        mse = mean_squared_error(y_test, est.predict(X_test))
        print('mean squared error = {}'.format(mse))
        y_pred = est.predict(X_test)
        r_squared = est.score(X_test, y_test)
        print('r squared = {}'.format(r_squared))
    else:
        mse = mean_squared_error(y_test, est.predict(X_test_std))
        print('mean squared error = {}'.format(mse))
        y_pred = est.predict(X_test_std)
        r_squared = est.score(X_test_std, y_test)
        print('r squared = {}'.format(r_squared))
        #

    test_score = np.zeros((n,), dtype=np.float64)

    for i, y_pred in enumerate(est.staged_predict(X_test)):
        test_score[i] = est.loss_(y_test, y_pred)

    plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1)
    # Plot feature importance
    feature_importance = est.feature_importances_
    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    sorted_idx = np.flip(sorted_idx)  # need to flip order to put highest first
    pos = np.arange(sorted_idx[:topn].shape[0]) + .5
    # plt.subplot(1, 2, 2)
    plt.barh(pos, feature_importance[sorted_idx][:topn], align='center')
    top = np.array(s1_data.columns)[sorted_idx][:topn]
    plt.yticks(pos, top)
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.subplots_adjust(wspace=0.3)
    plt.savefig(opj(fig_dir, 'importance_s{}_{}_top{}.png'.format(scen, targ_name, topn)))
    # plt.show()

# make plot for least important variables
    plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1)
    pos = np.arange(sorted_idx[-topn:].shape[0]) + .5
    #plt.subplot(1, 2, 2)
    plt.barh(pos, feature_importance[sorted_idx][-topn:], align='center')
    top20 = np.array(s1_data.columns)[sorted_idx][-topn:]
    plt.yticks(pos, top20)
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.subplots_adjust(wspace=0.3)
    plt.savefig(opj(fig_dir, 'importance_s{}_{}_bot{}.png'.format(scen, targ_name, topn)))
    # plt.show()

    # compute permutation importances as an alternative to biased importances above
    result = permutation_importance(est, X_test, y_test, n_repeats=10, random_state=42)
    sorted_idx = result.importances_mean.argsort()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.boxplot(result.importances[sorted_idx][:topn].T,
               vert=False, labels=np.array(s1_data.columns)[sorted_idx][:topn])
    ax.set_title("Permutation Importances (test set)")
    fig.tight_layout()
    plt.savefig(opj(fig_dir, 'perm_importance_s{}_{}_bot{}.png'.format(scen, targ_name, topn)))
    # plt.show()

    # ticks_font = mpl.font_manager.FontProperties(weight='bold')
    # plt.rcParams['axes.labelweight'] = 'bold'
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.boxplot(result.importances[sorted_idx][-topn:].T,
               vert=False, labels=np.array(s1_data.columns)[sorted_idx][-topn:])
    for label in ax.get_yticklabels():
        label.set_weight("heavy")
    ax.set_title("Permutation Importances (test set)")
    fig.tight_layout()
    plt.savefig(opj(fig_dir, 'perm_importance_s{}_{}_top{}.png'.format(scen, targ_name, topn)))
    # plt.show()

    formatter = mpl.ticker.FormatStrFormatter('%2.1e')
    # predict the whole dataset:
    y_pred = est.predict(X_test)
    xll = min([y_mat.min(), y_pred.min()])
    xul = max([y_mat.max(), y_pred.max()])
    buff = (xul - xll) * 0.1
    xmin = xll - buff
    xmax = xul + buff
    ymin = xll - buff
    ymax = xul + buff

    fig, ax = plt.subplots(figsize=(5, 5))
    pc = ax.scatter(y_test, y_pred, marker='o',
                    label='data', edgecolors='none', linewidths=None, alpha=0.7)
    ax.set_aspect('equal')
    ax.axis([xmin, xmax, ymin, ymax])
    ax.plot([xll, xul], [xll, xul], color='navy', lw=1, ls='--', label='1-to-1', alpha=0.5)
    txt = r'$r^2 = $ {:6.4f}'.format(r_squared)
    ax.text(.7, .1, txt, transform=ax.transAxes)
    ax.set_xlabel('{} Simulated with SEAWAT'.format(targ_name))
    ax.set_ylabel('{} Predicted with GBRT'.format(targ_name))
    ax.legend(loc='upper left')
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.tick_params(axis='both', labelsize=8)
    # cb = fig.colorbar(pc, ax=ax, shrink=0.6)
    # cb.set_label('actual Qmax ' r'$(ft^3/day)$')
    # cb.solids.set(alpha=1)  # since points use alpha<1, colorbar looks poor without this
    # plt.suptitle('{}'.format(targ_name), fontsize=12, y=0.95)
    plt.title('Simulated with model vs Predicted with metamodel',
              fontsize=8, color='gray', style='italic', loc='left')
    plt.tight_layout()
    plt.savefig(opj(fig_dir, 'y_pred_s{}_{}_test.png'.format(scen, targ_name)))
    # plt.show()

top5 = s2_result.columns[:440][sorted_idx][:5]
pcols = list(top5) + objs
# pcols = list(top5) + ['pumpage[ac-ft]', 'swi[1000mg/L]', 'dd_tot[ft]', 'dd_max[ft]']
plot_df = s2_result.loc[:, pcols]
# scatter_matrix(plot_df, alpha=0.3, ax=ax)

axs = scatter_matrix(plot_df, alpha=0.3, figsize=(10, 10))
for ax in axs.flatten():
    ax.set_xlabel(ax.get_xlabel(), rotation=45)
    ax.set_ylabel(ax.get_ylabel(), rotation=0)
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
plt.savefig(opj(fig_dir, 'scatter_matrix_s{}.png'.format(scen)))
plt.close()

corr_matrix = s1_result.corr()
print(corr_matrix['obj 1'].sort_values(ascending=False)[:20])
s1_all = pd.concat([s1_result, diffs_df], axis=1)
corr_matrix_all = s1_all.corr()
print(corr_matrix_all['obj 1'].sort_values(ascending=False)[:20])

plt.close('all')
