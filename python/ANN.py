import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn import preprocessing
from sklearn.decomposition import PCA
from pandas.plotting import scatter_matrix

opj = os.path.join

cwd = os.getcwd()

fig_dir = opj(cwd, 'scripts', '_fig')
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
s1_data = s1_result.iloc[:, :440]
s1_pca = PCA().fit(s1_data.values)

data_dir = opj(cwd, 'SB_Archive', 'ancillary', 'optimization', 'output', 'output_scenario_2')
s2_result = pd.read_csv(opj(data_dir, 'result.csv'))
s2_data = s2_result.iloc[:, :440]
s2_pca = PCA().fit(s2_data.values)

# scenario 1

objs = ['obj 0', 'obj 1', 'obj 2', 'obj 3']
cons = ['con 0', 'con 1', 'con 2']
scen = 1

# if additional features were created, add them here
ends_df = pd.read_csv(opj(cwd, 'scripts', '_tmp', 'cl_obs_end.csv'))
s1_data = pd.concat([s1_data, ends_df], axis=1)
diffs_df = pd.read_csv(opj(cwd, 'scripts', '_tmp', 'cl_obs_diff.csv'))
s1_data = pd.concat([s1_data, diffs_df], axis=1)

s1_all = pd.concat([s1_result, ends_df], axis=1)
s1_all = pd.concat([s1_result, diffs_df], axis=1)
corr_matrix_all = s1_all.corr()
print(corr_matrix_all['obj 1'].sort_values(ascending=False)[:30])

# repeat but with converted units and proper headers plus additional SOS and SU columns
# these are copied from the 's1_all' tab of E:\SB_research\mgmt\SB_Borg_v02\results\scenario1.xlsx
data_dir = opj(cwd, 'SB_Archive', 'ancillary', 'optimization', 'output', 'output_scenario_1')
s1_add = pd.read_csv(opj(data_dir, 's1_cnvrt_add.csv'))
corr_matrix_add = s1_add.corr()
corr_matrix_add.to_csv(opj(cwd, 'scripts', '_tmp', 'corr_matrix_add.csv'))
print(corr_matrix_add['swi[1000mg/L]'].sort_values(ascending=False)[:30])
plt.matshow(corr_matrix_add)
plt.show()

plt.figure(figsize=(10, 10))
plt.imshow(corr_matrix_add.iloc[:12, :12])
plt.xticks(range(12), s1_add.columns[:12], fontsize=6, rotation=45)
plt.yticks(range(12), s1_add.columns[:12], fontsize=6)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=6)
plt.savefig(opj(cwd, 'scripts', '_tmp', 'corr_matrix_add.png'), dpi=300)
plt.close()

X_mat = s1_data.values
obj = objs[1]
for obj in objs:
    targ_name = obj
    y_mat = s1_result[obj].values

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
    est = MLPRegressor(activation='identity', solver='lbfgs')
    print('Cross-val scores for {}, Scenario {}'.format(obj, scen))
    if ~trnsfrm:
        est.fit(X_train, y_train)
        scores = cross_val_score(est, X_train, y_train, cv=10)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        # [print('{:8.5f}'.format(i)) for i in scores]
        print(', '.join('{:6.3f}'.format(s) for s in scores))
    else:
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

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Deviance')
    plt.plot(np.arange(n) + 1, est.train_score_, 'b-',
         label='Training Set Deviance')
    plt.plot(np.arange(n) + 1, test_score, 'r-',
         label='Test Set Deviance')
    plt.legend(loc='upper right')
    plt.xlabel('Boosting Iterations')
    plt.ylabel('Deviance')
    # #############################################################################
    # Plot feature importance
    feature_importance = est.feature_importances_
    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    sorted_idx = np.flip(sorted_idx)  # need to flip order to put highest first
    pos = np.arange(sorted_idx[:20].shape[0]) + .5
    plt.subplot(1, 2, 2)
    plt.barh(pos, feature_importance[sorted_idx][:20], align='center')
    top20 = np.array(s1_data.columns)[sorted_idx][:20]
    plt.yticks(pos, top20)
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.subplots_adjust(wspace=0.3)
    plt.savefig(opj(fig_dir, 'importance_s{}_{}.png'.format(scen, obj)))
    plt.show()

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
    ax.set_xlabel('{}'.format(targ_name))
    ax.set_ylabel('{} Predicted'.format(targ_name))
    ax.legend(loc='upper left')
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    # cb = fig.colorbar(pc, ax=ax, shrink=0.6)
    # cb.set_label('actual Qmax ' r'$(ft^3/day)$')
    # cb.solids.set(alpha=1)  # since points use alpha<1, colorbar looks poor without this
    # plt.suptitle('{}'.format(targ_name), fontsize=12, y=0.95)
    plt.title('All data (after filtering)\nSimulated with model vs Predicted with metamodel',
              fontsize=8, color='gray', style='italic', loc='left')
    plt.tight_layout()
    plt.savefig(opj(fig_dir, 'y_pred_s{}_{}_test.png'.format(scen, targ_name)))
    plt.show()

top10 = s1_result.columns[:440][sorted_idx][:10]
pcols = list(top10) + objs
plot_df = s1_result.loc[:, pcols]
# scatter_matrix(plot_df, alpha=0.3, ax=ax)

axs = scatter_matrix(plot_df, alpha=0.3, figsize=(15, 15))
for ax in axs.flatten():
    # ax.xaxis.get_xticklabels().set_fontsize(4)
    ax.set_xlabel(ax.get_xlabel(), rotation=90)
    ax.set_ylabel(ax.get_ylabel(), rotation=0)
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    # ax.xaxis.set_tick_params(labelsize=4)
    # ax.yaxis.set_tick_params(labelsize=4)
plt.savefig(opj(fig_dir, 'scatter_matrix_s{}.png'.format(scen)))


plt.close()
