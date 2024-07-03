import numpy as np
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat
from glob import glob

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, auc, accuracy_score, mean_squared_error, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split

# !pip install basemap
from matplotlib import colors
from mpl_toolkits.basemap import Basemap, cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import rgb2hex, Normalize
from matplotlib.colorbar import ColorbarBase


################################
# Data preparation section
################################

def column_names():

    col_name = ['T2', 'q2', 'Ps', 'TPW', 
            'T10', 'T09', 'T08', 'T07', 'T06', 'T05', 'T04', 'T03', 'T02', 'T01',
            'q10', 'q09', 'q08', 'q07', 'q06', 'q05', 'q04', 'q03', 'q02', 'q01',
            'z10', 'z09', 'z08', 'z07', 'z06', 'z05', 'z04', 'z03', 'z02', 'z01',
            'c10', 'c09', 'c08', 'c07', 'c06', 'c05', 'c04', 'c03', 'c02', 'c01',
            'i10', 'i09', 'i08', 'i07', 'i06', 'i05', 'i04', 'i03', 'i02', 'i01',
            'v10', 'v09', 'v08', 'v07', 'v06', 'v05', 'v04', 'v03', 'v02', 'v01',
            'w10', 'w09', 'w08', 'w07', 'w06', 'w05', 'w04', 'w03', 'w02', 'w01',]

    df_col_name = col_name.copy()
    df_col_name.append('nc')
    df_col_name.append('date')

    return col_name, df_col_name



class CPF_Preparation():
    
    def __init__(self, path_data='/panfs/jay/groups/0/ebtehaj/kang0511/DIR_Jupyter/CPF_data/CPF_detection_seasonal.mat'):

        mat_detection = loadmat(path_data)
        col_name, df_col_name = column_names()
        
        self.mam = pd.DataFrame(mat_detection['array_mam'], columns=df_col_name)
        self.jja = pd.DataFrame(mat_detection['array_jja'], columns=df_col_name)
        self.son = pd.DataFrame(mat_detection['array_son'], columns=df_col_name)
        self.djf = pd.DataFrame(mat_detection['array_djf'], columns=df_col_name)


        
def CPF_test_dataset(path_data='/panfs/jay/groups/0/ebtehaj/kang0511/DIR_Jupyter/CPF_data/CPF_testdata.mat', arg_regression=1):
    
        mat_cont = loadmat(path_data)
        col_name, df_col_name = column_names()
        
        mat_X = mat_cont['mat_99_test']
        lab_y = mat_cont['convec_99_test']

        # construct a dataframe
        df = pd.DataFrame(mat_X, columns=col_name)
        df.insert(74, "nc", lab_y, True)
        df.insert(75, "date", mat_cont['date_99_test'])
        
        if arg_regression == 1:
            # dataset for regression
            nz_df = df[df['nc']>0]
            nz_df.reset_index(drop=True)

        else:
            # dataset for detection
            nz_df = df
            nz_df.loc[nz_df['nc']>0, 'nc'] = 1
                
        return nz_df


    
def CPF_train_dataset(path_data='/panfs/jay/groups/0/ebtehaj/kang0511/DIR_Jupyter/CPF_data/CPF_testdata.mat', arg_regression=1):
    
        mat_cont = loadmat(path_data)
        col_name, df_col_name = column_names()
        
        mat_X = mat_cont['mat_99_train']
        lab_y = mat_cont['convec_99_train']

        # construct a dataframe
        df = pd.DataFrame(mat_X, columns=col_name)
        df.insert(74, "label", lab_y, True)
        df.insert(75, "date", mat_cont['date_99_train'])
        
        if arg_regression == 1:
            # dataset for regression
            nz_df = df[df['label']>0]
            nz_df.reset_index(drop=True)

        else:
            # dataset for detection
            nz_df = df
            nz_df.loc[nz_df['label']>0, 'nc'] = 1
                
        return nz_df

    
    
class Para_settings():
    
    def __init__(self):
        self.det = {
                'max_depth': 10,
                'eta': 0.1,
                'lambda': 0.8,
                'alpha': 0.1,
                'tree_method': 'exact',
                'objective': 'binary:logistic',
                }
        
        self.reg = {
                'max_depth': 10,
                'eta': 0.1,
                'lambda': 0.8,
                'alpha': 0.1,
                'objective': 'count:poisson',
                'eval_metric': 'rmse',
                }
        

        
class Labels_occurrence():
    
    def __init__(self, sub_df):
        
        occur_mam = sub_df.mam['nc'].to_numpy().copy()
        occur_jja = sub_df.jja['nc'].to_numpy().copy()
        occur_son = sub_df.son['nc'].to_numpy().copy()
        occur_djf = sub_df.djf['nc'].to_numpy().copy()

        occur_mam[occur_mam>0] = 1
        occur_jja[occur_jja>0] = 1
        occur_son[occur_son>0] = 1
        occur_djf[occur_djf>0] = 1

        self.mam = occur_mam
        self.jja = occur_jja
        self.son = occur_son
        self.djf = occur_djf
    

    
class Labels_regression():
    
    def __init__(self, sub_df):
        
        label_mam = sub_df.mam['nc'].to_numpy().copy()
        label_jja = sub_df.jja['nc'].to_numpy().copy()
        label_son = sub_df.son['nc'].to_numpy().copy()
        label_djf = sub_df.djf['nc'].to_numpy().copy()

        label_mam = label_mam[label_mam>0]
        label_jja = label_jja[label_jja>0]
        label_son = label_son[label_son>0]
        label_djf = label_djf[label_djf>0]

        self.mam = label_mam
        self.jja = label_jja
        self.son = label_son
        self.djf = label_djf
    
        
class Labels_predict_detection():

    def __init__(self, sub_df, xgb_detect):
        
        col_name, df_col_name = column_names()

        self.mam = xgb_detect.predict(sub_df.mam[col_name])
        self.jja = xgb_detect.predict(sub_df.jja[col_name])
        self.son = xgb_detect.predict(sub_df.son[col_name])
        self.djf = xgb_detect.predict(sub_df.djf[col_name])
        
        
        
class Labels_predict_regression():

    def __init__(self, sub_df, xgb_reg):
        
        col_name, df_col_name = column_names()

        self.mam = xgb_reg.predict(sub_df.mam.loc[sub_df.mam['nc']>0, col_name])
        self.jja = xgb_reg.predict(sub_df.jja.loc[sub_df.jja['nc']>0, col_name])
        self.son = xgb_reg.predict(sub_df.son.loc[sub_df.son['nc']>0, col_name])
        self.djf = xgb_reg.predict(sub_df.djf.loc[sub_df.djf['nc']>0, col_name])

        
def Labels_regression_annual(test_df):
    
    label_annual = test_df['nc'].to_numpy().copy()
    
    return label_annual[label_annual>0]


def Labels_predict_regression_annual(test_df, xgb_reg):

    col_name, df_col_name = column_names()
    
    return xgb_reg.predict(test_df.loc[test_df['nc']>0, col_name])

################################
# Visualization section
################################

from matplotlib import cm
from matplotlib.colors import Normalize 
from scipy.interpolate import interpn


def density_scatter(x_0, y_0, ax = None, sort = True, bins = 50, **kwargs ):
    # https://stackoverflow.com/a/53865762
    
    x = x_0[(x_0<100)&(y_0<100)]
    y = y_0[(x_0<100)&(y_0<100)]
        
    if ax is None :
        fig, ax = plt.subplots()
    
    data, x_e, y_e = np.histogram2d(x, y, bins=bins, density=True )
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

    #To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter( x, y, c=z, s=0.5, **kwargs )

    norm = Normalize(vmin = np.min(z), vmax = np.max(z))
#     cbar = fig.colorbar(cm.ScalarMappable(norm = norm), ax=ax)
#     cbar.ax.set_ylabel('Density')

    return ax

def hrrr_coord():
    mat_cont = loadmat('/home/ebtehaj/kang0511/DIR_Jupyter/CPF_data/CPF_coordinates.mat')

#     lat_hrrr = mat_cont['lat_hrrr']
#     lon_hrrr = mat_cont['lon_hrrr']
#     mask_hrrr = mat_cont['mask_hrrr']

    lat_coarse = mat_cont['lat_coarse']
    lon_coarse = mat_cont['lon_coarse']
    mask_coarse = mat_cont['mask_coarse']
            
    return lat_coarse, lon_coarse, mask_coarse

    
def predict_storm(xgb_detect, xgb_reg, id_file=5083):

    lat_coarse, lon_coarse, mask_coarse = hrrr_coord()    
    mat_files = glob('/scratch.global/kang0511_scratch/coarse_compatible/csp_10_99/*.mat')

    
    # id_file = 5083  # be careful of index in Python and Matlab
    mat_files[id_file]

    tp_str = mat_files[0]
    tp_str.split('c99_')[1].split('.mat')[0]
    mat_cont = loadmat(mat_files[1])
    # sorted(mat_cont.keys())

    fpath = mat_files[id_file]

    mat_cont = loadmat(fpath)    
    str_date = fpath.split('c99_')[1].split('.mat')[0]

    var_X = mat_cont['mat_99']
    lab_Y = mat_cont['n_convec_99']

    lab_y = lab_Y.copy()
    lab_y[lab_y>0] = 1

    pred_y = xgb_detect.predict(var_X)
    pred_y = np.array(pred_y).reshape((-1,1))

    # False Positive
    pred_y[(pred_y==1)&(lab_y==0)] = -1
    # False Negative
    pred_y[(pred_y==0)&(lab_y==1)] = -2
        
    arr_pred = pred_y.reshape(np.shape(lat_coarse), order='F')
    arr_lab = lab_y.reshape(np.shape(lat_coarse), order='F')
    
    
    reg_y = xgb_reg.predict(var_X)
    reg_y = np.multiply(np.array(reg_y).reshape((-1,1)), lab_y)

    arr_anc = lab_Y.reshape(np.shape(lat_coarse), order='F')
    arr_reg = reg_y.reshape(np.shape(lat_coarse), order='F')

    mask_zero = arr_anc.copy()
    mask_zero[mask_zero==0] = np.nan

    zero_arr_anc = np.multiply(mask_zero, arr_anc)
    zero_arr_reg = np.multiply(mask_zero, arr_reg)

    return arr_lab, arr_pred, zero_arr_anc, zero_arr_reg

def plot_confusion_matrix(fig_conf, axes_conf, label_det, pred_det):
    axes_conf[0].set_title("MAM")
    cm_mam = ConfusionMatrixDisplay.from_predictions(label_det.mam, pred_det.mam, cmap=plt.cm.Blues, normalize='all', colorbar=False, ax=axes_conf[0])

    axes_conf[1].set_title("JJA")
    cm_mam = ConfusionMatrixDisplay.from_predictions(label_det.jja, pred_det.jja, cmap=plt.cm.Blues, normalize='all', colorbar=False, ax=axes_conf[1])

    axes_conf[2].set_title("SON")
    cm_mam = ConfusionMatrixDisplay.from_predictions(label_det.son, pred_det.son, cmap=plt.cm.Blues, normalize='all', colorbar=False, ax=axes_conf[2])

    axes_conf[3].set_title("DJF")
    cm_mam = ConfusionMatrixDisplay.from_predictions(label_det.djf, pred_det.djf, cmap=plt.cm.Blues, normalize='all', colorbar=False, ax=axes_conf[3])

    for ii in range(4):      
        axes_conf[ii].set_xlabel('')
        axes_conf[ii].set_ylabel('')
        if ii > 0:
            axes_conf[ii].set_yticklabels(['', ''])

    fig_conf.text(0.55, 0.00, 'Predicted occurrence', ha='center')
    fig_conf.text(0.04, 0.29, 'Actual occurrence', rotation='vertical')

    return fig_conf, axes_conf


def plot_density_scatter(fig_reg, axes_reg, label_reg, pred_reg):
    
    axes_reg[0].set_title("MAM")
    density_scatter(label_reg.mam, pred_reg.mam, ax=axes_reg[0])

    axes_reg[1].set_title("JJA")
    density_scatter(label_reg.jja, pred_reg.jja, ax=axes_reg[1])

    axes_reg[2].set_title("SON")
    density_scatter(label_reg.son, pred_reg.son, ax=axes_reg[2])

    axes_reg[3].set_title("DJF")
    density_scatter(label_reg.djf, pred_reg.djf, ax=axes_reg[3])

    for ii in range(4):  
            axes_reg[ii].plot([0, 100], [0, 100], 'k--', linewidth=1)        
            axes_reg[ii].set_xlim(0, 100)
            axes_reg[ii].set_ylim(0, 100)
            axes_reg[ii].set_aspect('equal', adjustable='box')
            axes_reg[ii].set_xticks([0, 25, 50, 75, 100])
            axes_reg[ii].set_yticks([0, 25, 50, 75, 100])
            axes_reg[ii].grid(visible=True, axis='both', linestyle=':')
            if ii > 0:
                axes_reg[ii].set_yticklabels(['', '', '', '', ''])

    fig_reg.text(0.54, 0.04, '$\hat{n}_c$', ha='center', fontsize=12)
    fig_reg.text(0.04, 0.54, '$n_c$', va='center', rotation='vertical', fontsize=12)
    
    return fig_reg, axes_reg


def plot_density_scatter_annual(fig_reg, axes_reg, label_reg, pred_reg):
    
    density_scatter(label_reg, pred_reg, ax=axes_reg, cmap=plt.cm.plasma, bins=50, alpha=0.04)

    axes_reg.plot([0, 100], [0, 100], 'k--', linewidth=1)        
    axes_reg.set_xlim(0, 100)
    axes_reg.set_ylim(0, 100)
    axes_reg.set_aspect('equal', adjustable='box')
    axes_reg.set_xticks([0, 25, 50, 75, 100])
    axes_reg.set_yticks([0, 25, 50, 75, 100])
    axes_reg.grid(visible=True, axis='both', linestyle=':')

    fig_reg.text(0.54, 0.04, '$\hat{n}_c$', ha='center', fontsize=12)
    fig_reg.text(0.04, 0.54, '$n_c$', va='center', rotation='vertical', fontsize=12)
    
    return fig_reg, axes_reg
