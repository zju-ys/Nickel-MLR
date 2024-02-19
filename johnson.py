import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def Johnson(X,Y):
    '''

    X is the independent variable
    Y is the dependent variable

    This script refers to the following content.
    Maida Ratke https://morioh.com/a/5666f4d6df21/key-driver-analysis-in-python
    Johnson , J. W. 2000 . A heuristic method for estimating the relative weight of predictor variables in multiple regression. . Multivar. Behav. Res. , 35 : 1 – 19 .
    
    '''
    column_name = X.columns
    X = StandardScaler().fit_transform(X)
    Y = StandardScaler().fit_transform(Y.reshape(-1,1)).flatten()#Standardize X and ddG


    df = pd.DataFrame(X,columns=column_name)
    df['Y'] = Y

    corr_matrix = df.apply(pd.to_numeric, errors='coerce').corr()

    # Extract the necessary parts of the correlation matrix
    corr_X = corr_matrix.iloc[:-1, :-1].copy()  # Exclude the last row and column (ddG)
    corr_Xy = corr_matrix.iloc[:-1, -1].copy()  # Only the last column, excluding itself

    # Eigen decomposition
    w_corr_X, v_corr_X = np.linalg.eig(corr_X)

    # Diagonal matrix for eigenvalues
    diag_idx = np.diag_indices(len(corr_X))
    diag = np.zeros((len(corr_X), len(corr_X)), float)
    diag[diag_idx] = w_corr_X

    # Calculate delta
    delta = np.sqrt(diag)

    # Calculate coefficients
    coef_xz = v_corr_X @ delta @ v_corr_X.T
    coef_yz = np.linalg.inv(coef_xz) @ corr_Xy

    # Calculate R^2
    r2 = sum(np.square(coef_yz))

    # Calculate raw and normalized relative weights
    raw_relative_weights = np.square(coef_xz) @ np.square(coef_yz[:, np.newaxis])
    normalized_relative_weights = (raw_relative_weights / r2) * 100

    raw_weights_df = pd.DataFrame(raw_relative_weights, index=column_name, columns=['Raw Relative Weights'])
    normalized_weights_df = pd.DataFrame(normalized_relative_weights, index=column_name, columns=['Normalized Relative Weights'])

    print("Raw Relative Weights:\n", raw_weights_df)
    print("\nNormalized Relative Weights:\n", normalized_weights_df)

    return raw_relative_weights

x= pd.read_csv('PhysOrg.csv')[['B1_9_11','BV_10','d_1_2']]
er=pd.read_csv('SMILES.csv')['er'].to_numpy()
ee = np.array([i if not np.isnan(i) else 0 for i in er])/100*2-1   #er2ee
T = 80 + 273.15
ddG = -8.314 * T * np.log((1-ee)/(1+ee))/1000/4.18
Johnson(x,ddG)