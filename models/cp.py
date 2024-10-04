from skgarden import RandomForestQuantileRegressor
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats.mstats import mquantiles
from sklearn.ensemble import RandomForestRegressor
from statsmodels.stats.weightstats import DescrStatsW

import warnings
warnings.filterwarnings("ignore")  # 忽略警告

def aecp(X, Y, X_test, alpha):
    """
    Compute aecp prediction interval

    Input
    X         : n x p data matrix of explanatory variables
    Y         : n x 1 vector of response variables
    X_test    : n x p test data matrix of explanatory variables
    black_box : sklearn model object with 'fit' and 'predict' methods
    alpha     : 1 - target coverage level
    """

    # Output placeholder
    lower = None
    upper = None

    # Split the data into training and calibration sets
    X_train, X_calib, Y_train, Y_calib = train_test_split(
        X, Y, test_size=0.4, shuffle=False, random_state=2022)


    # Black box model of choice
    black_box = RandomForestRegressor(
        n_estimators=10, min_samples_split=2, random_state=2022)

    # Fit the black box model on the training data
    black_box.fit(X_train, Y_train)

    # Compute residuals on the calibration data
    residuals_calib = np.abs(Y_calib - black_box.predict(X_calib))

    # Compute suitable empirical quantile of absolute residuals
    n_calib = len(Y_calib)
    level_adjusted = (1.0-alpha) * (1 + 1/n_calib)
    Q_hat = mquantiles(residuals_calib, prob=level_adjusted)[0]

    # Construct prediction bands
    Y_hat = black_box.predict(X_test)
    lower = Y_hat - Q_hat
    upper = Y_hat + Q_hat

    return np.array([lower[0], upper[0]]), Y_hat

def secp(X, Y, X_test, alpha):
    """
    Compute secp prediction interval

    Input
    X         : n x p data matrix of explanatory variables
    Y         : n x 1 vector of response variables
    X_test    : n x p test data matrix of explanatory variables
    black_box : sklearn model object with 'fit' and 'predict' methods
    alpha     : 1 - target coverage level
    """

    # Output placeholder
    lower = None
    upper = None

    # Split the data into training and calibration sets
    X_train, X_calib, Y_train, Y_calib = train_test_split(
        X, Y, test_size=0.4, shuffle=False, random_state=2022)

    # Black box model of choice
    black_box = RandomForestRegressor(
        n_estimators=10, min_samples_split=2, random_state=2022)

    # Fit the black box model on the training data
    black_box.fit(X_train, Y_train)

    # Compute residuals on the calibration data
    residuals_calib = Y_calib - black_box.predict(X_calib)

    # Compute suitable empirical quantile of absolute residuals
    n_calib = len(Y_calib)


    # Construct prediction bands
    Y_hat = black_box.predict(X_test)

    level_low = (1.0-alpha/2) * (1 + 1/n_calib)
    low = mquantiles(-residuals_calib, prob=level_low)[0]   
    level_high = (1.0-alpha/2) * (1 + 1/n_calib)
    high = mquantiles(residuals_calib, prob=level_high)[0]

    lower = Y_hat - low
    upper = Y_hat + high   

    return np.array([lower[0], upper[0]]), Y_hat


def aecp_aci(count, X, Y, X_test, Y_test, alpha, alpha_t, erro_t, gamma):
    """
    Compute aecp_aci prediction interval

    Input
    X         : n x p data matrix of explanatory variables
    Y         : n x 1 vector of response variables
    X_test    : n x p test data matrix of explanatory variables
    black_box : sklearn model object with 'fit' and 'predict' methods
    alpha     : 1 - target coverage level
    gamma     : update step
    erro_t    : empirical miscoverage frequency
    """

    # Output placeholder
    lower = None
    upper = None

    # Split the data into training and calibration sets
    X_train, X_calib, Y_train, Y_calib = train_test_split(
        X, Y, test_size=0.4, shuffle=False, random_state=2022)


    # Black box model of choice
    black_box = RandomForestRegressor(
        n_estimators=10, min_samples_split=2, random_state=2022)

    # Fit the black box model on the training data
    black_box.fit(X_train, Y_train)

    # Compute residuals on the calibration data
    residuals_calib = np.abs(Y_calib - black_box.predict(X_calib))

    rho = 1
    sita = 0
    k_n = 10

    if count <= 1:
        alpha_t = alpha
        sita = 0
    elif count <= k_n:
        ws = rho**(np.arange((count-1), 0, -1)-1)
        ws = ws / np.sum(ws)
        sita = alpha - np.sum(ws*erro_t[0:(count-1)])  # 论文公式(4)
    else:
        ws = rho**(np.arange((count-1), (count-k_n), -1)-1)
        ws = ws / np.sum(ws)
        sita = alpha - np.sum(ws*erro_t[(count-k_n):(count-1)])

    if(0 < alpha_t + gamma*sita < 1):
        alpha_t = alpha_t + gamma*sita

    n_calib = len(Y_calib)
    level_adjusted = (1.0-alpha_t) * (1 + 1/n_calib)
    Q_hat = mquantiles(residuals_calib, prob=level_adjusted)[0]

    # Construct prediction interval
    Y_hat = black_box.predict(X_test)

    lower = Y_hat - Q_hat
    upper = Y_hat + Q_hat

    if((Y_test[0] < lower[0]) | (Y_test[0] > upper[0])):
        erro_t[count] = 1

    return np.array([lower[0], upper[0]]), alpha_t, erro_t[count]


def adaSecp_aci(count, X, Y, X_test, Y_test, alpha, alpha_t, erro_t, lamda=10, window_size_k=10, gamma=0.005):
    """
    Compute adaSecp_aci prediction bands

    Input
    X         : n x p data matrix of explanatory variables
    Y         : n x 1 vector of response variables
    X_test    : n x p test data matrix of explanatory variables
    black_box : sklearn model object with 'fit' and 'predict' methods
    alpha     : 1 - target coverage level
    gamma     : update step
    erro_t    : empirical miscoverage frequency
    """

    # Output placeholder
    lower = None
    upper = None
   

    # Split the data into training and calibration sets
    X_train, X_calib, Y_train, Y_calib = train_test_split(
        X, Y, test_size=0.4, shuffle=False, random_state=2022)

    
    # Black box model of choice
    black_box = RandomForestRegressor(
        n_estimators=10, min_samples_split=2, random_state=2022)


    # Fit the black box model on the training data
    black_box.fit(X_train, Y_train)

    # Compute residuals on the calibration data
    residuals_calib_sign = Y_calib - black_box.predict(X_calib)

    # median
    sign_median = np.median(residuals_calib_sign)
    sign_local_median = np.median(residuals_calib_sign[-window_size_k:])
    fac = np.max(residuals_calib_sign)-np.min(residuals_calib_sign)
    local_entire = (sign_local_median - sign_median)/fac

    trans = np.tanh(lamda*local_entire)

    r_l = (1+trans)/2
    r_h = (1-trans)/2

    sita = 0
    k_n = 10
    if count <= 1:
        alpha_t = alpha
        sita = 0
    elif count <= k_n:
        sita = alpha - np.mean(erro_t[0:(count-1)])
    else:
        sita = alpha - np.mean(erro_t[(count-k_n):(count-1)])


    alpha_t = alpha_t + gamma*sita          # 当 alpha_t<0 时，取值为残差的最值

    # Compute suitable empirical quantile of absolute residuals
    n_calib = len(Y_calib)

    # Construct prediction bands
    Y_hat = black_box.predict(X_test)

    level_low = (alpha_t*r_l) * (1 + 1/n_calib)
    low = mquantiles(residuals_calib_sign, prob=level_low)[0]  # 这个分位数函数不会取无穷值，只取到最小值或者最大值
    level_high = (1.0-alpha_t*r_h) * (1 + 1/n_calib)
    high = mquantiles(residuals_calib_sign, prob=level_high)[0]

    # print(low, high)
    lower = Y_hat + low
    upper = Y_hat + high

    if((Y_test[0] < lower[0]) | (Y_test[0] > upper[0])):
        erro_t[count] = 1

    return np.array([lower[0], upper[0]]), alpha_t, erro_t[count], Y_hat

def adaSecp(X, Y, X_test, alpha):
    """
    Compute adaSecp prediction bands

    Input
    X         : n x p data matrix of explanatory variables
    Y         : n x 1 vector of response variables
    X_test    : n x p test data matrix of explanatory variables
    black_box : sklearn model object with 'fit' and 'predict' methods
    alpha     : 1 - target coverage level
    """

    # Output placeholder
    lower = None
    upper = None

    # Split the data into training and calibration sets
    X_train, X_calib, Y_train, Y_calib = train_test_split(
        X, Y, test_size=0.4, shuffle=False, random_state=2022)


    # Black box model of choice
    black_box = RandomForestRegressor(
        n_estimators=10, min_samples_split=2, random_state=2022)

    # Fit the black box model on the training data
    black_box.fit(X_train, Y_train)

    # Compute residuals on the calibration data
    residuals_calib_sign = Y_calib - black_box.predict(X_calib)

    # median
    sign_median = np.median(residuals_calib_sign)
    sign_local_median = np.median(residuals_calib_sign[-10:])
    fac = np.max(residuals_calib_sign)-np.min(residuals_calib_sign)
    local_entire = (sign_local_median - sign_median)/fac

    trans = np.tanh(local_entire)

    r_l = (1+trans)/2
    r_h = (1-trans)/2

    n_calib = len(Y_calib)

    Y_hat = black_box.predict(X_test)


    level_low = (1.0-alpha*r_l) * (1 + 1/n_calib)
    low = mquantiles(-residuals_calib_sign, prob=level_low)[0]   
    level_high = (1.0-alpha*r_h) * (1 + 1/n_calib)
    high = mquantiles(residuals_calib_sign, prob=level_high)[0]

    lower = Y_hat - low
    upper = Y_hat + high

    return np.array([lower[0], upper[0]]), Y_hat


def secp_aci(count, X, Y, X_test, Y_test, alpha, alpha_t, erro_t, gamma):
    """
    Compute secp_aci prediction interval

    Input
    X         : n x p data matrix of explanatory variables
    Y         : n x 1 vector of response variables
    X_test    : n x p test data matrix of explanatory variables
    black_box : sklearn model object with 'fit' and 'predict' methods
    alpha     : 1 - target coverage level
    gamma     : update step
    erro_t    : empirical miscoverage frequency
    """

    # Output placeholder
    lower = None
    upper = None

    # Split the data into training and calibration sets
    X_train, X_calib, Y_train, Y_calib = train_test_split(
        X, Y, test_size=0.4, shuffle=False, random_state=2022)

    # Black box model of choice
    black_box = RandomForestRegressor(
        n_estimators=10, min_samples_split=2, random_state=2022)

    # Fit the black box model on the training data
    black_box.fit(X_train, Y_train)

    # Compute residuals on the calibration data
    residuals_calib = Y_calib - black_box.predict(X_calib)

    rho = 1
    sita = 0
    k_n = 10

    if count <= 1:
        alpha_t = alpha
        sita = 0
    elif count <= k_n:
        ws = rho**(np.arange((count-1), 0, -1)-1)
        ws = ws / np.sum(ws)
        sita = alpha - np.sum(ws*erro_t[0:(count-1)])
    else:
        ws = rho**(np.arange((count-1), (count-k_n), -1)-1)
        ws = ws / np.sum(ws)
        sita = alpha - np.sum(ws*erro_t[(count-k_n):(count-1)])

    if(0 < alpha_t + gamma*sita < 1):
        alpha_t = alpha_t + gamma*sita

    # Compute suitable empirical quantile of absolute residuals
    n_calib = len(Y_calib)

    # Construct prediction bands
    Y_hat = black_box.predict(X_test)


    level_low = (1.0-alpha_t/2)  * (1 + 1/n_calib)
    low = mquantiles(-residuals_calib, prob=level_low)[0]   
    level_high = (1.0-alpha_t/2) * (1 + 1/n_calib)
    high = mquantiles(residuals_calib, prob=level_high)[0]

    # print(low, high)
    lower = Y_hat - low
    upper = Y_hat + high

    if((Y_test[0] < lower[0]) | (Y_test[0] > upper[0])):
        erro_t[count] = 1

    return np.array([lower[0], upper[0]]), alpha_t, erro_t[count]



def cqr(X, Y, X_test, alpha):
    """
    Compute split-conformal quantile regression prediction interval.
    Uses quantile random forests as a black box

    Input
    X         : n x p data matrix of explanatory variables
    Y         : n x 1 vector of response variables
    X_test    : n x p test data matrix of explanatory variables
    alpha     : 1 - target coverage level
    """

    # Output placeholder
    lower = None
    upper = None

    # Split the data into training and calibration sets
    X_train, X_calib, Y_train, Y_calib = train_test_split(
        X, Y, test_size=0.4,  shuffle=False, random_state=2022)  # shuffle=False,


    # Fit a quantile regression model
    black_box = RandomForestQuantileRegressor(
        n_estimators=10, min_samples_split=2, random_state=2022)
    black_box.fit(X_train, Y_train)

    # Estimate conditional quantiles for calibration set
    lower = black_box.predict(X_calib, quantile=100*alpha/2)
    upper = black_box.predict(X_calib, quantile=100*(1-alpha/2))

    # Compute conformity scores on the calibration data
    residuals_calib = np.maximum(
        Y_calib - upper, lower - Y_calib)

    # Compute suitable empirical quantile of absolute residuals
    n_calib = len(Y_calib)
    level_adjusted = (1.0-alpha)*(1.0+1.0/float(n_calib))
    # Q_hat = mquantiles(residuals_calib, prob=level_adjusted)[0] #原来使用的分位数函数

    wq = DescrStatsW(data=residuals_calib)
    Q_hat = wq.quantile(probs=np.array([level_adjusted]), return_pandas=False)

    # Construct prediction bands
    lower = black_box.predict(X_test, quantile=100*alpha/2)
    upper = black_box.predict(X_test, quantile=100*(1-alpha/2))
    lower = lower - Q_hat
    upper = upper + Q_hat

    return np.array([lower[0], upper[0]])



# calculate WIS
def calculate_wis(y_true, lower, upper, alpha):
    wis_total = 0
    k = len(y_true)
    
    for i in range(k):
        width = upper[i] - lower[i]
        penalty_lower = (lower[i] - y_true[i]) * (y_true[i] < lower[i])
        penalty_upper = (y_true[i] - upper[i]) * (y_true[i] > upper[i])
        wis = width + (2 / alpha) * penalty_lower + (2 / alpha) * penalty_upper
        wis_total += wis
    
    return wis_total / k
