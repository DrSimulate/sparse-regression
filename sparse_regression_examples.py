import numpy as np
from sklearn.linear_model import LinearRegression, Lasso
from matplotlib import pyplot as plt
np.random.seed(0) # fix the random seed to make sure that always the same noise is added to the data

'''
Note: Please be aware that usually the data is normalized or standardized before solving the sparse regression problem. For example, the feature vectors, i.e., the columns of the matrix features_2(x_data), may be devided by their norms. Then, after solving the regression problem, the parameters must be devided by these feature vector norms to recover the original parameters. This normalization has no influence on the solution of the ordinary regression problem, but it matters for the sparse regression problem. Here, no normalization is applied.
'''

'''
Defining the true function that we want to derive.
'''
def f_true(x):
    return np.cos(x)


'''
Generating data and adding random noise to the data.
'''
N_data = 120
x_data = np.linspace(0, 1*np.pi, N_data)
y_data = f_true(x_data)
y_data_noise = y_data + 0.05*np.random.randn(N_data)


'''
Example 1: Regression with 2 feature functions.
'''
print("\nExample 1:")
def features_2(x):
    return np.array([
        np.cos(x),
        np.exp(x),
        ]).T

# How does the linear system that we solve during lineal regression look like?
A = 2/N_data * features_2(x_data).T.dot(features_2(x_data))
b = 2/N_data * features_2(x_data).T.dot(y_data_noise)
w = np.linalg.solve(A,b)
print("System of linear equations: ")
print("Left hand side = \n" + str(A))
print("Right hand side = \n" + str(b))

# First, we apply ordinary regression to calibrate the parameters w. This is equivalent to sparse regression with alpha=0.
reg_2 = LinearRegression(fit_intercept=False).fit(features_2(x_data), y_data_noise)
print("Regression with 2 feature functions.")
print("w = " + str(reg_2.coef_))

# Next, we apply sparse regression.
spr_2 = Lasso(alpha=1e-2, fit_intercept=False).fit(features_2(x_data), y_data_noise)
# Note that Lasso stands for least absolute shrinkage and selection operator. This is another name for sparse regression with p=1.
print("Sparse regression with 2 feature functions.")
print("w = " + str(spr_2.coef_))

# Let's plot the results.
plt.plot(x_data,y_data_noise,"x",label="Data")
plt.plot(x_data,features_2(x_data).dot(reg_2.coef_),"-",label="Regression")
plt.plot(x_data,features_2(x_data).dot(spr_2.coef_),"r--",label="Sparse Regression")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(loc="upper right")


'''
Example 2: Regression with 6 feature functions.
'''
print("\nExample 2:")
def features_6(x):
    return np.array([
        np.cos(x),
        np.exp(x),
        np.sin(x),
        x,
        x**2,
        x**3,
        ]).T

# First, we apply ordinary regression to calibrate the parameters w. This is equivalent to sparse regression with alpha=0.
reg_6 = LinearRegression(fit_intercept=False).fit(features_6(x_data), y_data_noise)
print("Regression with 6 feature functions.")
print("Regression: w = " + str(reg_6.coef_))

# Next, we apply sparse regression.
spr_6 = Lasso(alpha=1e-2, fit_intercept=False).fit(features_6(x_data), y_data_noise)
print("Sparse regression with 6 feature functions.")
print("w = " + str(spr_6.coef_))












