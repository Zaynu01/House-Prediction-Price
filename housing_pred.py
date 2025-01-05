from linear_reg import load_house_data, run_gradient_descent, predict
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(precision=2)
plt.style.use('./deeplearning.mplstyle')

# Normalizing features
def zscore_normalize(X):
    """
    computes  X, zcore normalized by column
    
    Args:
      X (ndarray (m,n))     : input data, m examples, n features
      
    Returns:
      X_norm (ndarray (m,n)): input normalized by column
      mu (ndarray (n,))     : mean of each feature
      sigma (ndarray (n,))  : standard deviation of each feature
    """
    # finding the mean of each column/feature
    mu = np.mean(X, axis = 0)
    #finding the standard deviation of each column/feature
    sigma = np.std(X, axis = 0)
    # element-wise, subtract mu for that column from each example, divide by std for that column
    X_norm = (X - mu) / sigma

    return X_norm, mu, sigma

# load the dataset
X_train, y_train = load_house_data()
X_features = ['size(sqft)', 'bedrooms', 'floors', 'age']

# Let's view the dataset and its features by plotting each feature versus price.
fig,ax=plt.subplots(1, 4, figsize=(12, 3), sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:,i],y_train)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("Price (1000's)")
plt.show()

# normalize the original features
X_norm, X_mu, X_sigma = zscore_normalize(X_train)
print(f"X_mu = {X_mu}\nX_sigma = {X_sigma}\n\n")

w_norm, b_norm, hist = run_gradient_descent(X_norm, y_train, 2000, 1.0e-2)
print("\n\n")

#predict target using normalized features
m = X_norm.shape[0]
yp = np.zeros(m)
yp = predict(X_norm, w_norm, b_norm)
print(f"The prediction:\n {yp}\nThe target:\n {y_train}")

# First, normalize out example.
x_house = np.array([1200, 3, 1, 40])
x_house_norm = (x_house - X_mu) / X_sigma
print(x_house_norm)
x_house_predict = np.dot(x_house_norm, w_norm) + b_norm
print(f" predicted price of a house with 1200 sqft, 3 bedrooms, 1 floor, 40 years old = ${x_house_predict*1000:0.0f}")
