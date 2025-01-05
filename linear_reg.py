import copy, math
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('./deeplearning.mplstyle')
np.set_printoptions(precision=2)
dlblue = '#0096ff'; dlorange = '#FF9300'; dldarkred='#C00000'; dlmagenta='#FF40FF'; dlpurple='#7030A0';
dlcolors = [dlblue, dlorange, dldarkred, dlmagenta, dlpurple]
dlc = dict(dlblue = '#0096ff', dlorange = '#FF9300', dldarkred='#C00000', dlmagenta='#FF40FF', dlpurple='#7030A0')


##########################################################
# Regression Routines
##########################################################


# Function to predict the house price
def predict(x, w, b):
    """
    single predict using linear regression
    Args:
      x (ndarray): Shape (n,) example with multiple features
      w (ndarray): Shape (n,) model parameters   
      b (scalar):             model parameter 
      
    Returns:
      p (scalar):  prediction
    """
    p = np.dot(x, w) + b
    return p

# Compute the cost function
def compute_cost(X, y, w, b):
    """
    compute cost
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      cost (scalar): cost
    """
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb_i = np.dot(w, X[i]) + b
        cost += (f_wb_i - y[i]) ** 2
    cost /= (2 * m)
    return cost

#Function to calculate the cost faster using matrix multiplication
def compute_cost_matrix(X, y, w, b, verbose=False):
    """
    Computes the gradient for linear regression
     Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      verbose : (Boolean) If true, print out intermediate value f_wb
    Returns
      cost: (scalar)
    """
    m = X.shape[0]

    # calculate f_wb for all examples.
    f_wb = X @ w + b
    # calculate cost
    total_cost = (1/(2*m)) * np.sum((f_wb-y)**2)

    if verbose: print("f_wb:")
    if verbose: print(f_wb)

    return total_cost

# Extraxt the derivaive term w.r.t w and b
def compute_gradient(X, y, w, b):
    """
    Computes the gradient for linear regression 
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b. 
    """
    m, n = X.shape
    dj_dw = np.zeros(n)
    dj_db = 0
    for i in range(m):
        err = (np.dot(X[i], w) + b) - y[i]
        for j in range(n):
            dj_dw[j] += (err * X[i, j])
        dj_db += err
    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db

# Compute gradient faster using matrix multiplication
def compute_gradient_matrix(X, y, w, b):
    """
    Computes the gradient for linear regression

    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
    Returns
      dj_dw (ndarray (n,1)): The gradient of the cost w.r.t. the parameters w.
      dj_db (scalar):        The gradient of the cost w.r.t. the parameter b.

    """
    m,n = X.shape
    f_wb = X @ w + b
    e   = f_wb - y
    dj_dw  = (1/m) * (X.T @ e)
    dj_db  = (1/m) * np.sum(e)

    return dj_dw,dj_db


def gradient_descent(X, y, w, b, alpha, iter):
    """
    Performs batch gradient descent to learn w and b. Updates w and b by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      X (ndarray (m,n))   : Data, m examples with n features
      y (ndarray (m,))    : target values
      w_in (ndarray (n,)) : initial model parameters  
      b_in (scalar)       : initial model parameter
      cost_function       : function to compute cost
      gradient_function   : function to compute the gradient
      alpha (float)       : Learning rate
      num_iters (int)     : number of iterations to run gradient descent
      
    Returns:
      w (ndarray (n,)) : Updated values of parameters 
      b (scalar)       : Updated value of parameter 
      """
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []

    for i in range(iter):
 
        # Calculate the gradient and update the parameters
        dj_dw, dj_db = compute_gradient_matrix(X, y, w, b)

        # Update Parameters using w, b, alpha and gradient
        w = w - (alpha * dj_dw)
        b = b - (alpha * dj_db)

        # Save cost J at each iteration
        if i < 100000:      #prevent resource exhaustion
            J_history.append(compute_cost_matrix(X, y, w, b))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(iter / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}     ")

    return w, b, J_history #return final w,b and J history for graphing

# Load the dataset
def load_house_data():
    data = np.loadtxt("houses.txt", delimiter=',', skiprows=0)
    X = data[:,:4]
    y = data[:,4]
    return X, y


def run_gradient_descent(X, y, iterations = 1000, alpha = 1e-6):
    # initializing the model parameter
    b_init = 0.
    w_init = np.zeros(X.shape[1])
    # run gradient descent
    w_final, b_final, J_hist = gradient_descent(X, y, w_init, b_init, alpha, iterations)
    print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")

    return w_final, b_final, J_hist

##########################
# Code testing
##########################

# Load the dataset
"""
X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])
"""

# data is stored in numpy array/matrix
"""
print(f"X Shape: {X_train.shape}, X Type:{type(X_train)})")
print(X_train)
print(f"y Shape: {y_train.shape}, y Type:{type(y_train)})")
print(y_train)
"""

# initializing the model parameter
"""
b_init = 0.
w_init = np.zeros(X_train.shape[1])
"""

# some gradient descent settings
"""
iter = 1000
alpha = 5.0e-7
"""

# run gradient descent 
"""
w_final, b_final, J_hist = gradient_descent(X_train, y_train, w_init, b_init, alpha, iter)
print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")

m,_ = X_train.shape
for i in range(m):
    prediction = predict(X_train[i], w_final, b_final)
    print(f"prediction: {prediction}, target value: {y_train[i]}")
"""

# plot cost versus iteration  

"""
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_hist)
ax2.plot(100 + np.arange(len(J_hist[100:])), J_hist[100:])
ax1.set_title("Cost vs. iteration");  ax2.set_title("Cost vs. iteration (tail)")
ax1.set_ylabel('Cost')             ;  ax2.set_ylabel('Cost') 
ax1.set_xlabel('iteration step')   ;  ax2.set_xlabel('iteration step') 
plt.show()
"""

#x_vec = X_train[0, :]
#f_wb = predict(x_vec, w_init, b_init)
#print(f"The prediction: {f_wb}")

# Compute and display cost using our pre-chosen optimal parameters. 
#cost = compute_cost(X_train, y_train, w_init, b_init)
#print(f'Cost at optimal w : {cost}')

#Compute and display gradient
#tmp_dj_dw, tmp_dj_db = compute_gradient(X_train, y_train, w_init, b_init)
#print(f'dj_db at initial w,b: {tmp_dj_db}')
#print(f'dj_dw at initial w,b: \n {tmp_dj_dw}')