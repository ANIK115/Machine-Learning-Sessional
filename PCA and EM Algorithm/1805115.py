import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.stats import multivariate_normal

def gaussian(x, mean, cov):
    return multivariate_normal.pdf(x, mean=mean, cov=cov)

def expectation(data, k, pi, mu, sigma):
    num_of_rows, num_of_cols = data.shape
    probabilities = np.zeros((num_of_rows, k))
    
    for j in range(k):
        probabilities[:, j] = pi[j] * gaussian(data, mu[j, :], sigma[j, :])

    probabilities /= np.sum(probabilities, axis=1)[:, np.newaxis]
    
    return probabilities


def maximization(data, k, probabilities):
    num_of_rows, num_of_cols = data.shape
    pi = np.zeros(k)
    mu = np.zeros((k, num_of_cols))
    sigma = np.zeros((k, num_of_cols, num_of_cols))

    for j in range(k):
        nij = np.sum(probabilities[:, j])
        for i in range(num_of_rows):
            mu[j, :] += probabilities[i, j] * data[i, :]
        mu[j, :] /= nij

        for i in range(num_of_rows):
            sigma[j, :] += probabilities[i, j] * np.dot(np.transpose([data[i, :] - mu[j, :]]), [data[i, :] - mu[j, :]])
        sigma[j, :] /= nij
        pi[j] = nij / num_of_rows
    return pi, mu, sigma

def compute_log_likelihood(data, k, pi, mu, sigma):
    num_of_rows, num_of_cols = data.shape
    log_likelihood = 0

    for i in range(num_of_rows):
        temp = 0
        for j in range(k):
            temp += pi[j] * gaussian(data[i, :], mu[j, :], sigma[j, :])
        log_likelihood += np.log(temp)
    return log_likelihood

def em_gmm(data, k, num_of_trials = 5):
    #Initialize the parameters
    num_of_rows, num_of_cols = data.shape
    pi = np.random.dirichlet(np.ones(k))
    mu = np.random.rand(k, num_of_cols)
    sigma = np.array([np.eye(num_of_cols)] * k)
    log_likelihoods = []

    best_likelihood = 0
    best_params = {'pi': None, 'mu': None, 'sigma': None}

    for trial in range(num_of_trials):
        #E-Step
        probabilities = expectation(data, k, pi, mu, sigma)

        #M-Step
        pi, mu, sigma = maximization(data, k, probabilities)

        #Compute the log likelihood
        log_likelihood = compute_log_likelihood(data, k, pi, mu, sigma)
        log_likelihoods.append(log_likelihood)
        if log_likelihood > best_likelihood:
            best_likelihood = log_likelihood
            best_params['pi'] = pi
            best_params['mu'] = mu
            best_params['sigma'] = sigma

    return best_params, best_likelihood



def best_k(data):
    k_range = range(3, 9)
    log_likelihoods = []
    for k in k_range:
        params, log_likelihood = em_gmm(data, k)
        log_likelihoods.append(log_likelihood)
    
    plt.plot(k_range, log_likelihoods, marker='o')
    plt.title('Convergence Log-Likelihood vs. K')
    plt.xlabel('Number of Components (K)')
    plt.ylabel('Log-Likelihood')
    plt.savefig('log_likelihood_vs_K.png')
    plt.show()

    k_best = k_range[np.argmax(log_likelihoods)]
    best_params, best_log_likelihood = em_gmm(data, k_best)
    


def plot_data(data, title, save_path=None):
    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()
    plt.scatter(data[:, 0], data[:, 1])
    plt.title(title)
    plt.xlabel('Principal Axis 1')
    plt.ylabel('Principal Axis 2')
    if save_path:
        plt.savefig(save_path)

    plt.show()

def pca(df):
    U, S, V = np.linalg.svd(df, full_matrices=False)
    transformed_df = np.dot(df, V.T)
    return transformed_df

data_file = "2D_data_points_2.txt"
D = np.loadtxt(data_file, delimiter=',')

num_cols = D.shape[1]
columns = [f'col_{i}' for i in range(num_cols)]

#convert data to pandas dataframe
df = pd.DataFrame(data=D, columns=columns)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
transformed_df = pca(df) if num_cols > 2 else df

#remove the .txt extension from the data file
image_name = data_file[:-4]
image_name += '_pca.png'
plot_data(transformed_df, 'Transformed Data', image_name)