import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.stats import multivariate_normal

#apply seed
np.random.seed(42)

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
    #initialization
    pi = np.zeros(k)
    mu = np.zeros((k, num_of_cols))
    sigma = np.zeros((k, num_of_cols, num_of_cols))
    epsilon = 1e-6 #to avoid singular covariance matrix


    for j in range(k):
        nij = np.sum(probabilities[:, j])
        mu[j, :] = np.sum(data * probabilities[:, j, np.newaxis], axis=0) / nij

        for i in range(num_of_rows):
            diff = (data[i, :] - mu[j, :]).reshape(-1, 1)
            sigma[j] += probabilities[i, j] * np.dot(diff, diff.T)
        sigma[j] /= nij
        sigma[j] += np.eye(num_of_cols) * epsilon   #to avoid singular covariance matrix

        pi[j] = nij / num_of_rows

    return pi, mu, sigma


def compute_log_likelihood(data, k, pi, mu, sigma):
    num_of_rows, num_of_cols = data.shape
    log_likelihood = 0

    for i in range(num_of_rows):
        intermediate_sum = 0
        for j in range(k):
            intermediate_sum += pi[j] * gaussian(data[i, :], mu[j, :], sigma[j, :])
        log_likelihood += np.log(intermediate_sum)
    return log_likelihood

def em_gmm(data, k, num_of_trials=5):
    #initialization
    best_likelihood = -np.inf
    best_params = None
    all_log_likelihoods = []
    
    for trial in range(num_of_trials):
        # np.random.seed(trial)
        pi = np.random.dirichlet(np.ones(k))    #dirichlet distribution to ensure valid probabilities and unbiased initialization
        #random initial means within the range of the data
        data_range = np.max(data) - np.min(data)
        mu = np.random.rand(k, data.shape[1]) * data_range + np.min(data)
        #initial covariance matrix as identity matrix
        sigma = np.array([np.eye(data.shape[1]) for _ in range(k)])

        for _ in range(10):  # Assuming convergence within 10 iterations
            probabilities = expectation(data, k, pi, mu, sigma)
            pi, mu, sigma = maximization(data, k, probabilities)

        log_likelihood = compute_log_likelihood(data, k, pi, mu, sigma)
        if log_likelihood > best_likelihood:
            best_likelihood = log_likelihood
            best_params = (pi, mu, sigma)
        all_log_likelihoods.append(log_likelihood)
    # np.random.seed(42)
    print(best_likelihood)
    return best_params, best_likelihood, all_log_likelihoods




# Plot the best value of convergence log-likelihood against the value of K
def plot_log_likelihood(log_likelihood, save_path=None):
    plt.plot(log_likelihood, marker='o')
    plt.title('Convergence Log-Likelihood')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Log-Likelihood')
    if save_path:
        plt.savefig(save_path)
    plt.show()


    
def best_k(data, filename):
    k_range = range(3, 9)
    all_log_likelihoods = []
    all_params = []
    for k in k_range:
        params, best_log_likelihood, log_likelihoods = em_gmm(data, k)
        all_log_likelihoods.append(best_log_likelihood)
        all_params.append(params)
        # path = filename+'_log_likelihood_'+str(k)+'.png'
        # plot_log_likelihood(log_likelihoods, path)
    
    fig_name = filename+ ' log_likelihood_vs_K.png'
    plt.plot(k_range, all_log_likelihoods, marker='o')
    plt.title('Convergence Log-Likelihood vs. K')
    plt.xlabel('Number of Components (K)')
    plt.ylabel('Log-Likelihood')
    plt.savefig(fig_name)
    plt.show()

    # k_best = k_range[np.argmax(all_log_likelihoods)]
    # print('Best value of K: ', k_best)

    # params, best_log_likelihood, log_likelihoods = em_gmm(data, k_best)
    # plot_gmm(data, k_best, params, filename+'_gmm.png')

    #for each value of k, plot the estimated GMM for K by showing sample data points and Gaussian distributions in a 2D plot.
    for k in k_range:
        params = all_params[k-3]
        plot_gmm(data, k, params, filename+'_gmm_'+str(k)+'.png', filename)

    

#plot the estimated GMM for K’ by showing sample data points and Gaussian distributions in a 2D plot.   
def plot_gmm(data, k, params, save_path=None, title=None):
    num_of_rows, num_of_cols = data.shape
    trial_probabilities = np.zeros((num_of_rows, k))
    pi, mu, sigma = params
    #taking average of 10 trials
    for i in range(10):
        trial_probabilities += expectation(data, k, pi, mu, sigma)
    probabilities = trial_probabilities / np.sum(trial_probabilities, axis=1)[:, np.newaxis]
    # maximum likelihood assignment of each data point to a cluster
    cluster_assignments = np.argmax(probabilities, axis=1)

    # print(cluster_assignments.shape)
    # print(cluster_assignments[5:10])
    plt.figure(figsize=(10, 10))

    
    plt.scatter(data[:, 0], data[:, 1], c=cluster_assignments)
    plt.title('Estimated GMM for k= '+str(k)+'\n'+title)
    plt.xlabel('Principal Axis 1')
    plt.ylabel('Principal Axis 2')
    if save_path:
        plt.savefig(save_path)
    plt.show()


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
    #Project data onto the first two principal components
    transformed_df = np.dot(df, V[:2].T)
    # print(transformed_df.shape)
    return transformed_df


#read the filename from the command line
import sys
data_file = sys.argv[1]
# data_file = "3D_data_points.txt"
D = np.loadtxt(data_file, delimiter=',')

num_cols = D.shape[1]
columns = [f'col_{i}' for i in range(num_cols)]

#convert data to pandas dataframe
df = pd.DataFrame(data=D, columns=columns)

#centralize the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
transformed_df = pca(df) if num_cols > 2 else df

#remove the .txt extension from the data file
image_name = data_file[:-4]
image_name += '_pca.png'
plot_data(transformed_df, 'Transformed Data', image_name)

print(transformed_df.shape)

#reshape the data to 2D
if num_cols <= 2:
    transformed_df = transformed_df.to_numpy()

best_k(transformed_df, data_file[:-4])