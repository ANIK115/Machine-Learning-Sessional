import numpy as np


def generate_random_invertible_matrix(n):
    A = np.random.randint(100, size=(n, n))
    while np.linalg.det(A) == 0:
        A = np.random.randint(100, size=(n, n))
    return A


def eigen_decomposition(A):
    eigenvalues, eigenvectors = np.linalg.eig(A)
    return eigenvalues, eigenvectors


def matrix_reconstruction(A):
    eigenvalues, eigenvectors = eigen_decomposition(A)
    diagLambda = np.diag(eigenvalues)
    A_reconstructed = eigenvectors @ diagLambda @ np.linalg.inv(eigenvectors)
    return A_reconstructed


def main():
    n = int(input("Enter the size of the matrix: "))
    A = generate_random_invertible_matrix(n)
    print("A = ")
    print(A)
    eigenvalues, eigenvectors = eigen_decomposition(A)
    print("Eigenvalues of A: ")
    print(eigenvalues)
    print("Eigenvectors of A: ")
    print(eigenvectors)
    print("Reconstructed A: ")
    print(matrix_reconstruction(A))

    reconstruected = np.allclose(A, matrix_reconstruction(A))
    print("Reconstructed matrix is equal to the original matrix: ", reconstruected)

 
if __name__ == "__main__":
    main()