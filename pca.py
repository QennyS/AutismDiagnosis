import numpy as np
from numpy import linalg as la
from sklearn.decomposition import PCA

def svd_flip(u, v, u_based_decision=False):
    if u_based_decision:
        # columns of u, rows of v
        max_abs_cols = np.argmax(np.abs(u), axis=0)
        signs = np.sign(u[max_abs_cols, range(u.shape[1])])
        u *= signs
        v *= signs[:, np.newaxis]
    else:
        # rows of v, columns of u
        max_abs_rows = np.argmax(np.abs(v), axis=1)
        signs = np.sign(v[range(v.shape[0]), max_abs_rows])
        u *= signs
        v *= signs[:, np.newaxis]
    return u, v

def pca(k):
    data = np.loadtxt('0.1_data.csv', delimiter=',')
    print(data.shape)
    m, n = np.shape(data)
    # Normalization: set mean to 0
    average = np.mean(data, axis=0)
    normalized = data - np.tile(average, (m,1))
    # Standardization: set Standard Deviation to 1
    standardData = normalized / np.std(data)
    # normData.T: np.cov treats each row of array as a separate variable
    covMat = (1/m) * standardData.T.dot(standardData)
    eigValue, eigVec = la.eig(covMat)
    eigValSorted = np.argsort(-eigValue)
    #selectVec = np.matrix(eigVec.T[:k])
    selectVec = []
    for i in range(k):
        selectVec.append(eigVec[:,eigValSorted[i]])
    selectVec = np.array(selectVec)
    finalData = np.dot(standardData, selectVec.T)
    return finalData

def pca_svd(k):
    data = np.loadtxt('0.1_data.csv', delimiter=',')
    print(data.shape)
    m, n = np.shape(data)
    label = data[:,n-1]
    data = np.delete(data, [n-2,n-1], axis=1)
    print(data.shape)
    # Normalization: set mean to 0
    average = np.mean(data, axis=0)
    normalized = data - np.tile(average, (m,1))
    normalized = normalized.transpose()
    # Standardization: set Standard Deviation to 1
    # standardData = normalized / np.std(data)
    # normData.T: np.cov treats each row of array as a separate variable
    U, S, V = la.svd(normalized, full_matrices=False)
    U, V = svd_flip(U, V)
    topU = U[:, :k]
    eigVec = topU.transpose()
    finalData = np.dot(eigVec, normalized).transpose()

    finalData = np.c_[finalData, label]
    np.savetxt('pca_data.csv',finalData ,delimiter=',')
    return finalData

def sklearn_pca():
    X = np.loadtxt('0.1_data.csv', delimiter=',')
    print(X.shape)
    m, n = np.shape(X)
    label = X[:,n-1]
    X = np.delete(X, [n-2,n-1], axis=1)
    print(X.shape)
    pca = PCA(n_components=850)
    #pca.fit(X)
    reduced_x=pca.fit_transform(X)
    reduced_x = np.c_[reduced_x, label]
    np.savetxt('850.csv',reduced_x ,delimiter=',')
    #print(np.sum(pca.explained_variance_ratio_))
    return reduced_x.shape
