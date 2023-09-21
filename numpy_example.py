import numpy as np
data = np.array([1,2,3,4,5])
weights = np.random.rand(5,1)

print (data.shape)
print (data.dtype)

print (weights.shape)
print (weights.dtype)

# Array Indexing

print(weights.T)
print(weights [0]) #1st index
print(weights [1]) #2nd Index
print(weights [1:3]) #2nd and 3rd index
print(weights [-1]) #last index

def weight_summary(weight_arry):
    for i in range (len(weight_arry)):
        print("Weight-Index-", i , ":" , weight_arry[i])

weight_summary(weights)

#Array Operations

noise_magnetude = .1 * np.mean(data) #Standard deviation based on data
noise = np.random.normal(0, noise_magnetude, data.shape) #Generate Gaussian Noise with the same shape as the data
noisy_data = data + noise_magnetude*noise

#Advanced Indexing 

negative_noise_mask = data > noisy_data
negative_data = data[negative_noise_mask]
neg_magnitude = np.linalg.norm(negative_data, ord=2)

print("Negative Noise Condition:", negative_noise_mask)
print("Data with negative Noise:", negative_data)
print(f"Magnetude of Negative Nosie:, {neg_magnitude}") #f makes it so you can put non string variables inside of the print quotes, using c-brackets

#Use weights to get a weighted sum of ingle input of data

weights_transposed = weights.T
output = np.dot(weights_transposed, data)

print("Data Shape:", data.shape)
print("Transposed Weights Shape:", weights_transposed.shape)
print("Output Shape:", output.shape)

#Expand dimension
data = data.reshape(data.shape[0], 1)
output = np.dot(weights_transposed, data)

print("New Data Shape:", data.shape)
print("Output Shape:", output.shape)

#Batch operation
data = np.random.rand(data.shape[0], 100)
output = np.dot(weights_transposed, data)

print("Batch Data Shape:", data.shape)
print("Batch Output Shape:", output.shape)

#Linear algebra 
#Caclulate the covariance matrix on centered data
data_centered = data - np.mean(data, axis=0)
cov_matrix = np.cov(data_centered, rowvar=False)

#Compute Eigenvalues and Eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

#Sort the eigenvectors and corresponding eigenvalues
sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_indices]
sorted_eigenvectors = eigenvectors[:, sorted_indices]

#Select the top k eigenvectors (e.g. k=2)
k=2
top_k_eigenvectors = sorted_eigenvectors[:, :k]

#Transform the original data
principal_components = np.dot(data_centered, top_k_eigenvectors)

print("Principal Components Shape:", principal_components.shape)



