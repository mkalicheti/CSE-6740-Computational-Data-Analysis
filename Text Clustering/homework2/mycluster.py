import numpy as np

def cluster(T, K, num_iters = 1000, epsilon = 1e-12):
	"""

	:param bow:
		bag-of-word matrix of (num_doc, V), where V is the vocabulary size
	:param K:
		number of topics
	:return:
		idx of size (num_doc), idx should be 1, 2, 3 or 4
	"""
	n_c = 4 #number of clusters
	n_w = T.shape[1] #number of words
	n_d = T.shape[0] #number of documents 

	#initialising
	pi = np.ones(n_w)/n_w
	pi_prev = np.zeros(pi.shape)
	np.random.seed(23)
	mu = np.random.rand(n_w, n_c)

	#normalising
	for i in range(n_w):
		mu[i, :] /= np.sum(mu[i, :])
	
	gamma = np.zeros((n_d, n_c))

	iter = 0
	while(np.linalg.norm(pi - pi_prev) > epsilon and iter < num_iters):
		pi_prev = pi.copy()

		#Expectation
		num = np.zeros((n_d, n_c))
		den = np.zeros(n_d)

		for i in range(n_d):
			for c in range(n_c):
				mul_mu = 1
				for j in range(n_w):
					mul_mu *= mu[j][c]**T[i][j]
				num[i][c] = pi[c]*mul_mu

		for i in range(n_d):
			sum = 0
			for c in range(n_c):
				mul = 1
				for j in range(n_w):
					mul *= mu[j][c]**T[i][j]
				sum += pi[c]*mul
			den[i] = sum
		
		for i in range(n_d):
			for c in range(n_c):
				gamma[i][c] = num[i][c]/den[i]

		#Maximisation
		num = np.zeros((n_w, n_c))
		den = np.zeros(n_c)

		for j in range(n_w):
			for c in range(n_c):
				for i in range(n_d):
					num[j][c] += gamma[i][c]*T[i][j]
		
		for c in range(n_c):
			for i in range(n_d):
				for l in range(n_w):
					den[c] += gamma[i][c]*T[i][l]
		
		for j in range(n_w):
			for c in range(n_c):
				mu[j][c] = num[j][c]/den[c]

		for c in range(n_c):
			for i in range(n_d):
				pi[c] += gamma[i][c]
			pi[c] /= n_d

		iter += 1
	print(iter)
	idx = np.zeros(n_d)
	for i in range(n_d):
		idx[i] = np.argmax(gamma[i]) + 1

	return idx
