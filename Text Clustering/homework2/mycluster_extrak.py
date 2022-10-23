import numpy as np

def cluster_extra(T, K, num_iters = 50, epsilon = 5e-5):
    T = T.toarray()
    nd, nw = T.shape
    # print(f"(nd, nw) = {nd, nw}")
    np.random.seed(10)
    Pwz = np.random.rand(K, nw)
    Pwz = Pwz/np.expand_dims(np.sum(Pwz, axis=1), axis=1)
    Pdz = np.random.rand(K, nd)
    Pdz = Pdz/np.expand_dims(np.sum(Pdz, axis=1), axis=1)
    Pz = np.ones((K,1))
    # Pz = np.random.rand(K, 1)
    Pz = Pz/np.expand_dims(np.sum(Pz, axis=0), axis=1)
    Pz_old = np.zeros(K)
    Pzdw = np.zeros((K, nd, nw))
    iters = 0
    while ((np.linalg.norm(Pz - Pz_old) >= epsilon) and (iters < num_iters)):
        Pzdw_num = np.matmul(np.matmul(np.expand_dims(Pdz,-1), np.expand_dims(Pz,1)), np.expand_dims(Pwz, 1))
        Pzdw_den = np.sum(Pzdw_num, axis=0)
        Pzdw = Pzdw_num/Pzdw_den # k x nd x nw

        # update_num = np.tile(np.expand_dims(T, 0), (K, 1, 1))*Pzdw # k x nd x nw
        update_num  = Pzdw*np.expand_dims(T, 0)

        Pwz_num = np.sum(update_num, axis=1)
        Pwz_den = np.expand_dims(np.sum(Pwz_num, axis=-1), axis=-1)
        Pwz = Pwz_num/Pwz_den

        Pdz_num = np.sum(update_num, axis=-1)
        Pdz_den = np.expand_dims(np.sum(Pdz_num, axis=-1), axis=-1)
        Pdz = Pdz_num/Pdz_den

        Pz_num = np.expand_dims(np.sum(np.sum(update_num, axis=-1), axis=-1), axis=-1)
        Pz_den = np.expand_dims(np.sum(Pz_num, axis=0), axis=-1)
        Pz_old = Pz.copy()
        Pz = Pz_num/Pz_den
        
        iters += 1
        # print(iters, np.linalg.norm(Pz - Pz_old))

    return Pwz.T