import numpy as np

#Name: Manvitha Kalicheti 
#gtID: 903838438

def my_recommender(rate_mat, lr, with_reg):
    """

    :param rate_mat:
    :param lr:
    :param with_reg:
        boolean flag, set true for using regularization and false otherwise
    :return:
    """

    # TODO pick hyperparams
    max_iter = 500
    learning_rate = 2e-4
    reg_coef = 0.02
    n_user, n_item = rate_mat.shape[0], rate_mat.shape[1]

    if(with_reg == False):
        reg_coef = 0        

    U = np.random.rand(n_user, lr) / lr
    V = np.random.rand(n_item, lr) / lr

    # TODO implement your code here
    iter = 0
    thresh = 1e-4
    U_change = 1e6
    V_change = 1e6
    #Err = 1e6

    #converges when U and V stop changing
    while(iter < max_iter and (U_change > thresh or V_change > thresh)) :
        U_old = U
        V_old = V
        M_train = rate_mat > 0 #to make sure we only use the training values

        U_der = -2*(rate_mat - (U@V.T*M_train))@V + 2*reg_coef*U
        V_der = -2*(rate_mat - (U@V.T*M_train)).T@U + 2*reg_coef*V
        U = U - learning_rate*U_der
        V = V - learning_rate*V_der
    
        U_change = np.sum(np.linalg.norm(U-U_old)) 
        V_change = np.sum(np.linalg.norm(V-V_old))
        # Err = np.sum(np.power(rate_mat - U@V.T, 2)) + reg_coef*np.sum(np.power(U,2)) + np.sum(np.power(V,2))
        iter += 1

    return U, V