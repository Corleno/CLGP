#!/user/bin/env python3
'''
Create , 2018

@author: meng2
'''

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import pickle
import time
import matplotlib.pyplot as plt
import argparse
import logging
import os
import seaborn as sns
from scipy.optimize import minimize


class CLGP_R():
    """
    Variation Inference for temporal categorical latent gaussian process model
    """

    def __init__(self, M=20, Q=2, MC_T=5, error=1e-4, reg=0, init_loc=None, init_inducing=None):
        self.M=M
        self.Q=Q
        self.MC_T=MC_T
        self.error=error
        self.reg=reg
        self.init_loc=init_loc
        self.init_inducing=init_inducing

    # Define variables
    def _L2Ld(self, L):
        idx = np.zeros([self.M, self.M], dtype=np.int32)
        mask = np.zeros([self.M, self.M], dtype=np.bool)
        triu_idx = np.triu_indices(self.M)
        idx[triu_idx] = np.arange((self.M*(self.M+1)/2))
        mask[triu_idx] = True
        Ld = tf.where(mask, tf.gather(L, idx), tf.zeros([self.M, self.M], dtype=L.dtype))
        return tf.transpose(Ld)

    def _Cov_mat(self, theta, X1, X2 = None):
        # theta = (alpha, Lambda)
        sigmaf2= theta[0]
        _X2 = X1 if X2 is None else X2
        if len(X1.shape) == 1:
            X1 = tf.reshape(X1, [-1, 1])
        if len(_X2.shape) == 1:
            _X2 = tf.reshape(_X2, [-1, 1])
        l = theta[(theta.shape[0]-X1.shape[1]):]
        dist = tf.matmul(tf.reshape(tf.reduce_sum((X1/l)**2,1), [-1,1]), tf.reshape(tf.ones(_X2.shape[0]), [1,-1])) + tf.matmul(tf.reshape(tf.ones(X1.shape[0]), [-1,1]), tf.reshape(tf.reduce_sum((_X2/l)**2,1), [1,-1])) - 2*tf.matmul((X1/l), tf.transpose(_X2/l))
        cov_mat = sigmaf2 * tf.exp(-dist/2.0)
        if X2 is None:
            cov_mat += np.diag(np.ones(X1.shape[0]))*self.error
        return cov_mat

    def _tf_cov(self, x):
        mean_x = tf.reduce_mean(x, axis=0, keepdims=True)
        mx = tf.matmul(tf.transpose(mean_x), mean_x)
        vx = tf.matmul(tf.transpose(x), x)/tf.cast(tf.shape(x)[0], tf.float32)
        cov_xx = vx - mx
        cov_xx += np.diag(np.ones(cov_xx.shape[0]))*self.error
        return cov_xx

    def _init_L(self, scale = 1):
        mat = np.diag(np.ones(self.M))*scale
        vec = np.concatenate([mat[i, i:] for i in range(self.M)])
        res = np.array([vec for i in range(self.D)])
        return res

    def _create_graph(self):
        with tf.variable_scope('network_build') as net_build:
            self.lamb = tf.placeholder(tf.float32, name = 'lamb')
            self.indx = tf.placeholder(tf.int32, name = 'indx')
            self.init_tilde_X = tf.placeholder(tf.float32, shape=(self.N_test, self.Q))

            self.ltheta = tf.Variable(np.log(np.ones([self.D, self.Q + 1])*0.1), dtype=tf.float32, name="ltheta")
            # self.ltheta = tf.Variable(np.random.randn(self.D, self.Q + 1), dtype=tf.float32, name="ltheta")
            self.Z = tf.Variable(self.init_inducing, dtype=tf.float32, name="Z")
            self.mu = tf.Variable(np.random.randn(self.M, self.D, self.K_max)*0.01, dtype=tf.float32, name="mu")
            self.L = tf.Variable(self._init_L(0.01), dtype=tf.float32, name="L")
            self.m = tf.Variable(self.init_loc, dtype=tf.float32, name="m")
            self.log_s = tf.Variable(np.log(np.ones([self.N, self.Q])*0.01), dtype=tf.float32, name="s")
            self.log_hyper_s = tf.Variable(0, dtype = tf.float32, name="hyper_log_s")
            # self.log_s = tf.Variable(np.random.randn(self.N, self.Q), dtype=tf.float32, name="s")                
            self.tilde_X = tf.Variable(np.zeros([self.N_test, self.Q]), dtype=tf.float32, name="tilde_X")
            self.assign_rs_tilde_X = self.tilde_X.assign(self.init_tilde_X)

            self.theta = tf.exp(self.ltheta)
            self.s = tf.exp(self.log_s) 
            self.hyper_s = tf.exp(self.log_hyper_s) ### scale parameter which is a std
            # Estimation for X and U
            self.hat_X = self.m
            self.hat_U = self.mu

            # summarize m
            tf.summary.histogram("summary/m_0", self.m[:,0])
            tf.summary.histogram("summary/m_1", self.m[:,1])
            # summarize Z
            tf.summary.histogram("summary/Z_0", self.Z[:,0])
            tf.summary.histogram("summart/Z_1", self.Z[:,1])

            # Fix hyper-parameters of GP across time.
            # self.phi = tf.constant(np.tile([[1,0.1]],[self.Q,1]), dtype=tf.float32)

        tf.set_random_seed(1234)
        tfd = tfp.distributions
        
        with tf.name_scope('Dist_X'):
            # variational distribution of X independent Gaussian with dimension Q by N by T_max.
            QX_loc = [[self.m[n, q] for q in range(self.Q)]for n in range(self.N)]
            QX_scale = [[self.s[n, q] for q in range(self.Q)] for n in range(self.N)]
            QX = tfd.Independent(distribution=tfd.Normal(loc = QX_loc, scale = QX_scale), reinterpreted_batch_ndims=0, name = "QX")
            # prior distribution of X from Gauusian process prior with dimentsion Q by N by T
            PX_loc = [[0. for q in range(self.Q)] for n in range(self.N)]
            PX_scale = [[self.hyper_s for q in range(self.Q)] for n in range(self.N)]
            PX = tfd.Independent(distribution=tfd.Normal(loc = PX_loc, scale = PX_scale), reinterpreted_batch_ndims=0, name = "PX")

        Ld_list = []
        cov_mat_list = []
        for d in range(self.D):
            L_d = self.L[d,:]
            Ld_list.append(self._L2Ld(L_d))
            cov_mat_list.append(self._Cov_mat(self.theta[d,:], self.Z))

        with tf.name_scope('Dist_U'):
            QU_loc = [[self.mu[:, d, k] for k in range(self.K_max)] for d in range(self.D)]
            eyes = tf.eye(self.M)
            QU_scale_tril = [[Ld_list[d] if k < self.K[d] else eyes for k in range(self.K_max)] for d in range(self.D)]
            QU = tfd.Independent(distribution = tfd.MultivariateNormalTriL(loc = QU_loc, scale_tril=QU_scale_tril), reinterpreted_batch_ndims=0, name = "QU")
            PU_loc = [[tf.zeros(self.M) for k in range(self.K_max)] for d in range(self.D)]
            PU_covariance_matrix = [[cov_mat_list[d] if k < self.K[d] else eyes for k in range(self.K_max)] for d in range(self.D)]
            PU = tfd.Independent(distribution= tfd.MultivariateNormalFullCovariance(loc = PU_loc, covariance_matrix= PU_covariance_matrix), reinterpreted_batch_ndims=0, name = "PU")

        with tf.name_scope('KL_X'):
            self.KL_X = tf.reduce_sum(tf.distributions.kl_divergence(QX, PX, name = 'KL_X'))
            tf.summary.scalar("summary/KL_X", self.KL_X)

        with tf.name_scope('KL_U'):
            KL_U_mat = tf.distributions.kl_divergence(QU, PU, name = 'KL_U_mat')
            indx = [[d, k] for d in range(self.D) for k in range(self.K[d])]
            self.KL_U = tf.reduce_sum(tf.gather_nd(KL_U_mat, indx), name = 'KL_U')
            tf.summary.scalar("summary/KL_U", self.KL_U)

        with tf.name_scope('KL_ZX'):
            # estimate distribution of all Zs using gaussian distribution
            Q_Z_loc = tf.reduce_mean(self.Z, axis=0)
            Q_Z_cov = self._tf_cov(self.Z)
            Q_Z = tfd.MultivariateNormalFullCovariance(loc=Q_Z_loc, covariance_matrix=Q_Z_cov, name='ED_Z')
            # estimate distribution of all Xs for each time using gaussian distribution
            Q_X_loc = tf.reduce_mean(self.m, axis=0)
            Q_X_cov = self._tf_cov(self.m)
            Q_X = tfd.MultivariateNormalFullCovariance(loc=Q_X_loc, covariance_matrix=Q_X_cov, name='ED_X')
            # compute the KL divergence between X and Z
            self.KL_ZX = tf.distributions.kl_divergence(Q_Z, Q_X, name = 'KL_ZX')
            tf.summary.scalar("summary/KL_ZX", self.KL_ZX)

        with tf.name_scope('Comp_F'):
            self.Comp_F = tf.constant(0, dtype=tf.float32, name="Comp_F")
            for tt in range(self.MC_T):
                Comp_F_tt = 0

                ### Sample X
                sampled_eps_X = tf.random_normal([self.N, self.Q])
                self.sampled_X = self.m + tf.multiply(self.s, sampled_eps_X)
                print ("QX has been sampled.") # N by Q

                ### Sample U
                sampled_U = []
                Sigma_U_list = []
                U_noise_list = []
                for d in range(self.D):
                    sampled_eps_U_d = tf.random_normal([self.M, self.K[d]])
                    sampled_U_d = self.mu[:, d, :self.K[d]] + tf.matmul(Ld_list[d], sampled_eps_U_d)
                    paddings = np.array([[0, 0], [0, self.K_max-self.K[d]]])
                    sampled_U.append(tf.pad(sampled_U_d, paddings))
                    Sigma_U_list.append(tf.matmul(Ld_list[d], tf.transpose(Ld_list[d])))
                    U_noise_list.append(tf.matmul(Ld_list[d], sampled_eps_U_d))
                self.sampled_U = tf.stack(sampled_U)
                self.Sigma_U_list = Sigma_U_list
                self.U_noise_list = U_noise_list
                print ("QU has been sampled.") # D by M by K_max
                

                ### Sample F and compute log-likelihood
                sampled_f =[]
                for d in range(self.D):
                    Inv_cov_d_MM = tf.matrix_inverse(self._Cov_mat(self.theta[d,:], self.Z))
                    cov_NM = self._Cov_mat(self.theta[d,:], self.sampled_X, self.Z)
                    A_d = tf.matmul(cov_NM, Inv_cov_d_MM) #shape: Sum_T by M
                    B_d = tf.reshape(tf.reduce_sum(tf.multiply(A_d, cov_NM), axis=1), [-1]) #shape Sum_T
                    b_d = (self.theta[d,0])*tf.constant(np.ones(B_d.shape[0]), dtype=np.float32) - B_d
                    zeros = tf.zeros_like(b_d)
                    masked = b_d > 0
                    b_d = tf.where(masked, b_d, zeros)

                    sampled_eps_f_d = tf.random_normal([self.sampled_X.shape.as_list()[0], self.K[d]])
                    sampled_f_d = tf.matmul(A_d, self.sampled_U[d,:,:self.K[d]]) + tf.multiply(tf.tile(tf.reshape(tf.sqrt(b_d), [-1,1]), [1,self.K[d]]), sampled_eps_f_d)
                    y_d_indx = np.stack([np.linspace(0, self.y.shape[0]-1, self.y.shape[0]), self.y[:,d]], axis = 1).astype(np.int32)
                    
                    # Make softmax more numerically stable
                    # self.sampled_f_d = sampled_f_d - tf.reduce_max(sampled_f_d, axis=-1, keepdims=True)
                    # self.sampled_f_d = sampled_f_d
                    
                    Comp_F_tt += tf.reduce_sum(tf.log(tf.gather_nd(tf.nn.softmax(sampled_f_d), y_d_indx)))
                    paddings = np.array([[0,0], [0,self.K_max-self.K[d]]])
                    sampled_f.append(tf.pad(sampled_f_d, paddings))
                self.sampled_f = tf.stack(sampled_f)

                self.Comp_F += Comp_F_tt
                print ("QF has been sampled.")
                print ("{}th Monte Carlo iteration has been completed.".format(tt))
            self.Comp_F = self.Comp_F/self.MC_T
            tf.summary.scalar("summary/Comp_F", self.Comp_F)

    def _create_lp(self):
        self.ell_train, self.lp_train, _ = self._Prediction_train() 
        self.ell_test, self.lp_test, self.ell_test_vec = self._Prediction_test(train=True, n_incomplete=args.n_incomplete)
        self.ell_test_pred, self.lp_test_pred = self._Prediction_test(train=False, n_incomplete=args.n_incomplete)


    def _create_loss_optimizer(self):
        with tf.name_scope('elbo'):
            if self.reg == 0:
                self.elbo = - self.lamb*(self.KL_X + self.KL_U) + self.Comp_F
            else:
                self.elbo = - self.lamb*(self.KL_X + self.KL_U + self.reg*self.KL_ZX) + self.Comp_F
            # self.elbo = - self.lamb*(self.KL_X + self.KL_U) + self.Comp_F
            tf.summary.scalar("summary/ELBO", self.elbo)
            negelbo = -self.elbo
            nell_test = -self.ell_test
            nell_test_indx = -self.ell_test_vec[self.indx]

        with tf.variable_scope('network_train') as net_train:
            with tf.name_scope('train'):
                if self.method=="Adam":
                    opt = tf.train.AdamOptimizer(self.learning_rate_train)
                if self.method=='RSM':
                    opt = tf.train.RMSPropOptimizer(self.learning_rate_train)
                if self.method=='Adagrad':
                    opt = tf.train.AdagradOptimizer(self.learning_rate_train)
                if self.method_test=="Adam":
                    opt_test = tf.train.AdamOptimizer(self.learning_rate_test)
                if self.method_test=='RSM':
                    opt_test = tf.train.RMSPropOptimizer(self.learning_rate_test)
                if self.method_test=='Adagrad':
                    opt_test = tf.train.AdagradOptimizer(self.learning_rate_test)

                self.train = opt.minimize(negelbo)
                self.train_latent = opt.minimize(negelbo, var_list=[self.mu, self.L])
                self.train_hyper = opt.minimize(negelbo, var_list=[self.ltheta, self.Z, self.m, self.log_s])
                self.test = opt_test.minimize(nell_test, var_list=[self.tilde_X])
                self.test_indx = opt_test.minimize(nell_test_indx, var_list=[self.tilde_X])

    def _create_summary(self):
        self.summ =  tf.summary.merge_all()

    ### Helper functions    
    def _softmax(self, x, axis = -1):
        """Compute softmax values for a tensor x"""
        x = x - tf.reduce_max(x, axis = axis, keepdims=True)
        e_x = tf.exp(x)
        return e_x / tf.reduce_sum(e_x, axis = axis, keepdims=True)

    def _Compute_ell_lp(self, X, Y, train=True, n_incomplete=0):
        N, D = Y.shape
        ell_mat = []
        ell = 0
        if train:
            for d in range(D-n_incomplete):
                Inv_cov_d_MM = tf.matrix_inverse(self._Cov_mat(self.theta[d,:], self.Z))
                cov_NM = self._Cov_mat(self.theta[d,:], X, self.Z)
                V_d = tf.matmul(cov_NM, Inv_cov_d_MM) # shape: N by M
                est_f_d = tf.matmul(V_d, self.hat_U[:, d, :self.K[d]]) # shape N by K[d]
                y_d_indx = np.stack([np.linspace(0, Y.shape[0]-1, Y.shape[0]), Y[:,d]], axis = 1).astype(np.int32)            
                ell_mat.append(tf.log(tf.gather_nd(tf.nn.softmax(est_f_d), y_d_indx)))
                # paddings = np.array([[0,0], [0,self.K_max-self.K[d]]])
                # est_f.append(tf.pad(est_f_d, paddings))
            ell_mat = tf.stack(ell_mat, axis=1) # shape N by D
            # print("ell_mat", ell_mat)
            ell_vec = tf.reduce_sum(ell_mat, axis=1)
            ell = tf.reduce_sum(ell_mat)
            lp = ell/(N*(D-n_incomplete))
            return ell, lp, ell_vec
        else: 
            for d in range(D-n_incomplete, D):
                Inv_cov_d_MM = tf.matrix_inverse(self._Cov_mat(self.theta[d,:], self.Z))
                cov_NM = self._Cov_mat(self.theta[d,:], X, self.Z)
                V_d = tf.matmul(cov_NM, Inv_cov_d_MM) # shape: N by M
                est_f_d = tf.matmul(V_d, self.hat_U[:, d, :self.K[d]]) # shape N by K[d]
                y_d_indx = np.stack([np.linspace(0, Y.shape[0]-1, Y.shape[0]), Y[:,d]], axis = 1).astype(np.int32)            
                ell += tf.reduce_sum(tf.log(tf.gather_nd(tf.nn.softmax(est_f_d), y_d_indx)))
                # paddings = np.array([[0,0], [0,self.K_max-self.K[d]]])
                # est_f.append(tf.pad(est_f_d, paddings))
            lp = ell/(N*n_incomplete)
            return ell, lp

    def _Prediction_train(self):
        # Compute the training log likelihood and testing log perplexity
        return self._Compute_ell_lp(self.hat_X, y_train)

    def _Prediction_test(self, train=True, n_incomplete = 0):
        # Compute the testing log likelihood and testing log perplexity
        return self._Compute_ell_lp(self.tilde_X, y_test, train=train, n_incomplete=n_incomplete)

    def Create_graph(self, Y_train, Y_test, learning_rate_train=0.01, learning_rate_test=0.01, model_path='./model/model_tem.ckpt', hist_path='./logs/tem', method='Adam', method_test='RSM'):
        logging.info("Start to create graph!")
        self.learning_rate_train=learning_rate_train
        self.learning_rate_test=learning_rate_test 
        self.method=method
        self.method_test=method_test
        self.N, self.D = Y_train.shape
        self.N_test, self.D_test = Y_test.shape
        self.y = Y_train

        var_levels = np.zeros(self.D, dtype=np.int)
        for d in range(self.D):
            var_levels[d] = len(np.unique(self.y[:, d].reshape([-1])))
        self.K = var_levels
        self.K_max = max(self.K)

        # create graph
        self._create_graph()
        # create lp
        self._create_lp()
        # create optimizer
        self._create_loss_optimizer()
        # create summary
        self._create_summary()

        # Initializing the tensor flow variables
        self.init = tf.global_variables_initializer()


    def Fit(self, Y_train, display_step=5, lamb=0.001, lamb_inc=0.001, training_epochs=2000, model_path='./model/model_tem.ckpt', hist_path='./logs/tem', verbose=False):
        # create folder for training figures
        try:
            os.makedirs('train_figs')
        except:
            pass

        self.training_epochs=training_epochs

        # Launch the session
        self.sess = tf.InteractiveSession()
        self.sess.run(self.init)

        # create a history saver
        writer = tf.summary.FileWriter(hist_path, self.sess.graph)
        # cretate a model saver
        saver = tf.train.Saver()

        #### Training
        elbo_hist=[]
        lp_train_hist = []
        # Training with KL annealing
        # print(self.sess.run([self.Z, self.theta, self.KL_U, self.KL_X, self.KL_ZX, self.Comp_F, self.elbo, self.sampled_f_d], feed_dict={self.lamb: lamb}))

        logging.info("Start to train!")
        for epoch in range(self.training_epochs):
            # self.sess.run(self.train, feed_dict={self.lamb: lamb})
            self.sess.run(self.train_latent, feed_dict={self.lamb: lamb})
            self.sess.run(self.train_hyper, feed_dict={self.lamb: lamb})
            if epoch % display_step == 0:
                # print(self.sess.run([self.Z, self.theta]))
                # print training information
                self.est_lamb, self.est_elbo, self.est_theta, self.est_Z, self.est_m, self.est_sampled_X, self.est_mu, self.est_L, self.est_Sigma_U_list, self.est_U_noise_list, self.est_KL_U, self.est_KL_X, self.est_KL_ZX, self.est_Comp_F, self.summary = self.sess.run([self.lamb, self.elbo, self.theta, self.Z, self.m, self.sampled_X, self.mu, self.L, self.Sigma_U_list, self.U_noise_list, self.KL_U, self.KL_X, self.KL_ZX, self.Comp_F, self.summ], feed_dict={self.lamb: lamb})
                self.est_ell_train, self.est_lp_train= self.sess.run([self.ell_train, self.lp_train])
                logging.info("Epoch: {}".format(epoch+1))
                logging.info("elbo: {}, F: {}, KL_U:{}, KL_X:{}, KL_ZX:{}".format(self.est_elbo, self.est_Comp_F, self.est_KL_U, self.est_KL_X, self.est_KL_ZX))
                logging.info("lp_train: {}".format(self.est_lp_train))
                
                # plot all latent variables and inducing points
                x_vec = np.concatenate([clgp_r.est_m[:,0], clgp_r.est_Z[:,0]])
                y_vec = np.concatenate([clgp_r.est_m[:,1], clgp_r.est_Z[:,1]])
                label_vec = list(y_train_labels) + ["x" for i in range(M)]
                est_m_df = pd.DataFrame(data = {'x':x_vec, 'y':y_vec, 'label':label_vec})
                fig = sns.lmplot(data=est_m_df, x='x', y='y', hue='label', markers=[0,1,2,3,4,5,6,7,8,9,"x"], fit_reg=False, legend=True, legend_out=True)
                fig.savefig('train_figs/LS_{}_{}.png'.format(args.method, epoch))
                plt.close()

                # x_vec = np.concatenate([clgp_r.est_sampled_X[:,0], clgp_r.est_Z[:,0]])
                # y_vec = np.concatenate([clgp_r.est_sampled_X[:,1], clgp_r.est_Z[:,1]])
                # label_vec = list(y_train_labels) + ["x" for i in range(M)]
                # est_sampled_x_df = pd.DataFrame(data = {'x':x_vec, 'y':y_vec, 'label':label_vec})
                # fig = sns.lmplot(data=est_sampled_x_df, x='x', y='y', hue='label', markers=[0,1,2,3,4,5,6,7,8,9,"x"], fit_reg=False, legend=True, legend_out=True)
                # fig.savefig('train_figs/LS_{}_{}_sample.png'.format(args.method, epoch))
                # plt.close()                

            writer.add_summary(self.summary, epoch)
            lamb = min(1, lamb_inc+lamb)
            elbo_hist.append(self.est_elbo)
            lp_train_hist.append(self.est_lp_train)
            self.elbo_hist = elbo_hist
            self.lp_train_hist = lp_train_hist
            
        if verbose:
            fig=plt.figure()
            plt.plot(elbo_hist)
            plt.title('Elbo_trace')
            fig.savefig('EBLO_trace_{}.png'.format(args.method))
            plt.close()
            fig=plt.figure()
            plt.plot(lp_train_hist)
            plt.title('lp_train_trace')
            fig.savefig('lp_train_trace_{}.png'.format(args.method))
            plt.close()
            


        # Close history, saver, model saver and session
        writer.close()
        print("History has been saved under {}".format(hist_path))
        # Save model weights to disk
        saver.save(self.sess, model_path)
        print("Model has beend saved under{}".format(model_path))
        self.sess.close()

    def Test(self, Y_test, n_rs = 1, display_step=5, testing_epochs=1000, model_path='./model/model_tem.ckpt', verbose=False):
        # create folder for training figures
        try:
            os.makedirs('test_figs')
        except:
            pass
        try:
            os.makedirs('test_figs_seperate')
        except:
            pass


        self.testing_epochs=testing_epochs

        # Launch the session
        self.sess = tf.InteractiveSession()
        self.sess.run(self.init)

        # cretate a model saver
        saver = tf.train.Saver()
        # Initializing the tensor flow variables
        init = tf.global_variables_initializer()
        # Restore model
        saver.restore(self.sess, model_path)
        print("Model restored.")

        est_lp_test_pred_list = []
        est_tilde_X_list = []
        for indx in range(n_rs):
            # Random sample the testing embedding inputs
            self.sess.run(self.assign_rs_tilde_X, feed_dict={self.init_tilde_X: np.random.randn(self.N_test, self.Q)})
            self.est_tilde_X, self.est_Z = self.sess.run([self.tilde_X, self.Z])
            if verbose:
                # plot all latent variables and inducing points
                x_vec = np.concatenate([self.est_tilde_X[:,0], self.est_Z[:,0]])
                y_vec = np.concatenate([self.est_tilde_X[:,1], self.est_Z[:,1]])
                label_vec = list(y_test_labels) + ["x" for i in range(M)]
                est_tilde_df = pd.DataFrame(data = {'x':x_vec, 'y':y_vec, 'label':label_vec})
                fig = sns.lmplot(data=est_tilde_df, x='x', y='y', hue='label', markers=[0,1,2,3,4,5,6,7,8,9,"x"], fit_reg=False, legend=True, legend_out=True)
                fig.savefig('tilde_X_init_{}.png'.format(indx))
                plt.close()  

            #### Testing (Optimization together)
            ell_test_hist = []
            lp_test_pred_hist = []
            logging.info("Start to test!")

            for epoch in range(self.testing_epochs):
                self.sess.run(self.test)
                if epoch % display_step == 0:
                    # print(self.sess.run([self.Z, self.theta]))
                    # print training information
                    self.est_tilde_X, self.est_ell_test, self.est_ell_test_pred, self.est_lp_test_pred, self.est_Z = self.sess.run([self.tilde_X, self.ell_test, self.ell_test_pred, self.lp_test_pred, self.Z])
                    logging.info("Epoch: {}".format(epoch+1))
                    logging.info("lp_test_pred: {}".format(self.est_lp_test_pred))
                    
                    # # plot all latent variables and inducing points
                    # x_vec = np.concatenate([clgp_r.est_tilde_X[:,0], clgp_r.est_Z[:,0]])
                    # y_vec = np.concatenate([clgp_r.est_tilde_X[:,1], clgp_r.est_Z[:,1]])
                    # label_vec = list(y_test_labels) + ["x" for i in range(M)]
                    # est_tilde_df = pd.DataFrame(data = {'x':x_vec, 'y':y_vec, 'label':label_vec})
                    # fig = sns.lmplot(data=est_tilde_df, x='x', y='y', hue='label', markers=[0,1,2,3,4,5,6,7,8,9,"x"], fit_reg=False, legend=True, legend_out=True)
                    # fig.savefig('test_figs/LS_{}_{}_test.png'.format(args.method, epoch))
                    # plt.close()
                    
                ell_test_hist.append(self.est_ell_test)
                lp_test_pred_hist.append(self.est_lp_test_pred)
                self.lp_test_pred_hist = lp_test_pred_hist
                
            est_lp_test_pred_list.append(self.est_lp_test_pred)
            est_tilde_X_list.append(self.est_tilde_X)
            if verbose:
                # plot all latent variables and inducing points
                x_vec = np.concatenate([self.est_tilde_X[:,0], self.est_Z[:,0]])
                y_vec = np.concatenate([self.est_tilde_X[:,1], self.est_Z[:,1]])
                label_vec = list(y_test_labels) + ["x" for i in range(M)]
                est_tilde_df = pd.DataFrame(data = {'x':x_vec, 'y':y_vec, 'label':label_vec})
                fig = sns.lmplot(data=est_tilde_df, x='x', y='y', hue='label', markers=[0,1,2,3,4,5,6,7,8,9,"x"], fit_reg=False, legend=True, legend_out=True)
                fig.savefig('tilde_X_{}.png'.format(indx))
                plt.close()
                fig=plt.figure()
                plt.plot(ell_test_hist)
                plt.title('ell_test_trace')
                fig.savefig('ell_test_trace_{}.png'.format(indx))
                plt.close()
                fig=plt.figure()
                plt.plot(lp_test_pred_hist)
                plt.title('lp_test_pred_trace')
                fig.savefig('lp_test_pred_trace_{}.png'.format(indx))
                plt.close()

        max_indx = np.argmax(np.array(est_lp_test_pred_list))
        self.est_tilde_X = est_tilde_X_list[max_indx]
        # #### Testing (Optimization seperately)
        # lp_test_pred_hist = []
        # logging.info("Start to test!")
    
        # for epoch in range(self.testing_epochs):
        #     for indx in range(y_test.shape[0]):
        #         self.sess.run(self.test_indx, feed_dict={self.indx: indx})
        #     if epoch % display_step == 0:
        #         # print(self.sess.run([self.Z, self.theta]))
        #         # print training information
        #         self.est_tilde_X, self.est_ell_test_pred, self.est_lp_test_pred, self.est_Z = self.sess.run([self.tilde_X, self.ell_test_pred, self.lp_test_pred, self.Z])
        #         logging.info("Epoch: {}".format(epoch+1))
        #         logging.info("lp_test_pred: {}".format(self.est_lp_test_pred))
                
        #         # plot all latent variables and inducing points
        #         x_vec = np.concatenate([clgp_r.est_tilde_X[:,0], clgp_r.est_Z[:,0]])
        #         y_vec = np.concatenate([clgp_r.est_tilde_X[:,1], clgp_r.est_Z[:,1]])
        #         label_vec = list(y_test_labels) + ["x" for i in range(M)]
        #         est_tilde_df = pd.DataFrame(data = {'x':x_vec, 'y':y_vec, 'label':label_vec})
        #         fig = sns.lmplot(data=est_tilde_df, x='x', y='y', hue='label', markers=[0,1,2,3,4,5,6,7,8,9,"x"], fit_reg=False, legend=True, legend_out=True)
        #         fig.savefig('test_figs_seperate/LS_{}_{}_test.png'.format(args.method, epoch))
        #         plt.close()
                
        #     lp_test_pred_hist.append(self.est_lp_test_pred)
        #     self.lp_test_pred_hist = lp_test_pred_hist

        # if verbose:
        #     fig=plt.figure()
        #     plt.plot(lp_test_pred_hist)
        #     plt.title('lp_test_pred_trace')
        #     fig.savefig('lp_test_pred_trace_{}.png'.format(args.method))
        #     plt.close()

        self.sess.close()

    def Cov_mat(self, theta, X1, X2 = None):
        # theta = variance + scale parameters. parameters are all under log scale.
        sigmaf2= theta[0]
        _X2 = X1 if X2 is None else X2
        # guarantee X1 is a two-dimensional array
        if len(X1.shape) == 0:
            X1 = np.array([[X1]])
        if len(X1.shape) == 1:
            X1 = np.reshape(X1, [-1, 1])
        if len(_X2.shape) == 0:
            _X2 = np.array([[_X2]])
        if len(_X2.shape) == 1:
            _X2 = np.reshape(_X2, [-1, 1])
        #l: Automatic relevance determination
        l = theta[(theta.shape[0]-X1.shape[1]):]
        dist = np.matmul(np.reshape(np.sum((X1/l)**2,1), [-1,1]), np.reshape(np.ones(_X2.shape[0]), [1,-1])) + np.matmul(np.reshape(np.ones(X1.shape[0]), [-1,1]), np.reshape(np.sum((_X2/l)**2,1), [1,-1])) - 2*np.matmul((X1/l), np.transpose(_X2/l))
        cov_mat = sigmaf2 * np.exp(-dist/2.0)
        if X2 is None:
            # To guarantee the robustness of matrix inversion, we add epsilion on the diagonal.
            cov_mat += np.diag(np.ones(X1.shape[0]))*self.error
        return cov_mat

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--M", help="number of inducing points",type=int, default=20)
    parser.add_argument("--Q", help="dimenstion of latent variable", type=int, default=2)
    parser.add_argument("--MC_T", help="number of times of Monte Carlo Integration", type=int, default=5)
    parser.add_argument("--method", help="methods: Adam, RSM, Adagrad", type=str, default='Adam')
    parser.add_argument("--method_test", help="methods: Adam, RSM, Adagrad", type=str, default='Adam')
    parser.add_argument("--dataset", help="name of dataset", type=str, default='binaryalphadigs_small')
    parser.add_argument("--reg", help="regularization", type=float, default=-1)
    parser.add_argument("--training_epochs", help="number of training epochs", type=int, default=5000)
    parser.add_argument("--testing_epochs", help="number of testing epochs", type=int, default=250)
    parser.add_argument("--learning_rate_train", help="learning rate for training data", type=float, default=0.001)
    parser.add_argument("--learning_rate_test", help="learning rate for testing data", type=float, default=0.001)
    parser.add_argument("--lower_bound", help="lower_bound of length scale in GP across time", type=float, default=0)
    parser.add_argument("--n_incomplete", help="number of incomplete pixels for every testing data", type=int, default=20)
    parser.add_argument("--n_rs", help="number of random sampling for the embedding inputs of testing data", type=int, default=10)
    args=parser.parse_args()

    try:
        os.chdir(os.path.dirname(__file__))
    except:
        pass

    with open('../data/'+args.dataset+'.dat', 'rb') as f:
        data_raw = pickle.load(f)
    y_train, y_test, y_train_labels, y_test_labels = data_raw
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)

    with open('../data/binaryalphadigs_small_init.dat', 'rb') as f:
        init_loc, init_inducing = pickle.load(f)

    M=args.M
    Q=args.Q
    MC_T = args.MC_T
    if args.reg == -1:
        reg = args.M
    else:
        reg = args.reg
    
    # Generate directory
    os.chdir('../outcome')
    try:
        os.makedirs(args.dataset)
    except:
        pass
    os.chdir('./'+args.dataset)

    try:
        os.makedirs(args.method+"_hyper")
    except:
        pass
    os.chdir('./'+args.method+"_hyper")

    if args.reg!=0:
        try:
            os.makedirs('reg_'+str(reg))
        except:
            pass
        os.chdir('./reg_'+str(reg))

    try:
        os.makedirs('lb_'+str(args.lower_bound))
    except:
        pass
    os.chdir('./'+'lb_'+str(args.lower_bound))

    # Remove all stuffs under current directory
    # try:
    #     os.system("rm -r *")
    # except:
    #     pass
    logging.basicConfig(level=logging.DEBUG, filename='CLGP_R_{}.log'.format(args.method))

    # Train data
    np.random.seed(222)
    clgp_r = CLGP_R(M=M, Q=Q, MC_T=MC_T, reg=reg, init_loc=init_loc, init_inducing=init_inducing)
    # Create graph
    clgp_r.Create_graph(y_train, y_test, learning_rate_train=args.learning_rate_train, learning_rate_test=args.learning_rate_test)
    # Training step
    clgp_r.Fit(y_train, display_step=500, training_epochs=args.training_epochs, verbose=True)
    # Testing step
    # clgp_r.Test(y_test, n_rs=args.n_rs, display_step=50, testing_epochs=args.testing_epochs, verbose=True)

    # plot all latent variables and inducing points
    # x_vec = np.concatenate([clgp_r.est_m[:,0], clgp_r.est_Z[:,0]])
    # y_vec = np.concatenate([clgp_r.est_m[:,1], clgp_r.est_Z[:,1]])
    # label_vec = list(y_train_labels) + ["x" for i in range(M)]
    # est_m_stacked_df = pd.DataFrame(data = {'x':x_vec, 'y':y_vec, 'label':label_vec})
    # fig = sns.lmplot(data=est_m_stacked_df, x='x', y='y', hue='label', markers=[0,1,2,3,4,5,6,7,8,9,"x"], fit_reg=False, legend=True, legend_out=True)
    # fig.savefig('LS_{}.png'.format(args.method))
    # plt.close()

    # Saving parameters
    # with open("pars.pickle", "wb") as file:
    #     pickle.dump([clgp_r.elbo_hist, clgp_r.lp_train_hist, clgp_r.lp_test_hist, clgp_r.est_tilde_X, clgp_r.est_Z, clgp_r.est_m, clgp_r.est_theta], file