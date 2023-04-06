import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


%matplotlib qt




class LogisticRegression():
    
    def __init__(self, learning_rate=0.1, iterations = 2000):
        self.lr = learning_rate
        self.iterations = iterations
    
    def transform(self, X):
        m, n = X.shape
        X_transform = np.ones((m,n+1))
        X_transform[:,1:] = X
        
        return X_transform
    
    def standardize(self, X):
        X[:,1:] = (X[:,1:] - np.mean(X[:,1:],axis=0)) / np.std(X[:,1:],axis=0)
        
        return X
    
    def shuffle_sets(self, X, y):
        ind = np.arange(X.shape[0])
        np.random.shuffle(ind)
        X = X[ind]
        y = y[ind]
        
        return X, y
    
    def fit(self, X, y):
        r,c = X.shape
        X = self.transform(X)
        X = self.standardize(X)
        
        #l_rs = np.array([0.0001,0.01,0.1,0.2,0.3,0.7])
        #l_rs = np.array([0.03,0.1,0.2,0.3,0.5])
        l_rs = np.array([0.1])
        self.Thetas = np.zeros((c+1,3))
        X, y = self.shuffle_sets(X, y)
        ########## loop per classificator
        for cind in range(3):
            # Moved self.shuffle_sets and next four lines (y[y ==,!= cind]) to
            # loop per epochs. These lines must stay toghether. Also respecting
            # the order.
            
            ########## loop per learning rate
            for j in range(len(l_rs)):
                self.Thetas[:,cind] = 0
                theta = self.Thetas[:,cind]
                theta = theta.reshape(6,1)
                self.lr = l_rs[j]
                
                ########## mini-batch
                m_mb = 45 # optimal 50
                n_mb = r//m_mb
                n_iter = n_mb if n_mb*m_mb == r else n_mb+1
                rest = False
                if n_iter == n_mb+1: rest = True
                epochs = 5
                l = np.zeros((n_iter*epochs))
                cnt = 0
                for k in range(epochs): # loop per epoch
                    #X, y1 = self.shuffle_sets(X, y1)
                    
                    X, y = self.shuffle_sets(X, y)
                    y1 = np.copy(y)
                    y1[y1 == cind] = 4
                    y1[y1 < 4] = 0
                    y1[y1 == 4] = 1
                    
                    for i in range(n_iter): # loop per batch
                        if rest == True and i == n_iter-1:
                            lin_pred = X[i*m_mb:,:] @ theta
                            h_theta = 1/(1+np.exp(-lin_pred))
                            theta = theta + self.lr * (np.transpose(X[i*m_mb:,:]) @ (y1[i*m_mb:,:] - h_theta))
                            self.Thetas[:,cind] = theta[:,0]
                            l[cnt] = (1/(r % m_mb) * (np.transpose(y1[i*m_mb:,:]) @ np.log(h_theta) + np.transpose(1-y1[i*m_mb:,:]) @ np.log((1-h_theta))))
                            cnt += 1
                        else:
                            lin_pred = X[i*m_mb:(i+1)*m_mb,:] @ theta
                            h_theta = 1/(1+np.exp(-lin_pred))
                            theta = theta + self.lr * (np.transpose(X[i*m_mb:(i+1)*m_mb,:]) @ (y1[i*m_mb:(i+1)*m_mb,:] - h_theta))
                            self.Thetas[:,cind] = theta[:,0]
                            l[cnt] = (1/m_mb * (np.transpose(y1[i*m_mb:(i+1)*m_mb,:]) @ np.log(h_theta) + np.transpose(1-y1[i*m_mb:(i+1)*m_mb,:]) @ np.log((1-h_theta))))
                            cnt += 1
                ########## mini-batch
                
                plt.figure(1)
                x_axis = m_mb*(np.arange(n_iter*epochs)+1)
                if cind == 0:
                    plt.plot(x_axis,l,color='blue')
                elif cind == 1:
                    plt.plot(x_axis,l,color='red')
                else:
                    plt.plot(x_axis,l,color='green')
                
                #plt.scatter(x_axis,l,color='red') # np.arrange(self.iterations) -> for classic gradient descent
                plt.title('Мини-шаржни успон\n'r'$\alpha$'' = {} $m_b$ = {}'.format(self.lr,m_mb))
                plt.legend(['I класа','II класа','III класа'])
                #plt.legend(['II класа','III класа'])
                plt.xlabel('$пређено$ $примера$')
                plt.ylabel('$l($'r'$\theta$''$)$')
                plt.show()
                
            ########## loop per learning rate
        ########## loop per classificator
    
    def predict(self, X):
        X = self.transform(X)
        X = self.standardize(X)
        
        theta1 = self.Thetas[:,0]
        theta1 = theta1.reshape(6,1)
        theta2 = self.Thetas[:,1]
        theta2 = theta2.reshape(6,1)
        theta3 = self.Thetas[:,2]
        theta3 = theta3.reshape(6,1)
        
        lin_pred1 = X @ theta1
        h_theta1 = 1/(1+np.exp(-lin_pred1))
        lin_pred2 = X @ theta2
        h_theta2 = 1/(1+np.exp(-lin_pred2))
        lin_pred3 = X @ theta3
        h_theta3 = 1/(1+np.exp(-lin_pred3))
        
        h_theta = np.concatenate((h_theta1,h_theta2,h_theta3),axis=1)
        
        y_pred = np.argmax(h_theta,axis=1)
        y_pred = y_pred.reshape(X.shape[0],1)
        
        return y_pred




class GaussianNaiveBayes():
    
    # estimating statistical parameters
    def fit(self, X, y, N_classes): # need X without ones (from linear regression)
        m, n = X.shape # examples and features
        X = self.standardize(X)
        self.mean = np.zeros((N_classes,n)) # mean values for each class and each predictor
        self.var = np.zeros((N_classes,n)) # variances for each class and each predictor
        self.apriori = np.zeros((N_classes,1))
        self.N_classes = N_classes
        
        for i in range(self.N_classes):
            ind = np.argwhere(y == i)
            ind = ind[:,0]
            X1 = X[ind,:]
            #X1 = X[y == i]
            #print(X1.shape)
            self.mean[i,:] = X1.mean(axis=0) # or np.mean(X1,axis=0)
            #print(X1.mean(axis=0).shape)
            self.var[i,:] = X1.std(axis=0)
            self.apriori[i] = X1.shape[0] / m
    
    def predict(self, X):
        y_pred = []
        X = self.standardize(X)
        for x in X:
            l = []
            for i in range(self.N_classes):
                log_apriori = np.log(self.apriori[i]) # Pr(y)
                log_aposteriori = np.sum(np.log(self.pdf(i,x)))
                l_ = log_aposteriori + log_apriori
                l.append(l_[0])
            y_pred.append(np.argmax(l))
            
        return np.array(y_pred)
    
    # probability distribution fcn
    def pdf(self, class_ind, x):
        return 1/np.sqrt(2*np.pi*self.var[class_ind]) * np.exp(-(x-self.mean[class_ind])**2 / (2 * self.var[class_ind]))
    
    def standardize(self, X): # Needed!!!!!!!
        X[:,:] = (X[:,:] - np.mean(X[:,:],axis=0)) / np.std(X[:,:],axis=0)
        
        return X




########## main
if __name__ == '__main__':
    
    # reading data
    data = pd.read_csv('multiclass_data.csv',header=None)
    data1 = pd.DataFrame(data).to_numpy() # DataFrame -> np type
    
    r,c = data1.shape
    
    X = np.zeros((r,c-1))
    X[:,0:c-1] = data1[:,0:c-1]
    y = np.zeros((r,1))
    y[:,0] = data1[:,c-1]
    
    ################################################## logistic regression
    
    logreg = LogisticRegression()
    logreg.fit(X, y)
    y_pred = logreg.predict(X)
    
    total = y_pred.shape[0]
    tr1 = 0
    tr2 = 0
    tr3 = 0
    for j in range(3):
        for i in range(total):
            if y_pred[i] == j and y_pred[i] == y[i]:
                if j == 0:
                    tr1 += 1
                elif j == 1:
                    tr2 += 1
                else:
                    tr3 += 1
    tr1 = tr1/59*100
    tr2 = tr2/71*100
    tr3 = tr3/48*100
    print('##########')
    print('Логистичка регресија (по класама)')
    print('тачност1 = {:.2f} [%]'.format(tr1))
    print('тачност2 = {:.2f} [%]'.format(tr2))
    print('тачност3 = {:.2f} [%]\n'.format(tr3))
    
    tr = 0
    for i in range(total):
        if y_pred[i] == y[i]:
            tr += 1
    tr = tr/total*100
    print('Логистичка регресија')
    print('тачност = {:.2f} [%]'.format(tr))
    print('##########\n')
    
    ################################################## logistic regression
    
    
    ################################################## GNB
    
    modelGNB = GaussianNaiveBayes()
    modelGNB.fit(X, y, 3)
    y_predGNB = modelGNB.predict(X)
    
    total = y_predGNB.shape[0]
    tr = 0
    for i in range(total):
        if y_predGNB[i] == y[i]:
            tr += 1
    tr = tr/total*100
    print('##########')
    print('Гаусовски наивни Бејз')
    print('тачност = {:.2f} [%]'.format(tr))
    print('##########')
    
    ################################################## GNB


    
































    
