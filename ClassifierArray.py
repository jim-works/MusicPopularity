# -*- coding: utf-8 -*-
from sklearn.neural_network import MLPClassifier
import numpy as np
import sys

class ClassifierArray:
    models = []
    director_model = None
    def __init__(self, director_hidden_sizes = (20,10), hidden_sizes = (20,10), random=1, count=10, max_iterations=10, warm=True):
        self.models = []
        self.director_model = MLPClassifier(random_state=random, hidden_layer_sizes=director_hidden_sizes, max_iter=max_iterations, warm_start=warm)
        for i in range(count):
            self.models.append(MLPClassifier(random_state=random, hidden_layer_sizes=hidden_sizes, max_iter=max_iterations, warm_start=warm))
    
    def predict(self, X):
        results = []
        predictions= []
        cases = X.shape[0]
        """program = self.director_model.predict(X)
        for i in range(program.shape[0]):
            predictions.append(self.models[int(program[i])].predict(X[i,:].reshape(1,-1)))
        """
        for model in self.models:
            model_result = model.predict_proba(X)
            results.append(model_result)
        for i in range(cases):
            max_p = -2000000000.0
            max_class = -1
            for pred in results:
                c = np.argmax(pred[i,:])
                p = pred[i,c]
                if p > max_p:
                    max_class = c
                    max_p = p
            predictions.append(max_class) 
        return np.array(predictions)
    
    def fit(self, X_trn, X_trn_list, y_trn_list):
        if len(X_trn_list) != len(self.models) or len(y_trn_list) != len(self.models):
            tb = sys.exec_info()[2]
            raise RuntimeError("len(X_trn_list) %d or len(y_trn_list) %d doesn't match model count %d" % (len(X_trn_list),len(y_trn_list),len(self.models))).with_traceback(tb)
        y_trn_director = []
        for i in range(len(X_trn_list)):
            self.models[i].fit(X_trn_list[i],y_trn_list[i])
            #y_trn_director.append(np.ones(len(X_trn_list[i]))*i)
        #y_dir = np.concatenate(y_trn_director).ravel()
        #print(y_dir.shape)
        #print(y_dir)
        #self.director_model.fit(X_trn,y_dir)
            
            
            

