import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, normalized_mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import math


class Yelp_evaluation():
    def __init__(self):

        #load author label
        #id - label
        self.business_label = {}
        self.sample_num = 0
        with open('../data/yelp_business_category.txt') as infile:
            for line in infile.readlines():
                business, label = line.strip().split()[:2]
                business = int(business)
                label = int(label) - 1

                self.business_label[business] = label
                self.sample_num += 1

        self.train_link_label = list()
        self.test_link_label = list()
        with open('../data/yelp_lp/yelp_ub.test_0.8_new') as infile:
            for line in infile.readlines():
                u, b, label = [int(item) for item in line.strip().split()]
                self.test_link_label.append([u, b, label])

        with open('../data/yelp_lp/yelp_ub.train_0.8_lr') as infile:
            for line in infile.readlines():
                u, b, label = [int(item) for item in line.strip().split()]
                self.train_link_label.append([u, b, label])



    def evaluate_business_cluster(self, embedding_matrix):
        embedding_list = embedding_matrix.tolist()

        X = []
        Y = []
        for business in self.business_label:
            X.append(embedding_list[business])
            Y.append(self.business_label[business])

        pred_Y = KMeans(3).fit(np.array(X)).predict(X)
        score = normalized_mutual_info_score(np.array(Y), pred_Y)

        return score

    def evaluate_business_classification(self, embedding_matrix):
        embedding_list = embedding_matrix.tolist()

        X = []
        Y = []
        for business in self.business_label:
            X.append(embedding_list[business])
            Y.append(self.business_label[business])

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
        lr = LogisticRegression()
        lr.fit(X_train, Y_train)

        Y_pred = lr.predict(X_test)
        micro_f1 = f1_score(Y_test, Y_pred, average='micro')
        macro_f1 = f1_score(Y_test, Y_pred, average='macro')
        return micro_f1, macro_f1

    def sigmoid(self, x):
        return 1.0 / (1.0 + math.exp(-x))

    def evaluation_link_prediction(self, embedding_matrix):

        #score = self.evaluate_business_cluster(embedding_matrix)
        #print 'nmi = ', score
        embedding_list = embedding_matrix.tolist()

        train_x = []
        train_y = []
        for u, b, label in self.train_link_label:
            train_x.append(embedding_list[u] + embedding_list[b])
            train_y.append(float(label))

        test_x = []
        test_y = []
        for u, b, label in self.test_link_label:
            test_x.append(embedding_list[u] + embedding_list[b])
            test_y.append(float(label))

        lr = LogisticRegression()
        lr.fit(train_x, train_y)

        pred_y = lr.predict_proba(test_x)[:,1]
        pred_label = lr.predict(test_x)

        auc = roc_auc_score(test_y, pred_y)
        f1 = f1_score(test_y, pred_label)
        acc = accuracy_score(test_y, pred_label)

        return auc, f1, acc



def str_list_to_float(self, str_list):
    return [float(item) for item in str_list]

if __name__ == '__main__':
    dblp_evaluation = DBLP_evaluation()
