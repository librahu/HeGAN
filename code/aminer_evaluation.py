import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, normalized_mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


class Aminer_evaluation():
    def __init__(self):

        #load author label
        #id - label
        self.paper_label = {}
        self.sample_num = 0
        with open('../data/aminer_paper_label.dat') as infile:
            for line in infile.readlines():
                paper, label = line.strip().split('\t')[:2]
                paper = int(paper)
                label = int(label) - 1

                self.paper_label[paper] = label
                self.sample_num += 1


    def evaluate_paper_cluster(self, embedding_matrix):
        embedding_list = embedding_matrix.tolist()

        X = []
        Y = []
        for paper in self.paper_label:
            X.append(embedding_list[paper])
            Y.append(self.paper_label[paper])

        pred_Y = KMeans(6).fit(np.array(X)).predict(X)
        score = normalized_mutual_info_score(np.array(Y), pred_Y)

        return score

    def evaluate_paper_classification(self, embedding_matrix):
        embedding_list = embedding_matrix.tolist()

        X = []
        Y = []
        for paper in self.paper_label:
            X.append(embedding_list[paper])
            Y.append(self.paper_label[paper])

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
        lr = LogisticRegression()
        lr.fit(X_train, Y_train)

        Y_pred = lr.predict(X_test)
        micro_f1 = f1_score(Y_test, Y_pred, average='micro')
        macro_f1 = f1_score(Y_test, Y_pred, average='macro')
        return micro_f1, macro_f1

def str_list_to_float(self, str_list):
    return [float(item) for item in str_list]

if __name__ == '__main__':
    dblp_evaluation = DBLP_evaluation()
