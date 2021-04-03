# naive bayesian for spam mail fitering
# 谭树杰 11849060
from collections import Counter
import os
import numpy as np
from scipy.special import expit
# from sklearn.metrics import accuracy_score


class Bayes:
    def __init__(self):
        self.length = -1        # length of feature
        self.vectorcount = dict()   # vectorcount[1] indicates feature vector of spam mails

    def fit(self, feature_matrix: list, labels: list):
        if len(feature_matrix) != len(labels):
            raise ValueError("the length of feature_matrix is not equal to the length of labels")
        self.length = len(feature_matrix[0])
        self.vectorcount[0] = []
        self.vectorcount[1] = []
        for vector, label in zip(feature_matrix, labels):
            self.vectorcount[label].append(vector)
        print("training finished")
        return self

    def predict_one(self, test_feature):
        if self.length == -1:
            raise ValueError("NO training")
        # calculate the probability of test_feature belong to spam or ham
        # label_dict = dict()
        # labels_set = [0, 1]
        ham_vector = np.array(self.vectorcount[0]).T
        spam_vector = np.array(self.vectorcount[1]).T
        eta = 0
        for index in range(0, len(test_feature)):
            if test_feature[index] == 0:
                continue
            word_ham = list(ham_vector[index]).count(test_feature[index])
            word_spam = list(spam_vector[index]).count(test_feature[index])
            p = word_spam / float(word_spam + word_ham)
            if p == 1:
                p -= 0.00001
            if p == 0:
                p += 0.00001
            # print(p)
            eta += np.log(1-p) - np.log(p)
        prob = expit(-eta)
        label = 0
        if prob > 0.80:
            label = 1
        return label

    def predict(self, test_feature_matrix):
        predict_labels = []
        for feature in test_feature_matrix:
            # print(feature)
            predict_labels.append(self.predict_one(feature))
        return predict_labels


def make_Dictionary(train_dir):
    emails = [os.path.join(train_dir, f) for f in os.listdir(train_dir)]
    all_words = []
    for mail in emails:
        with open(mail) as m:
            for i, line in enumerate(m):
                if i == 2:
                    words = line.split()
                    all_words += words
    dictionary = Counter(all_words)
    list_to_remove = list(dictionary.keys())
    for item in list_to_remove:
        if not item.isalpha():
            del dictionary[item]
        elif len(item) == 1:
            del dictionary[item]
    dictionary = dictionary.most_common(3000)
    return dictionary


def extract_features(mail_dir):
    files = [os.path.join(mail_dir, fi) for fi in os.listdir(mail_dir)]
    features_matrix = np.zeros((len(files), 3000))
    docID = 0
    mail_labels = np.zeros(len(files))
    for fil in files:
        with open(fil) as fi:
            for i, line in enumerate(fi):
                if i == 2:
                    words = line.split()
                    for word in words:
                        wordID = 0
                        for i, d in enumerate(dictionary):
                            if d[0] == word:
                                wordID = i
                                features_matrix[docID, wordID] = words.count(word)
            file_path_list = fil.split('\\')

            # print(file_path_list[-1][0:5])
            # print(file_path_list)
            if file_path_list[-1][0:5] == "spmsg":
                mail_labels[docID] = 1      # 1 indicates spam mail
            docID = docID + 1
    return features_matrix, mail_labels


TRAIN_DIR = "./ling-spam/train-mails"
TEST_DIR = "./ling-spam/test-mails"
dictionary = make_Dictionary(TRAIN_DIR)
print("reading and processing emails from file.")
features_matrix, labels = extract_features(TRAIN_DIR)
features_matrix[features_matrix > 0] = 1
test_feature_matrix, test_labels = extract_features(TEST_DIR)
test_feature_matrix[test_feature_matrix > 0] = 1
model = Bayes()
print("Training model.")
# train model
model.fit(features_matrix, labels)
predicted_labels = model.predict(test_feature_matrix)
print("FINISHED classifying.")


TP, TN, FP, FN = 0, 0, 0, 0
for actual_label, predicted_label in zip(test_labels, predicted_labels):
    if actual_label == 1 and predicted_label == 1:
        TP += 1
    elif actual_label == 0 and predicted_label == 0:
        TN += 1
    elif actual_label == 1 and predicted_label == 0:
        FN += 1
    else:
        FP += 1

accuracy = float(TP + TN) / (TP + TN + FP + FN)
precision = float(TP) / (TP + FP)
recall = float(TP) / (TP + FN)
F1_score = 2 * precision * recall / (precision + recall)
print("accuracy score is ", accuracy)
print("recall score is ", recall)
print("F-1 score is ", F1_score)


