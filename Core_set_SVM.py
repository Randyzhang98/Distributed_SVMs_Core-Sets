from meb import MEB
import numpy as np
from sklearn.svm import SVC
from Data_load import read_UCI_data

def training_loss(SVM_model, train_data):
    alpha = np.abs(SVM_model.dual_coef_)

    loss = 0.5 * np.dot(alpha, SVM_model.decision_function(train_data[SVM_model.support_])) - np.sum(alpha)
    return loss


C = 7.0
gamma = 0.5

train_data, train_label, test_data, test_label = read_UCI_data(0)

# original SVM
model = SVC(C=C, kernel='rbf', gamma=gamma)

model.fit(train_data, train_label)

SV_size = model.n_support_[0] + model.n_support_[1]

# core-set SVM

# select an initial core-set
theta = 0.1
train_size = np.size(train_label, axis=0)

epsilon = 0.0001
R, dist_core, core_set = MEB(train_data, train_label, train_size, epsilon, C, 'rbf', gamma=gamma)

print(np.size(core_set))
# print(core_set)

core_train = train_data[core_set]
core_label = train_label[core_set]

core_model = SVC(C=C, kernel='rbf', gamma=gamma)
core_model.fit(core_train, core_label)
core_SV_size = core_model.n_support_[0] + core_model.n_support_[1]

print(model.score(test_data, test_label))
print(core_model.score(test_data, test_label))

# print(training_loss(model, train_data))
# print(training_loss(core_model, core_train))










