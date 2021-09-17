from mlxtend.evaluate import bias_variance_decomp
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression

import time
import constant
import pandas as pd
from constant import Keys as key
from constant import Classification as classification

from sklearn.metrics import zero_one_loss
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


class ClassificationResult:

    def __init__(self):
        self.params = {}

    def display(self):
        for each in self.params.keys():
            print(each.upper())
            print(self.params[each])


class Classification:

    def __init__(self, c_type, x_train=None, y_train=None,
                 x_test=None, y_test=None, binary=False, bias_variance=False):
        self.c_type = c_type
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.binary = binary
        self.bias_variance = bias_variance

    def compute(self):
        cr = ClassificationResult()
        cr.params[key.TYPE.value] = self.c_type
        now = int(round(time.time() * 1000))
        if self.c_type == classification.SVM:
            cr = self.compute_svm(cr)
        elif self.c_type == classification.KNN:
            cr = self.compute_knn(cr)
        elif self.c_type == classification.GNB:
            cr = self.compute_gnb(cr)
        elif self.c_type == classification.DTC:
            cr = self.compute_cart(cr)
        elif self.c_type == classification.RF:
            cr = self.compute_rf(cr)
        elif self.c_type == classification.ADABC:
            cr = self.compute_adab(cr)
        elif self.c_type == classification.LR:
            cr = self.compute_lr(cr)
        then = int(round(time.time() * 1000))
        cr.params[key.CT.value] = (then - now)/1000
        return cr

    def svm_grid_search(self, gamma_in, c_in, precomputed):
        gammas = [gamma_in/10, gamma_in, gamma_in*10]
        cs = [c_in/10, c_in, c_in*10]
        min_loss = len(self.y_test)
        # print('%30s%30s%30s%35s' % ('Time', 'C', 'Gamma', 'Test LOSS'))
        for gamma in gammas:
            for c in cs:
                test_lost = min_loss
                model = SVC(kernel='rbf', C=c, gamma=gamma, class_weight='balanced')
                # now = round(time.time() * 1000)
                if (c, gamma) not in precomputed:
                    model.fit(self.x_train, self.y_train)
                    yfit = model.predict(self.x_test)
                    test_lost = int(round(zero_one_loss(self.y_test, yfit)*len(self.y_test), 0))
                    precomputed[(c, gamma)] = test_lost
                else:
                    test_lost = precomputed[(c, gamma)]
                # then = round(time.time() * 1000)
                # print('%30s%30s%30s%35s' % (str(then-now), str(c), str(gamma), str(precomputed[(c, gamma)])))
                if (test_lost < min_loss):
                    min_loss = test_lost
                    gamma_min = gamma
                    c_min = c
        if (gamma_in == gamma_min and c_min == c_in):
            return (True, gamma_min, c_min, precomputed)
        else:
            return (True, gamma_min, c_min, precomputed)

    def compute_svm(self, cr) -> ClassificationResult:
        now = int(round(time.time() * 1000))
        gamma, C = 0.1, 1
        converged = False
        precomputed = {}
        while (not converged):
            converged, gamma, C, precomputed = self.svm_grid_search(gamma, C, precomputed)
        model = SVC(kernel='rbf', C=C, gamma=gamma, class_weight='balanced', probability=(not self.binary))
        model.fit(self.x_train, self.y_train)
        cr = self.collect_result(model, cr=cr)
        then = int(round(time.time() * 1000))
        cr.params[key.TT.value] = (then - now)/1000
        cr.params[key.SVM_GAMMA.value] = gamma
        cr.params[key.SVM_C.value] = C
        return cr

    def compute_gnb(self, cr) -> ClassificationResult:
        now = int(round(time.time() * 1000))
        model = GaussianNB()
        model.fit(self.x_train, self.y_train)
        then = int(round(time.time() * 1000))
        cr.params[key.TT.value] = (then - now)/1000
        cr = self.collect_result(model, cr=cr)
        return cr

    def search_knn_k(self, k_in, precomputed):
        ks = [max(k_in - 2, 1), max(k_in - 1, 1), max(k_in, 2), max(k_in + 1, 3), max(k_in + 2, 4)]
        min_loss = len(self.y_test)
        for k in ks:
            model = KNeighborsClassifier(n_neighbors=k)
            if k not in precomputed:
                model.fit(self.x_train, self.y_train)
                yfit = model.predict(self.x_test)
                test_lost = zero_one_loss(self.y_test, yfit)*len(self.y_test)
                precomputed[k] = test_lost
            else:
                test_lost = precomputed[k]
            if (test_lost < min_loss):
                min_loss = test_lost
                k_min = k
        if (k_in == k_min):
            return (True, k_min, precomputed)
        else:
            return (False, k_min, precomputed)

    def compute_knn(self, cr) -> ClassificationResult:
        k = 1
        now = int(round(time.time() * 1000))
        converged = False
        precomputed = {}
        while (not converged):
            converged, k, precomputed = self.search_knn_k(k, precomputed)
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(self.x_train, self.y_train)
        then = int(round(time.time() * 1000))
        cr.params[key.TT.value] = (then - now)/1000
        cr = self.collect_result(model,  cr=cr)
        cr.params[key.KNN_K.value] = k
        return cr

    def search_cart(self, depth_in, precomputed) -> ClassificationResult:
        depths = [max(1, depth_in-2), max(1, depth_in-1), max(depth_in, 2), max(depth_in+1, 3), max(depth_in+2, 4)]
        min_loss = len(self.y_test)
        for depth in depths:
            model = DecisionTreeClassifier(criterion=constant.DT_CRITERION, max_depth=depth, random_state=42)
            if depth not in precomputed:
                model.fit(self.x_train, self.y_train)
                yfit = model.predict(self.x_test)
                test_lost = zero_one_loss(self.y_test, yfit)*len(self.y_test)
                precomputed[depth] = test_lost
            else:
                test_lost = precomputed[depth]
            if (test_lost < min_loss):
                min_loss = test_lost
                depth_min = depth
        if (depth_in == depth_min):
            return (True, depth_min, precomputed)
        else:
            # print('in=%d min=%d min_loss=%3.2f' % (depth_in, depth_min, min_loss/len(self.y_test)))
            return (False, depth_min, precomputed)

    def compute_cart(self, cr) -> ClassificationResult:
        now = int(round(time.time() * 1000))
        depth = 1
        converged = False
        precomputed = {}
        while (not converged):
            converged, depth, precomputed = self.search_cart(depth, precomputed)
        model = DecisionTreeClassifier(criterion=constant.DT_CRITERION, max_depth=depth, random_state=42)
        model.fit(self.x_train, self.y_train)
        then = int(round(time.time() * 1000))
        cr.params[key.TT.value] = (then - now)/1000
        cr = self.collect_result(model,  cr=cr)
        cr.params[key.CART_DEPTH.value] = depth
        return cr

    def compute_rf(self, cr) -> ClassificationResult:
        now = int(round(time.time() * 1000))
        # The number of trees in the forest.
        trees = 100
        model = RandomForestClassifier(n_estimators=trees, random_state=42, oob_score=True)
        model.fit(self.x_train, self.y_train)
        then = int(round(time.time() * 1000))
        cr.params[key.TT.value] = (then - now)/1000
        cr.params[key.RF_TREES.value] = trees
        cr.params[key.OUT_OF_BAG_SCORE.value] = model.oob_score_
        cr.params[key.FEATURE_IMPORTANCE.value] = model.feature_importances_
        cr = self.collect_result(model, cr=cr)
        return cr

    def compute_adab(self, cr) -> ClassificationResult:
        now = int(round(time.time() * 1000))
        depth = 1
        converged = False
        precomputed = {}
        while (not converged):
            converged, depth, precomputed = self.search_cart(depth, precomputed)
        n_estimators = 100
        model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=depth),
                                   n_estimators=n_estimators, random_state=42)
        model.fit(self.x_train, self.y_train)
        then = int(round(time.time() * 1000))
        cr.params[key.TT.value] = (then - now)/1000
        cr.params[key.FEATURE_IMPORTANCE.value] = model.feature_importances_
        cr = self.collect_result(model, cr=cr)
        cr.params[key.ADAB_BASE_DEPTH.value] = depth
        return cr

    def compute_lr(self, cr) -> ClassificationResult:
        now = int(round(time.time() * 1000))
        model = LogisticRegression(random_state=0, class_weight='balanced', max_iter=1000)
        model.fit(self.x_train, self.y_train)
        then = int(round(time.time() * 1000))
        cr.params[key.TT.value] = (then - now)/1000
        cr = self.collect_result(model, cr=cr)
        return cr

    def collect_result(self, model, cr=ClassificationResult()):
        y_test_fit = model.predict(self.x_test)
        y_train_fit = model.predict(self.x_train)
        acc_test = accuracy_score(self.y_test, y_test_fit)
        acc_train = accuracy_score(self.y_train, y_train_fit)
        test_loss = zero_one_loss(self.y_test, y_test_fit)
        train_loss = zero_one_loss(self.y_train, y_train_fit)
        cr_test = classification_report(self.y_test, y_test_fit)
        cr_train = classification_report(self.y_train, y_train_fit)
        cm_test = confusion_matrix(self.y_test, y_test_fit)
        cm_train = confusion_matrix(self.y_train, y_train_fit)

        roc_test = 0
        roc_train = 0

        try:
            if self.binary:
                roc_test = roc_auc_score(self.y_test, model.predict(self.x_test))
                roc_train = roc_auc_score(self.y_train, model.predict(self.x_train))
            else:
                roc_test = roc_auc_score(self.y_test, model.predict_proba(self.x_test), multi_class='ovr')
                roc_train = roc_auc_score(self.y_train, model.predict_proba(self.x_train), multi_class='ovr')
        except Exception as e:
            print(e)

        scores = cross_val_score(model, pd.concat([self.x_train, self.x_test]),
                                 pd.concat([self.y_train, self.y_test]), cv=constant.cross_val_rounds_clas)
        if self.bias_variance:
            total, avg_bias, avg_var = bias_variance_decomp(
                model, self.x_train.values, self.y_train.values, self.x_test.values, self.y_test.values, loss='0-1_loss',
                num_rounds=constant.bias_variance_rounds_clas, random_seed=123)
            cr.params[key.AVG_BIAS.value] = avg_bias
            cr.params[key.AVG_VARIANCE.value] = avg_var

        cr.params[key.CVS.value] = scores
        cr.params[key.TEST_LOSS_0_1.value] = test_loss
        cr.params[key.TRAIN_LOSS_0_1.value] = train_loss
        cr.params[key.C_REPORT_TEST.value] = cr_test
        cr.params[key.C_REPORT_TRAIN.value] = cr_train
        cr.params[key.CM_TEST.value] = cm_test
        cr.params[key.CM_TRAIN.value] = cm_train
        cr.params[key.ROC_AOC_TEST.value] = roc_test
        cr.params[key.ROC_AOC_TRAIN.value] = roc_train
        cr.params[key.ACCURACY_TRAIN.value] = acc_train
        cr.params[key.ACCURACY_TEST.value] = acc_test
        return cr
