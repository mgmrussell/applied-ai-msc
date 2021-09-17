import constant as constant
from constant import Regression as regression


from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from mlxtend.evaluate import bias_variance_decomp
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge, ARDRegression, BayesianRidge, SGDRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.kernel_ridge import KernelRidge
# from sklearn.model_selection import GridSearchCV
from xgboost.sklearn import XGBRegressor
from sklearn.svm import SVR
# import xgboost as xgb

# from catboost import CatBoostRegressor
# from lightgbm import LGBMRegressor
import pandas as pd
# import numpy as np
from constant import Keys as key
from math import sqrt
import time
import grid_search as gs
# from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared


class RegressionResult:

    def __init__(self):
        self.params = {}

    def set_params(self, key, value):
        self.params[key] = value

    def display(self):
        for each in self.params.keys():
            print(each.upper())
            print(self.params[each])


class Regression:

    def __init__(self, r_type, x_train=None, y_train=None,
                 x_test=None, y_test=None, bias_variance=True):
        self.r_type = r_type
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.bias_variance = bias_variance

    def compute(self):
        rr = RegressionResult()
        rr.params[key.TYPE.value] = self.r_type
        now = int(round(time.time() * 1000))
        if self.r_type == regression.OLS:
            rr = self.compute_ols(rr)
        # elif self.r_type == regression.RIDGE:
        #    rr = self.compute_ridge(rr)
        # elif self.r_type == regression.ELASTIC_NET:
        #    rr = self.compute_elastic_net(rr)
        # elif self.r_type == regression.LASSO:
        #    rr = self.compute_lasso(rr)
        # elif self.r_type == regression.BR_ARD:
        #    rr = self.compute_br_ard(rr)
        # elif self.r_type == regression.BR:
        #    rr = self.compute_br(rr)
        # elif self.r_type == regression.KR:
        # rr = self.compute_kr(rr)
        elif self.r_type == regression.GBRT:
            rr = self.compute_gbr(rr)
        elif self.r_type == regression.XGB:
            rr = self.compute_xgb(rr)
        elif self.r_type == regression.SVR:
            rr = self.compute_svr(rr)
        elif self.r_type == regression.SGD:
            rr = self.compute_sgd(rr)
        elif self.r_type == regression.DTR:
            rr = self.compute_cart(rr)
        elif self.r_type == regression.ADABR:
            rr = self.compute_adab(rr)

        then = int(round(time.time() * 1000))
        rr.params[key.CT.value] = (then - now)/1000
        return rr

    def check_elastic_net_search(self, alpha, l1_ratio, precomputed):
        tol_alpha = round(alpha/10, 5)
        tol_l1_ratio = round(l1_ratio/10, 5)

        alphas_1 = max(round(alpha - tol_alpha, 5), 0.00001)
        alphas_3 = round(alpha + tol_alpha, 5)

        l1_ratio_1 = max(round(l1_ratio - tol_l1_ratio, 5), 0.00001)
        l1_ratio_3 = min(round(l1_ratio+tol_l1_ratio, 5), 0.99999)

        alphas = [alphas_1, alpha, alphas_3]
        l1_ratios = [l1_ratio_1, l1_ratio, l1_ratio_3]

        max_score = 0
        max_alpha = alpha
        max_l1_ratio = l1_ratio

        for alpha_i in alphas:
            for l1_ratio_i in l1_ratios:
                if (alpha_i, l1_ratio_i) not in precomputed:
                    model = ElasticNet(alpha=alpha_i, l1_ratio=l1_ratio_i, precompute=True)
                    model.fit(self.x_train, self.y_train)
                    # score = model.score(self.x_train, self.y_train)
                    score = - mean_squared_error(self.y_test, model.predict(self.x_test))
                    precomputed[(alpha_i, l1_ratio_i)] = score
                    # print(str(alpha_i) + ', ' + str(l1_ratio_i) + ' = ' + str(score))
                else:
                    score = precomputed[(alpha_i, l1_ratio_i)]
                if score > max_score:
                    max_alpha = alpha_i
                    max_l1_ratio = l1_ratio_i
                    max_score = score
        if (max_alpha == alpha and max_l1_ratio == l1_ratio):
            return (True, max_alpha, max_l1_ratio, precomputed)
        else:
            return (False, max_alpha, max_l1_ratio, precomputed)

    def check_lasso_search(self, alpha, precomputed):
        tol_alpha = round(alpha/10, 5)
        alphas_1 = max(round(alpha - tol_alpha, 5), 0.00001)
        alphas_3 = round(alpha + tol_alpha, 5)
        alphas = [alphas_1, alpha, alphas_3]
        max_score = 0
        max_alpha = alpha
        for alpha_i in alphas:
            if alpha_i not in precomputed:
                model = Lasso(alpha=alpha_i, precompute=True)
                model.fit(self.x_train, self.y_train)
                # score = model.score(self.x_train, self.y_train)
                score = - mean_squared_error(self.y_test, model.predict(self.x_test))
                precomputed[alpha_i] = score
                # print(str(alpha_i) + ' = ' + str(score))
            else:
                score = precomputed[alpha_i]
            if score > max_score:
                max_alpha = alpha_i
                max_score = score
        if (max_alpha == alpha):
            return (True, max_alpha, precomputed)
        else:
            return (False, max_alpha, precomputed)

    def check_ridge_search(self, alpha, precomputed):
        tol_alpha = round(alpha/10, 5)
        alphas_1 = max(round(alpha - tol_alpha, 5), 0.00001)
        alphas_3 = round(alpha + tol_alpha, 5)
        alphas = [alphas_1, alpha, alphas_3]
        max_score = 0
        max_alpha = alpha
        for alpha_i in alphas:
            if alpha_i not in precomputed:
                model = Ridge(alpha=alpha_i)
                model.fit(self.x_train, self.y_train)
                # score = model.score(self.x_test, self.y_test)
                # score = model.score(self.x_train, self.y_train)

                score = - mean_squared_error(self.y_test, model.predict(self.x_test))

                precomputed[alpha_i] = score
                # print(str(alpha_i) + ' = ' + str(score))
            else:
                score = precomputed[alpha_i]
            if score > max_score:
                max_alpha = alpha_i
                max_score = score

        if (max_alpha == alpha):
            return (True, max_alpha, precomputed)
        else:
            return (False, max_alpha, precomputed)

    def elastic_net_score_function(self, rr, alpha, l1_ratio):
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, precompute=True)
        model.fit(self.x_train, self.y_train)
        score = mean_squared_error(self.y_test, model.predict(self.x_test))
        del model
        return score

    def lasso_score_function(self, rr, alpha):
        model = Lasso(alpha=alpha, precompute=True)
        model.fit(self.x_train, self.y_train)
        score = mean_squared_error(self.y_test, model.predict(self.x_test))
        del model
        return score

    def ridge_score_function(self, rr, alpha):
        model = Ridge(alpha=alpha)
        model.fit(self.x_train, self.y_train)
        score = mean_squared_error(self.y_test, model.predict(self.x_test))
        del model
        return score

    def compute_ridge(self, rr):
        now = int(round(time.time() * 1000))
        # alpha = 1E9
        # converged = False
        # precomputed = {}
        # while not converged:
        #    converged, alpha, precomputed = self.check_ridge_search(alpha, precomputed)
        # alpha = gs.grid_search1D(self.ridge_score_function, x=alpha, maximize=False, xu=1E15, xl=1E-15)

        alpha = gs.perform_ridge_cv(self.x_train, self.y_train)
        model = Ridge(alpha=alpha)
        model.fit(self.x_train, self.y_train)
        then = int(round(time.time() * 1000))
        rr.params[key.TT.value] = (then - now)/1000
        rr.set_params(key.ALPHA.value, alpha)
        rr.set_params(key.COEFFICIENTS.value, model.coef_)
        rr.set_params(key.INTERCEPT.value, model.intercept_)
        rr = self.collect_result(model, rr=rr)
        del model
        return rr

    def compute_lasso(self, rr):
        now = int(round(time.time() * 1000))
        # alpha = 1E9
        # converged = False
        # precomputed = {}
        # while not converged:
        #     converged, alpha, precomputed = self.check_lasso_search(alpha, precomputed)
        # alpha = gs.grid_search1D(self.lasso_score_function, x=alpha, maximize=False, xu=1E15, xl=1E-15)

        alpha = gs.perform_lasso_cv(self.x_train, self.y_train)
        model = Lasso(alpha=alpha, precompute=True)
        model.fit(self.x_train, self.y_train)
        then = int(round(time.time() * 1000))
        rr.params[key.TT.value] = (then - now)/1000
        rr.set_params(key.ALPHA.value, alpha)
        rr.set_params(key.COEFFICIENTS.value, model.coef_)
        rr.set_params(key.INTERCEPT.value, model.intercept_)
        rr = self.collect_result(model, rr=rr)
        del model
        return rr

    def compute_kr(self, rr) -> RegressionResult:
        now = int(round(time.time() * 1000))
        # param_grid = {"alpha": [1e0, 1e-1, 1e-2, 1e-3],
        #              "kernel": [ExpSineSquared(l, p) for l in np.logspace(-2, 2, 5) for p in np.logspace(0, 2, 5)]}
        model = KernelRidge()
        # model = GridSearchCV(KernelRidge(), param_grid=param_grid)
        model.fit(self.x_train, self.y_train)
        then = int(round(time.time() * 1000))
        rr.params[key.TT.value] = (then - now)/1000
        # rr.set_params(key.ALPHA.value, model.alpha_)
        # rr.set_params(key.KERNEL.value, model.kernel_)
        rr = self.collect_result(model, rr=rr)
        del model
        return rr

    def search_cart(self, depth_in, precomputed):
        depths = [max(1, depth_in-2), max(1, depth_in-1), max(depth_in, 2), max(depth_in+1, 3), max(depth_in+2, 4)]
        min_loss = 1E20
        for depth in depths:
            model = DecisionTreeRegressor(max_depth=depth, random_state=42)
            if depth not in precomputed:
                model.fit(self.x_train, self.y_train)
                yfit = model.predict(self.x_test)
                test_lost = mean_squared_error(self.y_test, yfit)
                precomputed[depth] = test_lost
            else:
                test_lost = precomputed[depth]
            if (test_lost < min_loss):
                min_loss = test_lost
                depth_min = depth
        if (depth_in == depth_min):
            return (True, depth_min, precomputed)
        else:
            # print('in = %d min= %d min_loss = %f' % (depth_in, depth_min, min_loss))
            return (False, depth_min, precomputed)

    def compute_elastic_net(self, rr):
        now = int(round(time.time() * 1000))
        # Perfom Grid Search of the hyperparameters
        alpha = 1E9  # 0 to ∞ alpha = 0 is ols
        l1_ratio = 0.5  # 0 to 1
        # converged = False
        # precomputed = {}
        """
        while not converged:
            converged, alpha, l1_ratio, precomputed = self.check_elastic_net_search(alpha, l1_ratio, precomputed)

        alpha, l1_ratio = gs.grid_search2D(self.elastic_net_score_function,
                                           xi=alpha, yi=l1_ratio,
                                           maximize=False,
                                           xu=1E15, xl=1E-15,
                                           yu=1, yl=1E-15,)
        """
        alpha, l1_ratio = gs.perform_enet_cv(self.x_train, self.y_train)
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, precompute=True)
        model.fit(self.x_train, self.y_train)
        then = int(round(time.time() * 1000))
        rr.params[key.TT.value] = (then - now)/1000
        rr.set_params(key.ALPHA.value, alpha)
        rr.set_params(key.L1_RATIO.value, l1_ratio)
        rr.set_params(key.COEFFICIENTS.value, model.coef_)
        rr.set_params(key.INTERCEPT.value, model.intercept_)
        rr = self.collect_result(model, rr=rr)
        del model
        return rr

    def compute_br_ard(self, rr):
        now = int(round(time.time() * 1000))
        model = ARDRegression(compute_score=True, fit_intercept=True)
        model.fit(self.x_train, self.y_train)
        then = int(round(time.time() * 1000))
        rr.params[key.TT.value] = (then - now)/1000
        rr.set_params(key.ALPHA.value, model.alpha_)
        rr.set_params(key.LAMBDA.value, model.lambda_)
        rr.set_params(key.SIGMA.value, model.sigma_)
        rr.set_params(key.COEFFICIENTS.value, model.coef_)
        rr.set_params(key.SCORES.value, model.scores_)
        rr.set_params(key.INTERCEPT.value, model.intercept_)
        rr = self.collect_result(model, rr=rr)
        del model
        return rr

    def compute_br(self, rr):
        now = int(round(time.time() * 1000))
        model = BayesianRidge(compute_score=True, fit_intercept=True)
        model.fit(self.x_train, self.y_train)
        then = int(round(time.time() * 1000))
        rr.params[key.TT.value] = (then - now)/1000
        rr.set_params(key.ALPHA.value, model.alpha_)
        rr.set_params(key.LAMBDA.value, model.lambda_)
        rr.set_params(key.SIGMA.value, model.sigma_)
        rr.set_params(key.COEFFICIENTS.value, model.coef_)
        rr.set_params(key.SCORES.value, model.scores_)
        rr.set_params(key.INTERCEPT.value, model.intercept_)
        rr = self.collect_result(model, rr=rr)
        del model
        return rr

    def svr_grid_search(self, gamma_in, c_in, precomputed):
        gammas = [gamma_in/10, gamma_in, gamma_in*10]
        cs = [c_in/10, c_in, c_in*10]
        min_loss = 1E19
        for gamma in gammas:
            for c in cs:
                mse = min_loss
                model = SVR(kernel='rbf', C=c, gamma=gamma)
                now = round(time.time() * 1000)
                if (c, gamma) not in precomputed:
                    model.fit(self.x_train, self.y_train)
                    y_test_fit = model.predict(self.x_test)
                    mse = mean_squared_error(self.y_test, y_test_fit)
                    precomputed[(c, gamma)] = mse
                else:
                    mse = precomputed[(c, gamma)]
                then = round(time.time() * 1000)
                print('%30s%30s%30s%35s' % (str(then-now), str(c), str(gamma), str(precomputed[(c, gamma)])))
                if (mse < min_loss):
                    min_loss = mse
                    gamma_min = gamma
                    c_min = c
        if (gamma_in == gamma_min and c_min == c_in):
            return (True, gamma_min, c_min, precomputed)
        else:
            return (False, gamma_min, c_min, precomputed)

    def xgb_grid_search(self, alpha_in, lamda_in, precomputed):
        alphas = [alpha_in/10, alpha_in, alpha_in*10]
        lamdas = [lamda_in/10, lamda_in, lamda_in*10]
        min_loss = 1E19
        for alpha in alphas:
            for lamda in lamdas:
                mse = min_loss
                model = XGBRegressor(objective='reg:squarederror', reg_alpha=alpha, reg_lambda=lamda)
                now = round(time.time() * 1000)
                if (alpha, lamda) not in precomputed:
                    model.fit(self.x_train, self.y_train)
                    y_test_fit = model.predict(self.x_test)
                    mse = mean_squared_error(self.y_test, y_test_fit)
                    precomputed[(alpha, lamda)] = mse
                    mse = precomputed[(alpha, lamda)]
                then = round(time.time() * 1000)
                print('%30s%30s%30s%35s' % (str(then-now), str(alpha), str(lamda), str(precomputed[(alpha, lamda)])))
                if (mse < min_loss):
                    min_loss = mse
                    alpha_min = alpha
                    lamda_min = lamda
        if (alpha_in == alpha_min and lamda_min == lamda_in):
            return (True, alpha_min, lamda_min, precomputed)
        elif len(precomputed) > 50:
            xandy = min(precomputed, key=precomputed.get)
            return (True, xandy[0], xandy[1], precomputed)
        else:
            return (False, alpha_min, lamda_min, precomputed)

    def compute_ols(self, rr) -> RegressionResult:
        now = int(round(time.time() * 1000))
        model = LinearRegression()
        model.fit(self.x_train, self.y_train)
        then = int(round(time.time() * 1000))
        rr.params[key.TT.value] = (then - now)/1000
        rr.params[key.COEFFICIENTS.value] = model.coef_
        rr.params[key.INTERCEPT.value] = model.intercept_
        rr = self.collect_result(model, rr=rr)
        del model
        return rr

    def compute_svr(self, rr) -> RegressionResult:
        gamma, C = 0.1, 1
        now = int(round(time.time() * 1000))
        converged = False
        precomputed = {}
        while (not converged):
            converged, gamma, C, precomputed = self.svr_grid_search(gamma, C, precomputed)
        model = SVR(kernel='rbf', C=C, gamma=gamma)
        model.fit(self.x_train, self.y_train)
        then = int(round(time.time() * 1000))
        rr.params[key.TT.value] = (then - now)/1000
        rr.params[key.SVM_GAMMA.value] = gamma
        rr.params[key.SVM_C.value] = C
        rr = self.collect_result(model, rr=rr)
        del model
        return rr

    def compute_sgd(self, rr) -> RegressionResult:
        now = int(round(time.time() * 1000))
        alpha, l1_ratio = gs.sgd_search(self.x_train, self.y_train, self.x_test, self.y_test)
        model = SGDRegressor(penalty='elasticnet', alpha=alpha, l1_ratio=l1_ratio)
        model.fit(self.x_train, self.y_train)
        then = int(round(time.time() * 1000))
        rr.params[key.TT.value] = (then - now)/1000
        rr.set_params(key.ALPHA.value, alpha)
        rr.set_params(key.L1_RATIO.value, l1_ratio)
        rr = self.collect_result(model, rr=rr)
        del model
        return rr

    def compute_gbr(self, rr) -> RegressionResult:
        now = int(round(time.time() * 1000))
        depth = 1
        converged = False
        precomputed = {}
        while (not converged):
            converged, depth, precomputed = self.search_cart(depth, precomputed)
        model = GradientBoostingRegressor(max_depth=depth)
        model.fit(self.x_train, self.y_train)
        then = int(round(time.time() * 1000))
        rr.params[key.TT.value] = (then - now)/1000
        rr.params[key.FEATURE_IMPORTANCE.value] = model.feature_importances_
        rr.params[key.CART_DEPTH.value] = depth
        rr = self.collect_result(model, rr=rr)
        del model
        return rr

    def compute_xgb(self, rr) -> RegressionResult:
        now = int(round(time.time() * 1000))
        # model = xgb.XGBRegressor(objective ='reg:squarederror')
        converged = False
        precomputed = {}
        alpha, lamda = 0.1, 0.1
        while(not converged):
            converged, alpha, lamda, precomputed = self.xgb_grid_search(alpha, lamda, precomputed)
        depth = 3
        model = XGBRegressor(objective='reg:squarederror', reg_alpha=alpha, reg_lambda=lamda, max_depth=depth)
        model.fit(self.x_train, self.y_train)
        then = int(round(time.time() * 1000))
        rr.params[key.TT.value] = (then - now)/1000
        rr.set_params(key.ALPHA.value, alpha)
        rr.set_params(key.LAMBDA.value, lamda)
        rr.params[key.FEATURE_IMPORTANCE.value] = model.feature_importances_
        rr = self.collect_result(model, rr=rr)
        del model
        return rr

    def compute_cart(self, rr) -> RegressionResult:
        now = int(round(time.time() * 1000))
        depth = 1
        converged = False
        precomputed = {}
        while (not converged):
            converged, depth, precomputed = self.search_cart(depth, precomputed)
        model = DecisionTreeRegressor(max_depth=depth, random_state=42)
        model.fit(self.x_train, self.y_train)
        then = int(round(time.time() * 1000))
        rr.params[key.TT.value] = (then - now)/1000
        rr = self.collect_result(model,  rr=rr)
        rr.params[key.CART_DEPTH.value] = depth
        del model
        return rr

    def compute_adab(self, rr) -> RegressionResult:
        now = int(round(time.time() * 1000))
        depth = 1
        converged = False
        precomputed = {}
        while (not converged):
            converged, depth, precomputed = self.search_cart(depth, precomputed)
        model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=depth),
                                  loss='square',
                                  n_estimators=100, random_state=42)
        model.fit(self.x_train, self.y_train)
        then = int(round(time.time() * 1000))
        rr.params[key.TT.value] = (then - now)/1000
        rr.params[key.FEATURE_IMPORTANCE.value] = model.feature_importances_
        rr = self.collect_result(model, rr=rr)
        rr.params[key.ADAB_BASE_DEPTH.value] = depth
        del model
        return rr

    def collect_result(self, model, rr=RegressionResult()):
        y_test_fit = model.predict(self.x_test)
        y_train_fit = model.predict(self.x_train)
        test_r2 = model.score(self.x_test, self.y_test)
        train_r2 = model.score(self.x_train, self.y_train)
        test_mse = mean_squared_error(self.y_test, y_test_fit)
        train_mse = mean_squared_error(self.y_train, y_train_fit)
        test_rmse = sqrt(test_mse)
        train_rmse = sqrt(train_mse)
        scores = cross_val_score(model,
                                 pd.concat([self.x_test, self.x_train]),
                                 pd.concat([self.y_test, self.y_train]), cv=constant.cross_val_rounds_regr)
        if self.bias_variance:
            total, avg_bias, avg_var = bias_variance_decomp(
                model, self.x_train.values, self.y_train.values, self.x_test.values, self.y_test.values, loss='mse',
                num_rounds=constant.bias_variance_rounds_regr, random_seed=123)
            rr.params[key.AVG_BIAS.value] = avg_bias
            rr.params[key.AVG_VARIANCE.value] = avg_var
        rr.params[key.R2_TEST.value] = test_r2
        rr.params[key.R2_TRAIN.value] = train_r2
        rr.params[key.MSE_TEST.value] = test_mse
        rr.params[key.MSE_TRAIN.value] = train_mse
        rr.params[key.RMSE_TEST.value] = test_rmse
        rr.params[key.RMSE_TRAIN.value] = train_rmse
        rr.params[key.CVS.value] = scores

        return rr
