import warnings
warnings.filterwarnings('ignore')
from sklearn import linear_model
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from statistics import median

class Trainer(object):
    def __init__(self, participated_features):
        self.participated_features = participated_features   

    def calculate_fscore(self,X_train, X_test, y_train,  y_test):
        clf = linear_model.LogisticRegression(random_state=42)
        clf.fit(X_train, y_train)
        f1_score = clf.score(X_test, y_test)
        return f1_score*100

    def k_best_score(self, participated_features, k, train_X, test_X, train_y,  test_y):

        X1 = train_X[participated_features]
        X2 = test_X[participated_features]

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X1)
        k_best = SelectKBest(score_func = f_classif, k=k)
        k_best.fit(X_scaled, train_y)
        scores = k_best.scores_
        score_k_best = self.calculate_fscore(X1, X2, train_y,  test_y)

        best_feature ={}
        for feature, score in zip(participated_features, scores):
            best_feature[feature] = round(score, 3)  

        sorted_dict = sorted(best_feature, key=best_feature.get, reverse=True)

        advice_kbest = sorted_dict[:k]
        return  score_k_best, advice_kbest
    
    def decision_tree_score(self, participated_features,assertive_agents,hesitant_agents, train_X, test_X, train_y, test_y):
        train_X = train_X[participated_features]
        test_X = test_X[participated_features]

        tree = DecisionTreeClassifier(random_state= 42)

        tree.fit(train_X, train_y)

        y_test_pred1 = tree.predict(test_X)

        f1_score_DT = f1_score(test_y, y_test_pred1) * 100        

        importances_sk = tree.feature_importances_

        feature_importance_sk = {}
        for i, feature in enumerate(participated_features):
            feature_importance_sk[feature] = round(importances_sk[i], 3)

        assertive_scores = [feature_importance_sk[item] for item in assertive_agents]
        if len(assertive_scores) != 0: 
            G = median(assertive_scores)

            adviced_items = [item for item in hesitant_agents if feature_importance_sk[item] > G]
        else:
            G= 0
            adviced_items = []
        return f1_score_DT, adviced_items

    def decision_tree_score_MultiClass(self, participated_features,assertive_agents,hesitant_agents,train_X, test_X, train_y,  test_y ):
        
        train_X = train_X[participated_features]
        test_X = test_X[participated_features]

        tree = DecisionTreeClassifier(random_state= 42)
        tree.fit(train_X, train_y)
        y_test_pred1 = tree.predict(test_X)
        f1_score_DT = f1_score(test_y, y_test_pred1, average='micro') * 100   

        importances_sk = tree.feature_importances_
        feature_importance_sk = {}
        for i, feature in enumerate(participated_features):
            feature_importance_sk[feature] = round(importances_sk[i], 3)

        assertive_scores = [feature_importance_sk[item] for item in assertive_agents]
        if len(assertive_scores) != 0: 
            G = median(assertive_scores)
            adviced_items = [item for item in hesitant_agents if feature_importance_sk[item] > G]
        else:
            G= 0
            adviced_items = []
        return f1_score_DT, adviced_items
    
    def Warm_up(self,participated_agents,initial_actions_feat):

        hesitant_agents = []
        assertive_agents = []
        selected_features = []

        for i  in range(len(participated_agents)):
            if participated_agents[i] not in initial_actions_feat:
                hesitant_agents.append(participated_agents[i])
            elif participated_agents[i] in initial_actions_feat:
                assertive_agents.append(participated_agents[i])
        
        for j  in range(len(initial_actions_feat)):
            if initial_actions_feat[j] not in participated_agents:
                selected_features.append(initial_actions_feat[j])

        k = (len(assertive_agents)/2) + len(hesitant_agents) 

        return selected_features, k,assertive_agents,hesitant_agents