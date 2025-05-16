
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

class DJL:
    def __init__(self, m=10, mlp_max_iter=200, gamma=1.0):
        self.m = m
        self.mlp_max_iter = mlp_max_iter
        self.gamma = gamma
        self.Cost_Dict = {}
        self.Q_nn = {}
        self.bellm = np.zeros(self.m)
        self.tau = []
        self.R_set = []

    def initialize_training(self, df):
        self.train_data = df.copy()
        self.bellm = np.zeros(self.m)
        self.tau = []
        self.R_set = []
        self.Cost_Dict = {}
        self.Q_nn = {}

    def get_cost(self, l, r):
        key = f"{l}:{r}"
        if key not in self.Cost_Dict:
            sub = self.train_data[
                (self.train_data["at"] >= l / self.m) & (self.train_data["at"] <= r / self.m)
            ]
            if len(sub) == 0:
                self.Cost_Dict[key] = 0
            else:
                X = np.stack(sub["xt"])
                y = sub["yt"].values
                model = MLPRegressor(hidden_layer_sizes=(32, 32), max_iter=self.mlp_max_iter, random_state=42)
                model.fit(X, y)
                y_pred = model.predict(X)
                self.Cost_Dict[key] = np.sum((y - y_pred) ** 2)
                self.Q_nn[key] = model
        return self.Cost_Dict[key]

    def get_partition(self):
        self.R_set.append([-1])
        self.tau.append([])

        for v_star in range(self.m):
            bel_costs = []
            for v in self.R_set[v_star]:
                bel = -self.gamma if v == -1 else self.bellm[v]
                cost = self.get_cost(v + 1, v_star + 1)
                bel_costs.append(bel + cost + self.gamma)

            self.bellm[v_star] = np.min(bel_costs)
            v1 = self.R_set[v_star][np.argmin(bel_costs)]
            self.tau.append(sorted(set(self.tau[v1 + 1] + [v1])))

            new_R = []
            for v in self.R_set[v_star] + [v_star]:
                bel = -self.gamma if v == -1 else self.bellm[v]
                cost = self.get_cost(v + 1, v_star + 1)
                if bel + cost <= self.bellm[v_star]:
                    new_R.append(v)
            self.R_set.append(new_R)

        return np.array(self.tau[-1]) + 1

    def evaluate(self, tau, df_test, policy_fn):
        V_hat = 0.0
        n = len(df_test)

        for i in range(len(tau)):
            l = tau[i]
            r = tau[i + 1] if i < len(tau) - 1 else self.m

            sub = df_test[
                (df_test["at"] >= l / self.m) & 
                (df_test["at"] < r / self.m if i < len(tau) - 1 else df_test["at"] <= r / self.m)
            ]
            if len(sub) == 0:
                continue

            X = np.stack(sub["xt"])
            A_pi = np.array([policy_fn(x) for x in X])
            match = ((A_pi >= l / self.m) & (A_pi < r / self.m)) if i < len(tau) - 1 else ((A_pi >= l / self.m) & (A_pi <= r / self.m))

            Q_model = self.Q_nn[f"{l}:{r}"]
            Q_pred = Q_model.predict(X)

            V_hat += np.sum(Q_pred * match)

        return V_hat / n
