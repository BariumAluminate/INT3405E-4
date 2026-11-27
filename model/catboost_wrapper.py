from catboost import CatBoostClassifier, Pool


class catboostWrapper:
    def __init__(
        self,
        loss_function="Logloss",
        eval_metric="Logloss",
        iterations=2000,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3.0,
        thread_count=-1,
        random_seed=42,
    ):
        self.model = CatBoostClassifier(
            loss_function=loss_function,
            eval_metric=eval_metric,
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            l2_leaf_reg=l2_leaf_reg,
            thread_count=thread_count,
            random_seed=random_seed,
        )

    def fit(self, X, y=None, eval_set=None, early_stopping_rounds=100, verbose=200):
        if isinstance(X, Pool):
            self.model.fit(
                X,
                eval_set=eval_set,
                early_stopping_rounds=early_stopping_rounds,
                verbose=verbose,
            )
        else:
            self.model.fit(
                X,
                y,
                eval_set=eval_set,
                early_stopping_rounds=early_stopping_rounds,
                verbose=verbose,
            )
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        self.model.save_model(path)

    def load(self, path):
        self.model = CatBoostClassifier()
        self.model.load_model(path)
        return self