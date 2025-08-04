import os
import joblib
import numpy as np
import pandas as pd
import multiprocessing
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

class Hyperparameter_Grid_Search_Classification:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_file": ("STRING", {"tooltip": "Path to the input file"}),
                "algorithm": (["xgboost", "random_forest", "decision_tree", "lightgbm", "logistic", "lasso", "svm"],
                              {"tooltip": "Classification algorithm"}),
                "target_column": ("STRING", {"default": "Label", "tooltip": "Target column name"}),
                "advanced": ("BOOLEAN", {"default": False, "tooltip": "Advanced mode"}),
            },
            "optional": {
                
                "test_size": ("FLOAT", {"default": 0.2, "min": 0.1, "max": 0.5, "step": 0.05,
                                        "tooltip": "Test set size"}),
                "num_cores": ("INT", {"default": -1, "min": -1, "max": multiprocessing.cpu_count(), "step": 1,
                                      "tooltip": "Number of cores"}),
                "cv_splits": ("INT", {"default": 5, "min": 3, "max": 10, "step": 1,
                                       "tooltip": "Number of cross-validation splits"}),
                "verbose": ("INT", {"default": 1, "min": 0, "max": 2, "step": 1,
                                    "tooltip": "Verbosity level"}),
                "random_state": ("INT", {"default": 42, "min": 0, "max": 999, "step": 1,
                                         "tooltip": "Random seed"}),
                "n_estimators": ("STRING", {"default": "[100, 200, 300]",
                                            "tooltip": "Number of trees in the forest"}),
                "max_depth": ("STRING", {"default": "[3, 5, 7]",
                                         "tooltip": "Maximum depth of the trees"}),
                "learning_rate": ("STRING", {"default": "[0.01, 0.05, 0.1]",
                                             "tooltip": "Learning rate for boosting"}),
                "min_samples_split": ("STRING", {"default": "[2, 5, 10]",
                                                 "tooltip": "Minimum number of samples required to split an internal node"}),
                "min_samples_leaf": ("STRING", {"default": "[1, 2, 4]",
                                                "tooltip": "Minimum number of samples required to be at a leaf node"}),
                "criterion": ("STRING", {"default": "['gini', 'entropy']",
                                         "tooltip": "Criterion for splitting"}),
                "subsample": ("STRING", {"default": "[0.6, 0.8, 1.0]",
                                         "tooltip": "Proportion of samples to be used for each tree"}),
                "reg_alpha": ("STRING", {"default": "[0.1, 1, 10]",
                                         "tooltip": "L1 regularization term on weights"}),
                "reg_lambda": ("STRING", {"default": "[1, 10, 100]",
                                         "tooltip": "L2 regularization term on weights"}),
                "C": ("STRING", {"default": "[0.01, 0.1, 1, 10, 100]",
                                 "tooltip": "Regularization parameter"}),
                "penalty": ("STRING", {"default": "['l2']",
                                       "tooltip": "Regularization penalty"}),
                "solver": ("STRING", {"default": "['lbfgs']",
                                      "tooltip": "Solver for optimization"}),
                "kernel": ("STRING", {"default": "['linear', 'rbf']",
                                      "tooltip": "Kernel for SVM"}),
                "gamma": ("STRING", {"default": "['scale', 'auto']",
                                     "tooltip": "Kernel coefficient for 'rbf' kernel"}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING",)
    RETURN_NAMES = ("MODEL_PATH", "DESCRIPTORS_PATH", "X_TEST_PATH", "Y_TEST_PATH",)
    FUNCTION = "perform_grid_search"
    CATEGORY = "QSAR/CLASSIFICATION/GRID SEARCH"
    OUTPUT_NODE = True
    
    def perform_grid_search(self, input_file, algorithm, advanced,
                          test_size=0.2, num_cores=-1, cv_splits=5, verbose=1,
                          target_column="Label", random_state=42,
                          n_estimators="[100, 200, 300]",
                          learning_rate="[0.01, 0.05, 0.1]",
                          max_depth="[3, 5, 7]",
                          min_samples_split="[2, 5, 10]",
                          min_samples_leaf="[1, 2, 4]",
                          criterion="['gini', 'entropy']",
                          subsample="[0.6, 0.8, 1.0]",
                          reg_alpha="[0.1, 1, 10]",
                          reg_lambda="[1, 10, 100]",
                          C="[0.01, 0.1, 1, 10, 100]",
                          penalty="['l2']",
                          solver="['lbfgs']",
                          kernel="['linear', 'rbf']",
                          gamma="['scale', 'auto']"):
        """
        Perform grid search for model hyperparameter optimization.
        """
        output_dir = "QSAR/Model"
        os.makedirs(output_dir, exist_ok=True)

        def parse_param(param_str):
            try:
                safe_str = param_str.replace("None", "None")
                parsed = eval(safe_str)
                if not isinstance(parsed, list):
                     if parsed is None: return [None]
                     return [parsed]
                return parsed
            except Exception as e:
                print(f"Parameter parsing error for '{param_str}': {str(e)}. Using default fallback.")
                if 'estimators' in param_str.lower() or 'depth' in param_str.lower(): return [100]
                if 'rate' in param_str.lower(): return [0.1]
                if 'split' in param_str.lower() or 'leaf' in param_str.lower(): return [2]
                if 'criterion' in param_str.lower(): return ['gini']
                if 'subsample' in param_str.lower(): return [0.8]
                if 'alpha' in param_str.lower() or 'lambda' in param_str.lower(): return [1]
                if 'C' in param_str.lower(): return [1.0]
                if 'penalty' in param_str.lower(): return ['l2']
                if 'solver' in param_str.lower(): return ['lbfgs']
                if 'kernel' in param_str.lower(): return ['rbf']
                if 'gamma' in param_str.lower(): return ['scale']
                return []

        try:
            data = pd.read_csv(input_file)
        except Exception as e:
            error_msg = f"❌ Error loading input file: {str(e)}"
            return {"ui": {"text": error_msg}, "result": ("", "", "", "")}

        if target_column not in data.columns:
            error_msg = f"❌ Target column '{target_column}' not found in dataset."
            return {"ui": {"text": error_msg}, "result": ("", "", "", "")}

        X = data.drop(columns=[target_column])
        y = data[target_column]

        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                            random_state=random_state, stratify=y)
        except ValueError as split_e:
             error_msg = f"❌ Error splitting data: {split_e}"
             return {"ui": {"text": error_msg}, "result": ("", "", "", "")}


        if num_cores == -1:
            n_jobs = -1
        else:
            available_cores = max(1, multiprocessing.cpu_count())
            n_jobs = min(num_cores, available_cores)

        pipeline_needed = algorithm in ["logistic", "lasso", "svm"]
        param_prefix = "clf__" if pipeline_needed else ""

        model_instance = None
        param_grid = {}
        model_abbr = ""

        try:
            if algorithm == "xgboost":
                model_instance = XGBClassifier(eval_metric="logloss", random_state=random_state, use_label_encoder=False)
                model_abbr = "XGB"
                param_grid = {
                    f'{param_prefix}n_estimators': parse_param(n_estimators),
                    f'{param_prefix}learning_rate': parse_param(learning_rate),
                    f'{param_prefix}max_depth': parse_param(max_depth),
                }
            elif algorithm == "random_forest":
                model_instance = RandomForestClassifier(random_state=random_state, n_jobs=n_jobs)
                model_abbr = "RF"
                param_grid = {
                    f'{param_prefix}n_estimators': parse_param(n_estimators),
                    f'{param_prefix}max_depth': parse_param(max_depth),
                    f'{param_prefix}min_samples_split': parse_param(min_samples_split),
                    f'{param_prefix}min_samples_leaf': parse_param(min_samples_leaf),
                    f'{param_prefix}criterion': parse_param(criterion),
                }
            elif algorithm == "decision_tree":
                model_instance = DecisionTreeClassifier(random_state=random_state)
                model_abbr = "DT"
                param_grid = {
                    f'{param_prefix}max_depth': parse_param(max_depth),
                    f'{param_prefix}min_samples_split': parse_param(min_samples_split),
                    f'{param_prefix}min_samples_leaf': parse_param(min_samples_leaf),
                    f'{param_prefix}criterion': parse_param(criterion),
                }
            elif algorithm == "lightgbm":
                model_instance = LGBMClassifier(random_state=random_state, n_jobs=n_jobs)
                model_abbr = "LGBM"
                param_grid = {
                    f'{param_prefix}n_estimators': parse_param(n_estimators),
                    f'{param_prefix}learning_rate': parse_param(learning_rate),
                    f'{param_prefix}max_depth': [d if d is None or d > 0 else -1 for d in parse_param(max_depth)],
                    f'{param_prefix}subsample': parse_param(subsample),
                    f'{param_prefix}reg_alpha': parse_param(reg_alpha),
                    f'{param_prefix}reg_lambda': parse_param(reg_lambda),
                }
            elif algorithm == "logistic":
                model_instance = LogisticRegression(max_iter=2000, random_state=random_state, n_jobs=n_jobs)
                model_abbr = "LogReg"
                param_grid = {
                    f'{param_prefix}C': parse_param(C),
                    f'{param_prefix}penalty': parse_param(penalty),
                    f'{param_prefix}solver': parse_param(solver),
                }
            elif algorithm == "lasso":
                compatible_solvers = [s for s in parse_param(solver) if s in ['liblinear', 'saga']]
                if not compatible_solvers: compatible_solvers = ['liblinear']
                model_instance = LogisticRegression(penalty='l1', max_iter=2000, random_state=random_state, n_jobs=n_jobs)
                model_abbr = "LASSO"
                param_grid = {
                    f'{param_prefix}C': parse_param(C),
                    f'{param_prefix}solver': compatible_solvers,
                }
            elif algorithm == "svm":
                model_instance = SVC(probability=True, random_state=random_state)
                model_abbr = "SVM"
                param_grid = {
                    f'{param_prefix}C': parse_param(C),
                    f'{param_prefix}kernel': parse_param(kernel),
                    f'{param_prefix}gamma': parse_param(gamma),
                }
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")

            if pipeline_needed:
                final_model = Pipeline([("scaler", StandardScaler()), ("clf", model_instance)])
            else:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                final_model = model_instance

            param_grid_str = "\n".join([f"  - {k}: {v}" for k, v in param_grid.items()])
            print(f"Algorithm: {algorithm}\nParameter Grid:\n{param_grid_str}")

        except Exception as setup_e:
            error_msg = f"❌ Error setting up model or parameters: {str(setup_e)}"
            return {"ui": {"text": error_msg}, "result": ("", "", "", "")}


        try:
            scoring = {
                'accuracy': 'accuracy',
                'f1': make_scorer(f1_score),
                'roc_auc': 'roc_auc',
                'precision': make_scorer(precision_score),
                'recall': make_scorer(recall_score),
            }
            cv_strategy = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

            grid_search = GridSearchCV(final_model, param_grid, cv=cv_strategy, scoring=scoring,
                                    refit='accuracy', return_train_score=True,
                                    verbose=verbose, n_jobs=n_jobs)

            grid_search.fit(X_train, y_train)

        except Exception as grid_e:
            error_msg = f"❌ Error during GridSearchCV: {str(grid_e)}"
            import traceback
            error_msg += f"\nTraceback:\n{traceback.format_exc()}"
            return {"ui": {"text": error_msg}, "result": ("", "", "", "")}

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        predictions = best_model.predict(X_test)
        try:
            if hasattr(best_model, "predict_proba"):
                 y_pred_proba = best_model.predict_proba(X_test)[:, 1]
                 roc_auc = roc_auc_score(y_test, y_pred_proba)
            else:
                 roc_auc = np.nan

            eval_results = {
                "accuracy": accuracy_score(y_test, predictions),
                "f1_score": f1_score(y_test, predictions),
                "roc_auc": roc_auc,
                "precision": precision_score(y_test, predictions),
                "recall": recall_score(y_test, predictions),
            }
        except Exception as eval_e:
             error_msg = f"❌ Error evaluating best model: {eval_e}"
             eval_results = {k: np.nan for k in ['accuracy', 'f1_score', 'roc_auc', 'precision', 'recall']}


        try:
            model_filename = f"Best_Classifier_{model_abbr}.pkl"
            model_path = os.path.join(output_dir, model_filename)
            joblib.dump(best_model, model_path)

            descriptors_path = os.path.join(output_dir, "Final_Selected_Descriptors_GridSearch.txt")
            with open(descriptors_path, "w") as f:
                f.write("\n".join(X.columns))

            X_test_df = pd.DataFrame(X_test, columns=X.columns)
            y_test_df = pd.DataFrame(y_test, columns=[target_column])
            X_test_path = os.path.join(output_dir, "X_test_GridSearch.csv")
            y_test_path = os.path.join(output_dir, "y_test_GridSearch.csv")
            X_test_df.to_csv(X_test_path, index=False)
            y_test_df.to_csv(y_test_path, index=False)

            best_params_path = os.path.join(output_dir, f"Best_Hyperparameters_{model_abbr}.txt")
            with open(best_params_path, "w") as f:
                f.write(f"Algorithm: {algorithm}\n")
                for param, value in best_params.items():
                    clean_param = param.split('__')[-1]
                    f.write(f"{clean_param}: {value}\n")

        except Exception as save_e:
            error_msg = f"❌ Error saving artifacts: {str(save_e)}"
            return {"ui": {"text": error_msg}, "result": (model_path if 'model_path' in locals() else "",
                                                                                  descriptors_path if 'descriptors_path' in locals() else "",
                                                                                  X_test_path if 'X_test_path' in locals() else "",
                                                                                  y_test_path if 'y_test_path' in locals() else "")}

        best_params_text = "\n".join([f"  - {param.split('__')[-1]}: {value}" for param, value in best_params.items()])
        log_message = (
            "========================================\n"
            "🔹 Grid Search Hyperparameter Optimization Completed! 🔹\n"
            "========================================\n"
            f"✅ Best Accuracy: {eval_results['accuracy']:.4f}\n"
            f"✅ Best Model: {model_path}\n"
            f"✅ Best Hyperparameters: {best_params_text}\n"
            f"💾 Saved Best Model: {model_path}\n"
            f"💾 Saved Descriptors: {descriptors_path}\n"
            f"💾 Saved Test Data (X_test): {X_test_path}\n"
            f"💾 Saved Test Data (y_test): {y_test_path}\n"
            "========================================\n"
        )
        
        return {
            "ui": {"text": log_message},
            "result": (str(model_path), str(descriptors_path), str(X_test_path), str(y_test_path),)
        }

# Node registration
NODE_CLASS_MAPPINGS = {
    "Hyperparameter_Grid_Search_Classification": Hyperparameter_Grid_Search_Classification
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Hyperparameter_Grid_Search_Classification": "Grid Search Hyperparameter (Classification)"
} 