import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor


def train_regressor(rssi_train, coords_train, rssi_val, coords_val, selected_aps):
    """
    Train a Random Forest regressor for multi-output regression (3D localization).
    Returns the trained model and predictions.
    """
    print("Training random forest regressor...")

    X_train = rssi_train[selected_aps]
    X_val = rssi_val[selected_aps]

    models = {}
    predictions = {}

    # Optimized Random Forest for indoor localization
    rf_model = MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=100,      # More trees for stability
            max_depth=12,          # Deeper trees for complex 3D patterns
            min_samples_split=8,   # Less restrictive for ensemble
            min_samples_leaf=5,    # Fine-grained predictions
            max_features=0.7,      # Feature subsampling for diversity
            bootstrap=True,        # Standard bagging
            oob_score=True,        # Out-of-bag evaluation
            random_state=42,
            n_jobs=-1
        )
    )
    rf_model.fit(X_train, coords_train)
    models['random_forest'] = rf_model
    predictions['rf_train'] = rf_model.predict(X_train)
    predictions['rf_val'] = rf_model.predict(X_val)
    print("✓ Enhanced Random Forest trained")

    # Print OOB score for model quality assessment
    if hasattr(rf_model.estimators_[0], 'oob_score_'):
        avg_oob = np.mean([est.oob_score_ for est in rf_model.estimators_])
        print(f"   Average OOB Score: {avg_oob:.4f}")

    return models, predictions
