import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_params(path="params.yaml"):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("preprocess", {})

def preprocess(input_path, output_dir, test_size, random_state):
    # 1. Read raw data
    df = pd.read_csv(input_path)

    # 2. Split into train & test
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )

    # 3. Identify features & target
    target = "income"
    features = [c for c in df.columns if c != target]
    numeric_feats = train_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_feats = train_df.select_dtypes(include=["object"]).columns.tolist()

    # 4. Build transformers
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
    ])
    preprocessor = ColumnTransformer([
        ("num", num_pipe, numeric_feats),
        ("cat", cat_pipe, categorical_feats)
    ])

    # 5. Fit & transform
    X_train = preprocessor.fit_transform(train_df[features])
    X_test  = preprocessor.transform(test_df[features])

    # 6. Reconstruct DataFrames
    ohe_cols = preprocessor.named_transformers_["cat"]["onehot"] \
                    .get_feature_names_out(categorical_feats)
    all_cols = numeric_feats + list(ohe_cols)

    train_out = pd.DataFrame(X_train, columns=all_cols)
    train_out[target] = train_df[target].reset_index(drop=True)

    test_out = pd.DataFrame(X_test, columns=all_cols)
    test_out[target] = test_df[target].reset_index(drop=True)

    # 7. Save outputs
    os.makedirs(output_dir, exist_ok=True)
    train_out.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    test_out.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    print(f"Saved processed files to {output_dir}/")

if __name__ == "__main__":
    # Load parameters
    params = load_params("params.yaml")
    test_size    = params.get("test_size", 0.2)
    random_state = params.get("random_state", 42)

    # Define paths
    raw_data_path = "data/raw/adult.csv"
    processed_dir = "data/processed"

    # Run preprocessing
    preprocess(
        input_path=raw_data_path,
        output_dir=processed_dir,
        test_size=test_size,
        random_state=random_state
    )
