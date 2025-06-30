import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_params(path="params.yaml"):
    """Baca parameter dari params.yaml, kembalikan dict di bawah key 'preprocess'."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("preprocess", {})

def preprocess(input_path, output_dir, test_size, random_state):
    # 1. Baca data mentah
    df = pd.read_csv(input_path)
    # Hilangkan leading/trailing whitespace di nama kolom
    df.columns = df.columns.str.strip()
    # Juga strip value target jika ada spasi
    df["income"] = df["income"].astype(str).str.strip()

    # 2. Definisikan fitur & target
    target = "income"
    features = [col for col in df.columns if col != target]

    # 3. Split train/test
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )

    # 4. Pisahkan X dan y
    X_train = train_df[features]
    X_test  = test_df[features]

    # 5. Tentukan fitur numerik & kategorikal
    numeric_feats     = X_train.select_dtypes(include=["int64","float64"]).columns.tolist()
    categorical_feats = X_train.select_dtypes(include=["object"]).columns.tolist()

    # 6. Build preprocessing pipelines
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        # scikit-learn >=1.2: gunakan sparse_output
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ("num", num_pipe, numeric_feats),
        ("cat", cat_pipe, categorical_feats)
    ])

    # 7. Fit & transform
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans  = preprocessor.transform(X_test)

    # 8. Buat nama kolom baru setelah one-hot
    ohe_cols = preprocessor.named_transformers_["cat"]["onehot"] \
                      .get_feature_names_out(categorical_feats)
    new_columns = numeric_feats + list(ohe_cols)

    # 9. Rekonstruksi DataFrame hasil preprocessing
    train_out = pd.DataFrame(X_train_trans, columns=new_columns)
    train_out[target] = train_df[target].values

    test_out  = pd.DataFrame(X_test_trans, columns=new_columns)
    test_out[target]  = test_df[target].values

    # 10. Simpan ke output_dir
    os.makedirs(output_dir, exist_ok=True)
    train_out.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    test_out .to_csv(os.path.join(output_dir, "test.csv" ), index=False)

    print(f"Saved processed files to '{output_dir}/'")

if __name__ == "__main__":
    # Load parameter
    params       = load_params("params.yaml")
    test_size    = params.get("test_size", 0.2)
    random_state = params.get("random_state", 42)

    # Jalankan preprocessing
    preprocess(
        input_path   = "data/raw/adult.csv",
        output_dir   = "data/processed",
        test_size    = test_size,
        random_state = random_state
    )
