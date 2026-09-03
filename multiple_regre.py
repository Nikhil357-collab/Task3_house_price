import pandas as pd
from sklearn.model_selection import train_test_split


def load_and_preprocess_data(file_path):
    # Load dataset
    df = pd.read_csv(file_path)

    print("Dataset shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())

    # Separate features and target
    X = df.drop("price", axis=1)
    y = df["price"]

    # Convert categorical columns into dummy variables
    X = pd.get_dummies(X, drop_first=True)

    # Convert boolean columns to integers
    X = X.astype(int)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test


    print("\nTraining data:", X_train.shape)
    print("Testing data:", X_test.shape)


if __name__ == "__main__":

    file_path = r"AIML_T2MODEL\data\Housing.csv"

    load_and_preprocess_data(file_path)