
import argparse
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from azureml.core.run import Run
# from azureml.core import Dataset


path = "data.csv"


def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_estimators', type=int, default=100, help="Number of trees in the forest")
    parser.add_argument('--min_samples_split', type=int, default=2, help="Minimum number of samples required to split an internal node")
    parser.add_argument('--max_features', type=str, default='auto', help="{'auto', 'sqrt', 'log2'}")

    args = parser.parse_args()

    df = pd.read_csv(path)

    x = df.copy()
    x.drop(['diagnosis', 'id', 'Unnamed: 32'], inplace=True, axis=1)
    y = df.pop('diagnosis')

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2025)

    run = Run.get_context()

    run.log("n_estimators:", np.int(args.n_estimators))
    run.log("min_samples_split:", np.int(args.min_samples_split))
    run.log("max_features:", np.str(args.max_features))

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        min_samples_split=args.min_samples_split,
        max_features=args.max_features
    ).fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    run.log("Accuracy", np.float(accuracy))

    # os.makedirs('outputs', exist_ok=True)
    joblib.dump(value=model, filename='outputs/model_hyperdrive.pkl')


if __name__ == '__main__':
    main()
