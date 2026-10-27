import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4

MONTHS = {
    "jan": 0, "feb": 1, "mar": 2, "apr": 3, "may": 4, "jun": 5,
    "jul": 6, "aug": 7, "sep": 8, "oct": 9, "nov": 10, "dec": 11,
}


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """Return (evidence, labels) loaded from the CSV file."""
    evidence = []
    labels = []

    with open(filename, newline="") as f:
        for row in csv.DictReader(f):
            evidence.append([
                int(row["Administrative"]),
                float(row["Administrative_Duration"]),
                int(row["Informational"]),
                float(row["Informational_Duration"]),
                int(row["ProductRelated"]),
                float(row["ProductRelated_Duration"]),
                float(row["BounceRates"]),
                float(row["ExitRates"]),
                float(row["PageValues"]),
                float(row["SpecialDay"]),
                MONTHS[row["Month"].strip()[:3].lower()],
                int(row["OperatingSystems"]),
                int(row["Browser"]),
                int(row["Region"]),
                int(row["TrafficType"]),
                1 if row["VisitorType"] == "Returning_Visitor" else 0,
                1 if row["Weekend"].strip().upper() == "TRUE" else 0,
            ])
            labels.append(1 if row["Revenue"].strip().upper() == "TRUE" else 0)

    return evidence, labels


def train_model(evidence, labels):
    """Return a 1-nearest-neighbor model fitted on the data."""
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model


def evaluate(labels, predictions):
    """Return (sensitivity, specificity)."""
    true_positive = true_negative = 0
    total_positive = total_negative = 0

    for actual, predicted in zip(labels, predictions):
        if actual == 1:
            total_positive += 1
            true_positive += predicted == 1
        else:
            total_negative += 1
            true_negative += predicted == 0

    return true_positive / total_positive, true_negative / total_negative


if __name__ == "__main__":
    main()
