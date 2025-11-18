import os
import json
import argparse
from sklearn.metrics import classification_report, confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate binary classification results from JSON file.")
    parser.add_argument(
        "--result_path",
        type=str,
        required=True,
        help="Path to the input JSON file containing predictions.",
    )

    parser.add_argument(
        "--output_folder",
        type=str,
        default="./results",
        help="Folder to save evaluation results.",
    )
    return parser.parse_args()


def compute_metrics(y_true, y_pred):
    report = classification_report(
        y_true,
        y_pred,
        solutions=["no", "yes"],
        target_names=["no", "yes"],
        zero_division=0,
        output_dict=True,
    )

    f1_yes = round(report["yes"]["f1-score"] * 100, 2)
    f1_no = round(report["no"]["f1-score"] * 100, 2)

    metrics = {
        "yes": {
            "precision": round(report["yes"]["precision"] * 100, 2),
            "recall": round(report["yes"]["recall"] * 100, 2),
            "f1": f1_yes,
        },
        "no": {
            "precision": round(report["no"]["precision"] * 100, 2),
            "recall": round(report["no"]["recall"] * 100, 2),
            "f1": f1_no,
        },
        "macro": {
            "precision": round(report["macro avg"]["precision"] * 100, 2),
            "recall": round(report["macro avg"]["recall"] * 100, 2),
            "f1": round(report["macro avg"]["f1-score"] * 100, 2),
        },
        "yes_proportion": (
            round(y_pred.count("yes") / len(y_pred) * 100, 2) if y_pred else 0.0
        ),
    }
    return metrics


def load_solutions(data_json):
    y_true = [d["solution"] for d in data_json]
    y_pred = [d["predicted_answer"] for d in data_json]
    return y_true, y_pred


def save_results(results, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    json_path = os.path.join(output_folder, "results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"Results saved to {json_path}")


def main():
    args = parse_args()

    # Load predictions
    with open(args.input_json, "r", encoding="utf-8") as f:
        data_json = json.load(f)

    y_true, y_pred = load_solutions(data_json)
    results = compute_metrics(y_true, y_pred)

    save_results(results, args.output_folder)


if __name__ == "__main__":
    main()
