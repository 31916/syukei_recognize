import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model

def evaluate_model_with_visualization(X, y, subjects, model_path, label_encoder_path, output_path="./output"):
    # Load model and label encoder
    model = load_model(model_path)
    with open(label_encoder_path, "r") as f:
        label_encoder = json.load(f)
    
    # Ensure X and y are numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Generate predictions
    predictions = model.predict(X)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(y, axis=1)
    
    # Decode labels
    true_label_names = [label_encoder[label] for label in true_labels]
    predicted_label_names = [label_encoder[label] for label in predicted_labels]

    # Calculate classification report
    class_report = classification_report(true_label_names, predicted_label_names, output_dict=True)

    # Create confusion matrix
    cm = confusion_matrix(true_label_names, predicted_label_names, labels=label_encoder)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig(f"{output_path}/confusion_matrix.png")
    plt.close()

    # Save label-wise accuracy
    label_accuracy = {
        label: {
            "correct": cm[i, i],
            "total": sum(cm[i]),
            "accuracy": cm[i, i] / sum(cm[i]) if sum(cm[i]) > 0 else 0
        }
        for i, label in enumerate(label_encoder)
    }
    
    # Visualize label-wise accuracy
    accuracies = [value["accuracy"] for value in label_accuracy.values()]
    plt.bar(label_encoder, accuracies)
    plt.xticks(rotation=90)
    plt.title("Label-wise Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Labels")
    plt.tight_layout()
    plt.savefig(f"{output_path}/label_accuracy.png")
    plt.close()

    # Save TOP3 predictions
    top3_predictions = []
    for i, probs in enumerate(predictions):
        top3_indices = np.argsort(probs)[::-1][:3]
        top3_labels = [label_encoder[idx] for idx in top3_indices]
        top3_probs = [probs[idx] for idx in top3_indices]
        top3_predictions.append({
            "true_label": true_label_names[i],
            "top3_predictions": [
                {"label": label, "probability": float(prob)}
                for label, prob in zip(top3_labels, top3_probs)
            ]
        })
    
    # Save results to JSON
    results = {
        "mean_accuracy": np.mean(accuracies),
        "std_accuracy": np.std(accuracies),
        "label_accuracy": label_accuracy,
        "top3_predictions": top3_predictions
    }
    with open(f"{output_path}/results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"Evaluation completed. Results saved to {output_path}/results.json")

# Example usage
evaluate_model_with_visualization(
    X=angles,
    y=labels,
    subjects=subjects,
    model_path="./model/model.h5",
    label_encoder_path="./model/label_encoder.json",
    output_path="./output"
)
