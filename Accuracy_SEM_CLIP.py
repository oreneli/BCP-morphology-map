import os
import pandas as pd

def calculate_accuracy_with_errors(csv_file, base_path, plot_cm=True, save_cm_path=None):
    """
    Calculate accuracy + per-class error table + (optional) confusion matrix plot/save.
    """
    # Read CSV
    df = pd.read_csv(csv_file)
    df.columns = [c.strip().lower() for c in df.columns]

    required_cols = {"image name", "predicted class", "similarity"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"CSV must contain columns: {required_cols}")

    classes = ["dots", "lines", "mixed"]
    stats = {cls: {"total": 0, "correct": 0, "errors": 0} for cls in classes}

    y_true, y_pred = [], []

    for _, row in df.iterrows():
        image_name = str(row["image name"]).strip()
        predicted_class = str(row["predicted class"]).strip().lower()

        if predicted_class not in classes:
            print(f"⚠️ Warning: Predicted class '{predicted_class}' not in {classes} for {image_name}. Skipping.")
            continue

        ground_truth_class = None
        for cls in classes:
            if os.path.exists(os.path.join(base_path, cls, image_name)):
                ground_truth_class = cls
                break

        if ground_truth_class is None:
            print(f"⚠️ Warning: Image {image_name} not found in any ground truth folder. Skipping.")
            continue

        y_true.append(ground_truth_class)
        y_pred.append(predicted_class)

        stats[ground_truth_class]["total"] += 1
        if predicted_class == ground_truth_class:
            stats[ground_truth_class]["correct"] += 1
        else:
            stats[ground_truth_class]["errors"] += 1

    # Print results
    print("\n📊 Per-class Error Table:")
    print("{:<10} {:<10} {:<10} {:<10} {:<10}".format("Class", "Total", "Correct", "Errors", "Error Rate"))
    print("-" * 55)
    for cls in classes:
        total_cls = stats[cls]["total"]
        correct_cls = stats[cls]["correct"]
        errors_cls = stats[cls]["errors"]
        error_rate = (errors_cls / total_cls * 100) if total_cls > 0 else 0
        print("{:<10} {:<10} {:<10} {:<10} {:<9.2f}%".format(cls, total_cls, correct_cls, errors_cls, error_rate))

    total_found = len(y_true)
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    accuracy = correct / total_found if total_found > 0 else 0
    print(f"\n✅ Overall Accuracy: {accuracy:.2%} ({correct}/{total_found})")

    # Confusion matrix plot/save
    if plot_cm and total_found > 0:
        # If saving, force a headless-safe backend
        if save_cm_path:
            import matplotlib
            matplotlib.use("Agg")  # safe in headless environments

        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

        cm = confusion_matrix(y_true, y_pred, labels=classes)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

        fig, ax = plt.subplots(figsize=(6, 5))
        out = disp.plot(ax=ax, values_format="d", colorbar=True)
        out.im_.set_cmap("Blues")

        cm_max = out.confusion_matrix.max() if hasattr(out, "confusion_matrix") else cm.max()
        thresh = cm_max * 0.5

        for i, t in enumerate(out.text_.ravel()):
            r = i // len(classes)
            c = i % len(classes)
            t.set_color("white" if cm[r, c] > thresh else "black")

        ax.set_title("Confusion Matrix")
        plt.tight_layout()

        if save_cm_path:
            save_cm_path = os.path.abspath(save_cm_path)
            os.makedirs(os.path.dirname(save_cm_path), exist_ok=True)
            fig.savefig(save_cm_path, dpi=200, bbox_inches="tight")
            print(f"💾 Confusion matrix saved to: {save_cm_path}")

        plt.close(fig)

    return accuracy


######
csv_file = "/home/oren/Documents/Models/Order_phase/CLIP_class/test4/sem_clip_results_out2.csv"
base_path = "/home/oren/Documents/Models/Order_phase/CLIP_class/test4/test_images_taged/"

# IMPORTANT: give a save path if you want a file written
out_png = "/home/oren/Documents/Models/Order_phase/CLIP_class/test4/confusion_matrix.png"

calculate_accuracy_with_errors(csv_file, base_path, plot_cm=True, save_cm_path=out_png)
