"""Train and run a Bio_ClinicalBERT specialty classifier.

Training data must be de-identified and stored as CSV with two columns:
``text,specialty``. The text should contain the completed patient case and the
specialty column should contain exactly one target specialty per row.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
TEXT_COLUMN = "text"
LABEL_COLUMN = "specialty"


@dataclass(frozen=True)
class SpecialtyPrediction:
    specialty: str
    probability: float
    probabilities: Dict[str, float]


def build_case_text(
    *,
    complaint: str,
    symptoms: str,
    location: str,
    duration: str,
    severity: str,
    history: str = "",
    medications: str = "",
    test_results: str = "",
) -> str:
    """Create the de-identified text representation used by the classifier."""
    fields = {
        "chief complaint": complaint,
        "symptoms": symptoms,
        "location": location,
        "duration": duration,
        "severity": severity,
        "relevant history": history,
        "medications": medications,
        "test results": test_results,
    }
    return "\n".join(f"{name}: {value.strip()}" for name, value in fields.items() if value and value.strip())


def load_labeled_cases(csv_path: str | Path) -> tuple[List[str], List[str]]:
    """Load and validate de-identified training cases from a CSV file."""
    texts: List[str] = []
    labels: List[str] = []
    with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {TEXT_COLUMN, LABEL_COLUMN}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("Training CSV must contain 'text' and 'specialty' columns.")
        for row_number, row in enumerate(reader, start=2):
            text = (row.get(TEXT_COLUMN) or "").strip()
            label = (row.get(LABEL_COLUMN) or "").strip()
            if not text or not label:
                raise ValueError(f"Row {row_number} must contain both text and specialty.")
            texts.append(text)
            labels.append(label)

    if len(set(labels)) < 2:
        raise ValueError("Training data must contain at least two specialty labels.")
    return texts, labels


def train_classifier(
    data_path: str | Path,
    output_dir: str | Path,
    *,
    model_name: str = MODEL_NAME,
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    max_length: int = 256,
    test_size: float = 0.2,
    seed: int = 42,
) -> Dict[str, float]:
    """Fine-tune Bio_ClinicalBERT and save a specialty classifier checkpoint."""
    try:
        import numpy as np
        import torch
        from sklearn.metrics import accuracy_score, f1_score
        from sklearn.model_selection import train_test_split
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Training requires torch, transformers, scikit-learn, and numpy. "
            "Install requirements.txt first."
        ) from exc

    texts, labels = load_labeled_cases(data_path)
    label_names = sorted(set(labels))
    label_to_id = {label: index for index, label in enumerate(label_names)}
    numeric_labels = np.array([label_to_id[label] for label in labels])

    train_texts, eval_texts, train_labels, eval_labels = train_test_split(
        texts,
        numeric_labels,
        test_size=test_size,
        random_state=seed,
        stratify=numeric_labels,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_names),
        id2label={index: label for label, index in label_to_id.items()},
        label2id=label_to_id,
    )

    class CaseDataset(torch.utils.data.Dataset):
        def __init__(self, case_texts: Sequence[str], case_labels: Sequence[int]):
            self.encodings = tokenizer(
                list(case_texts),
                truncation=True,
                padding=True,
                max_length=max_length,
            )
            self.labels = list(case_labels)

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
            item = {key: torch.tensor(value[index]) for key, value in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
            return item

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    train_dataset = CaseDataset(train_texts, train_labels)
    eval_dataset = CaseDataset(eval_texts, eval_labels)

    def compute_metrics(eval_prediction):
        predictions = np.argmax(eval_prediction.predictions, axis=-1)
        return {
            "accuracy": accuracy_score(eval_prediction.label_ids, predictions),
            "macro_f1": f1_score(eval_prediction.label_ids, predictions, average="macro", zero_division=0),
        }

    training_args = TrainingArguments(
        output_dir=str(output_path),
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        seed=seed,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    (output_path / "labels.json").write_text(json.dumps(label_names, indent=2), encoding="utf-8")
    return {key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))}


def predict_specialty(
    case_text: str,
    checkpoint_dir: str | Path,
    *,
    max_length: int = 256,
) -> SpecialtyPrediction:
    """Return one specialty and softmax probabilities from a trained checkpoint."""
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Prediction requires torch and transformers.") from exc

    checkpoint_path = Path(checkpoint_dir)
    if not case_text.strip():
        raise ValueError("case_text must contain the completed patient case.")
    if not (checkpoint_path / "labels.json").exists():
        raise FileNotFoundError(f"No trained classifier found in {checkpoint_path}.")

    labels = json.loads((checkpoint_path / "labels.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint_path))
    model.eval()
    encoded = tokenizer(case_text, return_tensors="pt", truncation=True, max_length=max_length)
    with torch.inference_mode():
        probabilities = torch.softmax(model(**encoded).logits, dim=-1)[0].tolist()

    probability_map = {label: float(probabilities[index]) for index, label in enumerate(labels)}
    best_label = max(probability_map, key=probability_map.get)
    return SpecialtyPrediction(best_label, probability_map[best_label], probability_map)


__all__ = [
    "MODEL_NAME",
    "SpecialtyPrediction",
    "build_case_text",
    "load_labeled_cases",
    "predict_specialty",
    "train_classifier",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Bio_ClinicalBERT for specialty prediction.")
    parser.add_argument("--data", required=True, help="CSV file with text,specialty columns")
    parser.add_argument("--output", default="models/specialty_classifier", help="Checkpoint output directory")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    print(train_classifier(args.data, args.output, epochs=args.epochs, batch_size=args.batch_size))
