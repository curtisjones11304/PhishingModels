# train_classifier.py
import os
from langformers import tasks
from csv_cleaner import CSVEmailCleaner

# Clean CSV
csv_path = "CEAS_08.csv"
clean_csv_path = "classifier_input.csv"  # will be created for Langformers

# Run cleaning process
clean_csv_path = CSVEmailCleaner(csv_path, clean_csv_path).clean()

# Langformers training configuration
training_config = {
    "max_length": 384,  # longer email bodies
    "num_train_epochs": 3,
    "per_device_train_batch_size": 12,  # fits RTX 2070 Super with gradient accumulation
    "per_device_eval_batch_size": 24,
    "learning_rate": 2e-5,
    "gradient_accumulation_steps": 2,  # simulates larger batch
    "fp16": True,  # mixed precision for speed & memory
    "save_total_limit": 2,
    "save_steps": 500,
    "logging_steps": 100,
    "eval_strategy": "steps",
    "test_size": 0.2,
    "val_size": 0.1,
    "metric_for_best_model": "f1_macro",
    "early_stopping_patience": 3,
    "output_dir": "trained_models/roberta-phishing"
}

}
print("Training Model with Langformers...")

# Create classifier: it will automatically detect classes and fine-tune the model.
# text_column defaults to "text", label_column defaults to "label".
model = tasks.create_classifier(
    model_name="roberta-base",     # Hugging Face model name
    csv_path=clean_csv_path,       # cleaned CSV created above
    text_column="text",
    label_column="label",
    training_config=training_config
)

# Start fine-tuning. This will run training and save the best model to disk.
# Training progress will be printed and logged according to training_config.
model.train()

# After training finishes, the trained model directory will be available (see logs/output).
# You can also load it for inference with tasks.load_classifier()
print("Training finished. Use tasks.load_classifier('/path/to/saved/model') to load and predict.")
