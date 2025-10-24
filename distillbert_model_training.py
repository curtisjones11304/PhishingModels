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
    "max_length": 256,
    "num_train_epochs": 7,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "lr_scheduler_type": "cosine",
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 32,
    "gradient_accumulation_steps": 2,
    "fp16": True,
    "eval_strategy": "steps",
    "early_stopping_patience": 3,
    "load_best_model_at_end": True,
    "metric_for_best_model": "f1_macro",
    "logging_steps": 50,
    "save_steps": 100,
    "report_to": ["tensorboard"]
}

print("Training Model with Langformers...")

# Create classifier using DistilBERT
model = tasks.create_classifier(
    model_name="distilbert-base-uncased",
    csv_path=clean_csv_path,
    text_column="text",       # adjust column names
    label_column="label",
    training_config=training_config
)

# Start fine-tuning. This will run training and save the best model to disk.
# Training progress will be printed and logged according to training_config.
model.train()

# After training finishes, the trained model directory will be available (see logs/output).
# You can also load it for inference with tasks.load_classifier()
print("Training finished. Use tasks.load_classifier('/path/to/saved/model') to load and predict.")

# Start training
model.train()