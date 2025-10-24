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
    "max_length": 128,
    "num_train_epochs": 5,
    "learning_rate": 3e-5,
    "per_device_train_batch_size": 16,   # safer for 8GB VRAM
    "per_device_eval_batch_size": 32,
    "fp16": True,                        # mixed precision
    "gradient_accumulation_steps": 2,    # effective batch size 32
    "eval_strategy": "steps",
    "logging_steps": 50,
    "save_steps": 100,
    "early_stopping_patience": 3,
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