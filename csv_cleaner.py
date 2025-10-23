# csv_cleaner.py
import csv
import pandas as pd
import os

class CSVEmailCleaner:
    def __init__(self, input_path, output_path="classifier_input.csv"):
        self.input_path = input_path
        self.output_path = output_path
        self.df = None
        self.tag = "\033[92m[csv_cleaner]\033[0m"

    def load_csv(self):
        # Load CSV while handling quoted/multiline fields
        self.df = pd.read_csv(
            self.input_path,
            engine="python",
            quoting=csv.QUOTE_ALL,
            escapechar='\\',
            on_bad_lines='skip'
        )
        self.df = self.df.fillna('').astype(str)
        return self

    def normalize_fields(self):
        # Clean up newline and extra whitespace characters
        self.df = self.df.replace(r'\r\n', ' ', regex=True)
        self.df = self.df.replace(r'\n', ' ', regex=True)
        self.df = self.df.replace(r'\r', ' ', regex=True)
        self.df = self.df.replace(r'\s+', ' ', regex=True)
        return self

    def ensure_columns(self):
        # Check for required columns
        for col in ["sender", "receiver", "subject", "body", "label", "urls"]:
            if col not in self.df.columns:
                self.df[col] = ""
        return self

    def compose_text(self):
        # Create a structured text column combining email fields
        def compose_row(row):
            return (
                f"Sender: {row['sender']} | "
                f"Receiver: {row['receiver']} | "
                f"Subject: {row['subject']} | "
                f"Body: {row['body']} | "
                f"Urls: {row['urls']}"
            )

        self.df["text"] = self.df.apply(compose_row, axis=1)
        return self

    def convert_labels(self):
        # Convert numeric labels into string labels for Langformers
        def label_to_str(x):
            return "phishing" if str(x).strip() == "1" else "legit"
        self.df["label"] = self.df["label"].apply(label_to_str)
        return self

    def convert_urls(self):
        # Convert numeric urls into string urls for Langformers
        def urls_to_str(x):
            return "yes" if str(x).strip() == "1" else "no"
        self.df["urls"] = self.df["urls"].apply(urls_to_str)
        return self

    def save_clean_csv(self):
        # Save the cleaned dataframe into a CSV ready for classification
        df_to_save = self.df[["text", "label"]].copy()
        df_to_save.to_csv(self.output_path, index=False)
        print(f"{self.tag} Saved cleaned classifier CSV to: {self.output_path} (rows: {len(df_to_save)})")
        return self.output_path

    def clean(self):
        # Run the full cleaning pipeline
        print(f"{self.tag} Cleaning CSV file in directory: ({self.input_path})...")
        return (
            self.load_csv()
                .normalize_fields()
                .ensure_columns()
                .compose_text()
                .convert_urls()
                .convert_labels()
                .save_clean_csv()

        )
