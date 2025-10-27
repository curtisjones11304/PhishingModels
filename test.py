# Small-scale testing program for phishing email classifier

from langformers import tasks

# Load the pre-trained classifier model

option = int(input("1. RoBERTa"
      "2. distilBERT"))
classifier = None
if option == 1: classifier = tasks.load_classifier('./langformers-classifier-d20251023-t200514\\best_model') # roberta
elif option == 2: classifier = tasks.load_classifier('./langformers-classifier-d20251024-t212115\\best_model') #distilbert
def main():
    print("=== AI-Powered Phishing Email Confidence Tester ===\n")

    # Prompt user for input fields
    sender = input("Enter the sender email address: ").strip()
    subject = input("Enter the email subject line: ").strip()
    body = input("Enter the email body text: ").strip()

    # Combine input into a single text string
    combined_text = f"From: {sender}\nSubject: {subject}\n\n{body}"

    print("\nProcessing... Please wait.\n")

    # Classify the text (model expects a list)
    result = classifier.classify([combined_text])

    # Print raw output for transparency
    print("Raw model output:", result, "\n")

    # Extract label and probability
    if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
        label = result[0].get("label", "Unknown")
        confidence_score = result[0].get("prob", "N/A")
    else:
        label, confidence_score = "Unknown", "N/A"

    # Display results
    print("=== Result ===")
    print(f"Predicted Label: {label}")
    if isinstance(confidence_score, (float, int)):
        print(f"Confidence Score: {confidence_score:.4f} ({confidence_score * 100:.2f}%)")
    else:
        print(f"Confidence Score: {confidence_score}")
    print("=============================")

if __name__ == "__main__":
    main()
