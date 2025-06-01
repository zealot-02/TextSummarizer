import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from collections import Counter
from heapq import nlargest
from transformers import BartTokenizer, BartForConditionalGeneration
import tkinter as tk
from tkinter import scrolledtext, messagebox


def load_spacy_model():
    """Load and return the SpaCy language model."""
    return spacy.load('en_core_web_lg')


def extractive_summarization(text, nlp_model, top_n=3):
    """
    Perform extractive summarization using SpaCy.

    Args:
        text (str): The input text to summarize.
        nlp_model: Loaded SpaCy NLP model.
        top_n (int): Number of top sentences to include in the summary.

    Returns:
        str: Extractive summary of the text.
    """
    doc = nlp_model(text)
    stopwords = list(STOP_WORDS)
    pos_tags = {'ADJ', 'PROPN', 'NOUN', 'VERB'}

    # Extract keywords
    keywords = [
        token.text for token in doc
        if token.text not in stopwords and token.text not in punctuation and token.pos_ in pos_tags
    ]

    # Calculate word frequencies
    word_frequencies = Counter(keywords)
    max_frequency = max(word_frequencies.values(), default=1)
    word_frequencies = {word: freq / max_frequency for word, freq in word_frequencies.items()}

    # Calculate sentence scores
    sentence_scores = {}
    for sent in doc.sents:
        for word in sent:
            if word.text in word_frequencies:
                sentence_scores[sent] = sentence_scores.get(sent, 0) + word_frequencies[word.text]

    # Select top sentences
    summarized_sentences = nlargest(top_n, sentence_scores, key=sentence_scores.get)
    return ' '.join([sent.text for sent in summarized_sentences])


def abstractive_summarization(text, model_name='facebook/bart-large-cnn', max_length=100):
    """
    Perform abstractive summarization using BART.

    Args:
        text (str): The input text to summarize.
        model_name (str): Hugging Face model name.
        max_length (int): Maximum length of the summary.

    Returns:
        str: Abstractive summary of the text.
    """
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)

    # Tokenize input text
    inputs = tokenizer(
        text, return_tensors="pt", max_length=1024, truncation=True
    )

    # Generate summary
    summary_ids = model.generate(
        inputs['input_ids'], num_beams=4, max_length=max_length, early_stopping=True
    )
    return tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)


# GUI Implementation
def summarize_text():
    text = input_text.get("1.0", tk.END).strip()
    if not text:
        messagebox.showerror("Error", "Please enter some text to summarize.")
        return

    try:
        if summarization_method.get() == "Extractive":
            summary = extractive_summarization(text, nlp, top_n=3)
        elif summarization_method.get() == "Abstractive":
            summary = abstractive_summarization(text)
        else:
            summary = "Invalid summarization method selected."
    except Exception as e:
        summary = f"Error during summarization: {str(e)}"

    output_text.delete("1.0", tk.END)
    output_text.insert(tk.END, summary)


# Load SpaCy model
nlp = load_spacy_model()

# Initialize GUI
root = tk.Tk()
root.title("Text Summarization Tool")
root.geometry("800x600")

# Input Section
tk.Label(root, text="Input Text", font=("Arial", 14)).pack(pady=5)
input_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=90, height=10, font=("Arial", 12))
input_text.pack(padx=10, pady=5)

# Summarization Method Selection
tk.Label(root, text="Select Summarization Method", font=("Arial", 14)).pack(pady=5)
summarization_method = tk.StringVar(value="Extractive")
tk.Radiobutton(root, text="Extractive", variable=summarization_method, value="Extractive", font=("Arial", 12)).pack()
tk.Radiobutton(root, text="Abstractive", variable=summarization_method, value="Abstractive", font=("Arial", 12)).pack()

# Summarize Button
summarize_button = tk.Button(root, text="Summarize", command=summarize_text, font=("Arial", 14), bg="blue", fg="white")
summarize_button.pack(pady=10)

# Output Section
tk.Label(root, text="Summary", font=("Arial", 14)).pack(pady=5)
output_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=90, height=10, font=("Arial", 12))
output_text.pack(padx=10, pady=5)

# Run the application
root.mainloop()
