import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from collections import Counter
from heapq import nlargest
from transformers import BartTokenizer, BartForConditionalGeneration


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


if __name__ == "__main__":
    # Load SpaCy model
    nlp = load_spacy_model()

    # Prompt user for input text
    print("Enter the text you want to summarize (press Enter twice to finish):")
    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)
    sample_text = "\n".join(lines)

    # Check if text is provided
    if not sample_text.strip():
        print("No text provided. Exiting...")
    else:
        # Extractive Summarization
        print("\nExtractive Summary:")
        print(extractive_summarization(sample_text, nlp))

        # Abstractive Summarization
        print("\nAbstractive Summary:")
        print(abstractive_summarization(sample_text))
