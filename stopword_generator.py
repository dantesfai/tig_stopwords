import os
import json
import regex as re
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

class StopwordGenerator:
    """
    A class to generate stopwords for Tigrinya using an IDF-based approach.
    Supports both CSV and TXT file inputs and dynamically detects text columns.
    """

    def __init__(self, corpus_files, config_file="config.json"):
        """
        Initialize the StopwordGenerator with a list of files and configuration.

        Args:
            corpus_files (list): List of file paths (.txt or .csv).
            config_file (str): Path to configuration file for customization.
        """
        self.corpus_files = corpus_files
        self.config = self._load_config(config_file)
        self.text_column = self.config.get('text_column', None)
        self.output_dir = self.config.get('output_dir', "output")
        self.output_filename = self.config.get('output_filename', "stopwords.txt")
        self.plot_filename = self.config.get('plot_filename', "idf_curve.png")  # Added for plot filename
        self.idf_threshold = self.config.get('idf_threshold', 2.5)
        
        self.document_count = 0
        self.word_doc_freq = defaultdict(int)
        self.word_idf_scores = {}
        self.word_freq = Counter()

    def _load_config(self, config_file):
        """Load configuration from a JSON file."""
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file {config_file} not found.")
        
        with open(config_file, "r") as f:
            return json.load(f)

    def _clean_text(self, text):
        """
        Clean and preprocess the text: remove punctuation, normalize spaces, and convert to lowercase.
        """
        text = text.strip()
        text = re.sub(r"[^\p{L}\s]", "", text)  # Keep only letters and spaces
        text = re.sub(r"\s+", " ", text)  # Normalize spaces
        text = re.sub(r"[።፧?!]", " ", text)  # Replace special punctuation with a space
        return text

    def _detect_text_column(self, df):
        """
        Dynamically detects and selects the best text column from a CSV file.
        """
        text_cols = [col for col in df.columns if df[col].dtype == 'object']
        print(text_cols)

        if not text_cols:
            raise ValueError("No text columns found in the CSV file.")

        if len(text_cols) == 1:
            return text_cols[0]

        print("Multiple text columns found. Please choose one:")
        for idx, col in enumerate(text_cols, 1):
            print(f"{idx}. {col}")

        choice = int(input("Enter the number of the text column: ")) - 1
        return text_cols[choice]

    def load_text_from_files(self):
        """
        Reads text content from multiple .txt and .csv files.
        Dynamically selects the correct text column for CSV files.
        """
        documents = []

        for file in self.corpus_files:
            if file.endswith(".txt"):
                with open(file, "r", encoding="utf-8") as f:
                    text = self._clean_text(f.read())
                    documents.append(text)

            elif file.endswith(".csv"):
                df = pd.read_csv(file, encoding="utf-8")

                if self.text_column is None:
                    self.text_column = self._detect_text_column(df)

                if self.text_column not in df.columns:
                    print(f"Warning: Column '{self.text_column}' not found in {file}. Skipping.")
                    continue

                for text in df[self.text_column].dropna():
                    cleaned_text = self._clean_text(str(text))
                    documents.append(cleaned_text)

            else:
                print(f"Skipping unsupported file format: {file}")

        return documents

    def compute_idf(self):
        """
        Compute IDF for each word in the corpus and also count word frequencies.
        """
        documents = self.load_text_from_files()
        self.document_count = len(documents)

        if self.document_count == 0:
            print("No valid documents found.")
            return {}

        for text in documents:
            words = text.split()
            self.word_freq.update(words)  # Count overall word frequency
            unique_words = set(words)  # Unique words for IDF calculation
            for word in unique_words:
                self.word_doc_freq[word] += 1

        # Compute IDF scores
        self.word_idf_scores = {
            word: math.log(self.document_count / (1 + df)) 
            for word, df in self.word_doc_freq.items()
        }

        return self.word_idf_scores

    def plot_idf_curve(self, threshold=1.5):
        """
        Plot an IDF curve to visualize the distribution of word importance with a threshold line.
        """
        if not self.word_idf_scores:
            print("No IDF scores available. Run compute_idf() first.")
            return

        sorted_idfs = sorted(self.word_idf_scores.values())
        x_values = np.arange(len(sorted_idfs))

        # Ensure the output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Plot the IDF curve and save the plot
        plt.figure(figsize=(10, 5))
        plt.plot(x_values, sorted_idfs, marker="o", linestyle="-", color="b", markersize=3)
        plt.axhline(y=threshold, color="r", linestyle="--", label=f"Threshold ({threshold})")
        plt.xlabel("Word Index (Sorted by IDF)")
        plt.ylabel("Inverse Document Frequency (IDF)")
        plt.title("IDF Curve for Tigrinya Words with Threshold")
        plt.legend()

        # Save the plot
        plot_path = os.path.join(self.output_dir, self.plot_filename)
        plt.savefig(plot_path)
        plt.close()

        print(f"IDF curve saved to {plot_path}")

    def generate_stop_words(self, threshold=2.5):
        """
        Select stop words based on a given IDF threshold and sort by frequency.
        """
        self.compute_idf()
        stop_words = {word for word, idf in self.word_idf_scores.items() if idf < threshold}
        sorted_stop_words = sorted(stop_words, key=lambda word: self.word_freq[word], reverse=True)
        return [(word, self.word_freq[word]) for word in sorted_stop_words]

    def save_stop_words(self):
        """
        Saves the sorted stop words list with their frequency to a text file.
        """
        # Ensure the output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        output_file = os.path.join(self.output_dir, self.output_filename)
        sorted_stop_words = self.generate_stop_words(self.idf_threshold)

        with open(output_file, "w", encoding="utf-8") as file:
            for word, freq in sorted_stop_words:
                file.write(f"{word}\t{freq}\n")

        print(f"Stop words saved to {output_file}")

# Example CLI or usage
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate Stopwords for Tigrinya using IDF")
    parser.add_argument("corpus_files", help="List of text or CSV files", nargs="+")
    parser.add_argument("-c", "--config", help="Path to config file", default="config.json")
    parser.add_argument("-o", "--output", help="Output file for stopwords", default="output/stopwords.txt")
    parser.add_argument("-t", "--threshold", type=float, help="Threshold for IDF to identify stopwords", default=2.5)
    args = parser.parse_args()

    stopword_generator = StopwordGenerator(corpus_files=args.corpus_files, config_file=args.config)
    stopword_generator.save_stop_words()
    stopword_generator.plot_idf_curve(threshold=args.threshold)
