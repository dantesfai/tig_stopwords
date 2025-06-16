# Stopwords Generator for Tigrinya

## Purpose

**Stopword Generator** is a Python tool for generating stopwords in **Tigrinya** text using an **Inverse Document Frequency (IDF)**-based approach. This tool can handle both **CSV** and **TXT** file formats and supports preprocessing text, removing non-informative words, and visualizing word importance through IDF scores.

### Features

- **Multi-file support**: Process both **TXT** and **CSV** files.
- **Customizable settings**: Control how text is processed via a **config.json** file.
- **IDF-based stopword generation**: Identifies and removes common words based on their IDF scores.
- **Visualization**: Generate IDF plots to visualize the distribution of word importance.
- **Configurable IDF threshold**: Control which words are considered stopwords using the IDF threshold.
- **Command-Line Interface (CLI)**: A simple CLI for ease of use.

## Installation

### Prerequisites

Ensure that you have **Python 3.x** installed. You'll also need to install the required dependencies via **pip**.

### Steps to Install

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/yourusername/stopword-generator.git
   ```

2. Navigate to the project directory:

    ```bash
    cd stopword-generator 
    ```

3. Install required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. (Optional) If you want to run tests, you can install pytest:

   ```bash
    pip install pytest
    ```

## Usage
### Command-Line Interface (CLI)

  To run the stopword generator on your text files:

    ```bash
 python stopword_generator.py data/input.txt -c config.json -o output/stopwords.txt -t 2.5
 ```
**Arguments**:

- data/input.txt: Path to the input file (either .txt or .csv file).
- -c config.json: Path to the configuration file (default is config.json).
- -o output/stopwords.txt: Path where the stopwords file will be saved (default is stopwords.txt).
- -t 2.5: Threshold for IDF to identify stopwords (default is 2.5).

### Programmatic Usage
   ```python
 
    from stopword_generator import StopwordGenerator

    # Initialize the Stopword Generator
    stopword_gen = StopwordGenerator(corpus_files=["data/input.txt"])

    # Generate stopwords
    stopwords = stopword_gen.generate_stop_words(threshold=2.5)

    # Print the top 10 stopwords
    for word, freq in stopwords[:10]:
        print(f"{word}: {freq}")

    # Save stopwords to a file
    stopword_gen.save_stop_words(output_file="stopwords.txt", threshold=2.5)
```

## Configuration
You can customize the cleaning process and output settings by modifying the config.json file.
    ```
    Example config.json:

    json 
    {
        "text_column": "text",  // Name of the text column in CSV files (if applicable)
        "output_dir": "output", // Directory where stopwords will be saved
        "output_filename": "stopwords.txt", // Output file name
        "idf_threshold": 2.5  // Threshold for IDF-based stopword generation
    }
    ```

## Performance Optimization
If you're working with very large files, consider the following optimizations:

**File Streaming**: If the input files are large, modify the tool to process the file line-by-line or in chunks. This will prevent memory overload and allow the tool to handle large datasets efficiently.
**Parallel Processing**: For large corpora or if the cleaning process is computationally intensive, you can use multiprocessing or concurrent.futures to speed up processing.

## Tests
To ensure the tool works correctly, unit tests are provided. You can run the tests using **pytest**:

    ```bash
pytest
```
This will run all the tests defined in the tests/ directory, ensuring that the tool functions as expected.

## License
This tool is open-source and free to use for research purposes under the MIT License. See the LICENSE file for details.

## Contributing
Contributions are welcome! If you have improvements, bug fixes, or new features to contribute, feel free to fork this repository, make your changes, and submit a pull request.

## Acknowledgements
Thanks to the NLP community for open-source contributions and resources that helped guide the development of this tool.
Special thanks to contributors and researchers working on Tigrinya language resources, whose work helped shape this project. 
