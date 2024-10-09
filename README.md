# News Article Translation and Analysis

This Python script fetches articles from specified URLs, translates their content using EasyNMT and Google Translate, and combines the translated results into a single CSV file. The script utilizes the `NewsPlease` library to extract articles and provides easy-to-use functionality for analyzing news content across multiple languages.

## Features

- **Article Retrieval**: Automatically fetches articles from a list of URLs using the `NewsPlease` library.
- **Multi-Language Translation**: Translates articles into English using both EasyNMT and Google Translate.
- **Data Storage**: Saves the original and translated articles in CSV format for easy access and analysis.
- **Data Merging**: Merges the translation results from multiple methods into a single comprehensive dataset.

## Requirements

To run this script, you will need the following libraries:

- `pandas`
- `news-please`
- `googletrans`
- `easynmt`
- `torch`

You can install these libraries using pip. Uncomment the installation commands in the script or run the following commands in your terminal:

```bash
pip install googletrans==4.0.0-rc1
pip install news-please
pip install -U easynmt
