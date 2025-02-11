# Sentence Variation Generator

This project generates variations of input sentences while maintaining their core meaning, using NLP techniques.

## Setup

### Prerequisites
- Python 3.12
- Virtual Environment (recommended)

### Environment Setup



```bash
python -m venv venv
source venv/bin/activate 
pip install -r requirements.txt
```


## Usage

### Command

```bash
python assessment.py <path_to_json_file>
```

The program takes a single argument: the path to the JSON file to be analyzed.

### Example

```bash
python assessment.py data_sample.json
```

## Output Format

The program generates a JSON file with the following structure:

```json
[
    {
        "id": "0",
        "conversations": [
            {
                "from": "human",
                "value": "Original sentence..."
            },
            {
                "from": "gpt",
                "value": "Original sentence",
                "variation": "original"
            },
            {
                "from": "gpt",
                "value": "First variation",
                "variation": "variation_1"
            },
            // ... more variations
        ]
    }
]
```


## Dependencies

Main dependencies include:
- nltk
- spacy
- textblob
- json

