import nltk
from nltk.corpus import wordnet
import spacy
import random
import re
from textblob import TextBlob
from nltk.corpus import words
from nltk.corpus import brown
import json
import sys

def setup_nltk():
    """Download required NLTK resources"""
    try:
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('wordnet')
        nltk.download('words')
        nltk.download('brown')
    except Exception as e:
        print(f"Error downloading NLTK resources: {e}")
        raise

def load_spacy_model():
    """Load spaCy model with error handling"""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading spaCy model...")
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        return spacy.load("en_core_web_sm")

# Load NLP model
nlp = load_spacy_model()

# Example sentence
sentence = """No, the robot failed at the tasks because it failed to rotate the laptop lid cover to close"""

# Create a set of common English words
common_words = set(w.lower() for w in brown.words())
common_words.update(['no', 'the', 'it', 'because', 'failed', 'to', 'onto', 'at'])

def is_common_word(word):
    """Check if a word is a common English word"""
    return word.lower() in common_words

def filter_uncommon_words(sentence):
    """Replace uncommon words with their original counterparts"""
    original_words = sentence.split()
    new_words = []
    
    for orig_word in original_words:
        if not is_common_word(orig_word.lower()) and orig_word.isalpha():
            # Find the corresponding word in the original sentence
            orig_index = original_words.index(orig_word)
            if orig_index < len(sentence.split()):
                new_words.append(sentence.split()[orig_index])
        else:
            new_words.append(orig_word)
    
    return ' '.join(new_words)

# Method 1: Synonym replacement
def synonym_replacement(sentence):
    """Enhanced synonym replacement with more variations"""
    words = nltk.word_tokenize(sentence)
    new_sentence = []
    for word in words:
        syns = wordnet.synsets(word)
        if syns and random.random() > 0.5:  # Increased replacement probability
            # Get all unique lemmas from all synsets
            synonyms = []
            for syn in syns:
                for lemma in syn.lemmas():
                    if lemma.name() != word and '_' not in lemma.name():
                        synonyms.append(lemma.name())
            if synonyms:
                new_sentence.append(random.choice(synonyms))
                continue
        new_sentence.append(word)
    return " ".join(new_sentence)

# Method 2: Syntactic transformation
def syntactic_transformation(sentence):
    """Enhanced syntactic transformation with multiple patterns"""
    doc = nlp(sentence)
    pattern = random.choice(['passive', 'subject_move', 'clause_reorder'])
    
    if pattern == 'passive':
        # Convert to passive voice using TextBlob
        try:
            return str(TextBlob(sentence).parse().to_passive())
        except:
            pass
    
    elif pattern == 'subject_move':
        # Move subject to different position
        words = [token.text for token in doc]
        for token in doc:
            if token.dep_ == "nsubj":
                words.remove(token.text)
                insert_pos = random.randint(0, len(words))
                words.insert(insert_pos, token.text)
                return " ".join(words)
    
    else:  # clause_reorder
        # Reorder clauses if multiple exist
        clauses = sentence.split("because")
        if len(clauses) > 1:
            return "Because" + clauses[1].strip() + "," + clauses[0].strip()
    
    return sentence

# Method 3: Sentence splitting
def split_sentence(sentence):
    return ". ".join(nltk.tokenize.sent_tokenize(sentence))

# Method 4: Tense transformation
def tense_transformation(sentence):
    return re.sub(r"\bis\b", "was", sentence)

# Method 5: TextBlob transformation
def textblob_transformation(sentence):
    """Enhanced TextBlob transformation with more variations"""
    blob = TextBlob(sentence)
    transformation = random.choice(['passive', 'plural', 'sentiment'])
    
    try:
        if transformation == 'passive':
            return str(blob.parse().to_passive())
        
        elif transformation == 'plural':
            words = blob.words
            new_words = []
            for word in words:
                if random.random() > 0.7:
                    try:
                        new_word = word.pluralize() if random.random() > 0.5 else word.singularize()
                        new_words.append(str(new_word))
                    except:
                        new_words.append(str(word))
                else:
                    new_words.append(str(word))
            return ' '.join(new_words)
        
        else:  # sentiment adjustment
            words = blob.words
            if blob.sentiment.polarity < 0:
                words.insert(0, random.choice(['Regrettably,', 'Unfortunately,']))
            return ' '.join([str(w) for w in words])
            
    except:
        return sentence

def add_clarification(sentence):
    """Add clarifying phrases to the sentence"""
    clarifications = [
        "in this specific case",
        "as observed",
        "unfortunately",
        "specifically",
        "to be precise",
        "evidently",
    ]
    words = sentence.split()
    insert_pos = random.randint(1, len(words) - 1)
    words.insert(insert_pos, random.choice(clarifications))
    return " ".join(words)

def add_emphasis(sentence):
    """Add emphasis words to the sentence"""
    emphasis = [
        "really",
        "very",
        "quite",
        "particularly",
        "especially",
        "notably",
        "certainly"
    ]
    words = sentence.split()
    insert_pos = random.randint(1, len(words) - 1)
    words.insert(insert_pos, random.choice(emphasis))
    return " ".join(words)

def change_perspective(sentence):
    """Change perspective of the sentence"""
    perspectives = [
        ("i think", "beginning"),
        ("in my view", "beginning"),
        ("from what i can see", "beginning"),
        ("it seems that", "beginning"),
        ("apparently", "beginning"),
        ("in this case", "beginning"),
        ("based on this", "beginning")
    ]
    perspective, position = random.choice(perspectives)
    if position == "beginning":
        return f"{perspective} {sentence}"
    return sentence

def add_condition(sentence):
    """Add conditional phrases to the sentence"""
    conditions = [
        "in this situation",
        "under these circumstances",
        "in this context",
        "given these conditions",
        "in this particular case"
    ]
    return f"{random.choice(conditions)}, {sentence}"

def reorder_phrases(sentence):
    """Reorder phrases in the sentence if possible"""
    doc = nlp(sentence)
    # Split into phrases
    phrases = []
    current_phrase = []
    
    for token in doc:
        current_phrase.append(token.text)
        if token.text in [',', 'and', 'or', 'but']:
            phrases.append(' '.join(current_phrase))
            current_phrase = []
    
    if current_phrase:
        phrases.append(' '.join(current_phrase))
    
    if len(phrases) > 1:
        random.shuffle(phrases)
        return ' '.join(phrases)
    return sentence

def process_json_file(file_path):
    """Process the JSON file and generate variations for each conversation"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        results = []
        variation_id = 0
        
        for item in data:
            if 'conversations' in item and len(item['conversations']) >= 2:
                sentence = item['conversations'][1]['value']
                
                # Generate variations
                variations = set()
                transformations = [
                    (synonym_replacement, 0.25),
                    (syntactic_transformation, 0.15),
                    (split_sentence, 0.1),
                    (tense_transformation, 0.1),
                    (textblob_transformation, 0.1),
                    (add_clarification, 0.1),
                    (add_emphasis, 0.05),
                    (change_perspective, 0.05),
                    (add_condition, 0.05),
                    (reorder_phrases, 0.05)
                ]
                
                max_attempts = 50
                attempts = 0
                
                while len(variations) < 15 and attempts < max_attempts:
                    attempts += 1
                    result = sentence
                    num_transforms = random.randint(1, 3)
                    
                    for _ in range(num_transforms):
                        func = random.choices(
                            [f[0] for f in transformations],
                            weights=[f[1] for f in transformations],
                            k=1
                        )[0]
                        result = func(result)
                    
                    result = filter_uncommon_words(result).lower()
                    if result != sentence.lower() and len(result.split()) <= len(sentence.split()) * 2:
                        variations.add(result)
                
                # Create item with variations
                result_item = {
                    "id": str(variation_id),
                    "conversations": [
                        {
                            "from": "human",
                            "value": "Generate variations of the following sentence: " + sentence
                        },
                        {
                            "from": "gpt",
                            "value": sentence,
                            "variation": "original"
                        }
                    ]
                }
                
                # Add variations with numbered keys
                for i, variation in enumerate(variations, 1):
                    result_item["conversations"].append({
                        "from": "gpt",
                        "value": variation,
                        "variation": f"variation_{i}"
                    })
                
                # Pad with remaining variations if less than 15
                while len(result_item["conversations"]) < 17:  # 1 human + 1 original + 15 variations
                    i = len(result_item["conversations"]) - 1
                    result_item["conversations"].append({
                        "from": "gpt",
                        "value": sentence,  # Use original sentence as fallback
                        "variation": f"variation_{i}"
                    })
                
                results.append(result_item)
                variation_id += 1
                
        # Save results to JSON file
        with open('result.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        
        print(f"\nResults saved to result.json with {len(results)} items")
                
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
    except json.JSONDecodeError:
        print(f"Error: File '{file_path}' is not a valid JSON file")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def main():
    # Check if file path is provided
    if len(sys.argv) != 2:
        print("Usage: python assessment.py <path_to_json_file>")
        sys.exit(1)
    
    # Setup resources
    setup_nltk()
    nlp = load_spacy_model()
    
    # Process the JSON file
    file_path = sys.argv[1]
    process_json_file(file_path)

if __name__ == "__main__":
    main()