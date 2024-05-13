from flask import Flask, request, jsonify
import torch
from transformers import BertForTokenClassification, AutoTokenizer
import json
from collections import Counter
import torch.nn.functional as F

app = Flask(__name__)


# Load the label mappings from config
config = json.load(open("ner_bert_model/config.json"))
color_mapping = {'tag1': '\033[94m', 'tag2': '\033[92m', 'tag3': '\033[91m', 'O': '\033[0m'}  # Update accordingly

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("./tokenizer")
model = BertForTokenClassification.from_pretrained("./ner_bert_model")
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

ids2tags ={0: 'O', 1: 'B-GEO', 2: 'B-GPE', 3: 'B-PER', 4: 'I-GEO', 5: 'B-ORG', 6: 'I-ORG', 7: 'B-TIM', 8: 'B-ART', 9: 'I-ART', 10: 'I-PER', 11: 'I-GPE', 12: 'I-TIM', 13: 'B-NAT', 14: 'B-EVE', 15: 'I-EVE', 16: 'I-NAT'} 

def get_predicted_tag(word, tags, scores):
    most_common_tag = Counter(tags).most_common()[0][0]
    correct_scores = [score if tag == most_common_tag else -score for score, tag in zip(scores, tags)]
    score = round(sum(correct_scores) / len(correct_scores), 4)
    predicted_tag = ids2tags[most_common_tag].split("-")[-1]
    return {"word": word, "entity_group": predicted_tag, "score": score}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']
    display_formatted_text = data.get('display', True)

    # Call the new predict function
    predictions = custom_predict(text, display_formatted_text)
    return jsonify(predictions)

def custom_predict(raw_text, display_formatted_text=True):
    raw_words = raw_text.split()
    encoded_dict = tokenizer(raw_words, is_split_into_words=True, add_special_tokens=True,
                             return_attention_mask=True, return_tensors='pt')

    input_ids = encoded_dict['input_ids'][0].unsqueeze(0).to(device)
    input_mask = encoded_dict['attention_mask'][0].unsqueeze(0).to(device)

    output = model(input_ids, token_type_ids=None, attention_mask=input_mask)
    normalized_output = F.softmax(output.logits.detach().cpu(), dim=2)
    predictions = torch.max(normalized_output, 2)
    predicted_labels = predictions.indices.numpy().flatten()
    predicted_scores = predictions.values.numpy().flatten()

    result = []
    prev_token_id = None
    tags = []
    scores = []

    for token_id, predicted_label, score in zip(encoded_dict.word_ids(), predicted_labels, predicted_scores):
        if token_id is None:
            continue
        if token_id != prev_token_id:
            if prev_token_id is not None:
                result.append(get_predicted_tag(raw_words[prev_token_id], tags, scores))
            tags = []
            scores = []
        tags.append(predicted_label)
        scores.append(score)
        prev_token_id = token_id

    if prev_token_id is not None:
        result.append(get_predicted_tag(raw_words[prev_token_id], tags, scores))

    if display_formatted_text:
        formatted_text = " ".join([color_mapping[entity["entity_group"]] + entity["word"] for entity in result]) + color_mapping['O']
        print(formatted_text)

    return result

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
