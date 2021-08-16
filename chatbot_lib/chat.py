import random
import json
import torch
from model_lib.model import NeuralNet
from utils.text_utils import bag_of_words, tokenize
from settings.model_config import intents_file_path, model_info_file_path


def get_response(text):
    """
    This function returns response given text.
    It stems each word using porter stemmer.
    Then text converted into vector based on bag of word method.
    :param text: string sentence
    :return: response text
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(intents_file_path, "r") as f:
        intents = json.load(f)

    model_info = torch.load(model_info_file_path)

    input_size = model_info["input_size"]
    hidden_size = model_info["hidden_size"]
    num_classes = model_info["num_classes"]
    vocabulary = model_info["vocabulary"]
    tags = model_info["tags"]
    model_state = model_info["model_state"]

    model = NeuralNet(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes).to(device)
    model.load_state_dict(model_state)
    model.eval()

    text = tokenize(text)
    X = bag_of_words(text, vocabulary)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(torch.float32)

    output = model(X)

    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probabilities = torch.softmax(output, dim=1)
    probability = probabilities[0][predicted.item()]

    # If predicted label's probability is grater than 0.6 then print random response attached to that tag.
    if probability.item() > 0.60:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return f"{random.choice(intent['responses'])}"

    # Else return "I do not understand"
    else:
        return f"Ne demek istediğini anlamadım."
