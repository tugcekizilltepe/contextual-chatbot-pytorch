import json
import numpy as np
from utils.text_utils import tokenize, stem, bag_of_words
from model import NeuralNet
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from settings.model_config import hyperparameters, model_info_file_path,  intents_file_path


class ChatDataset(Dataset):
    def __init__(self, X_train, y_train):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


def train_model(intents_file_path):
    with open(intents_file_path, "r") as f:
        intents = json.load(f)

    vocabulary = []
    tags = []
    patterns_and_tags = []

    for intent in intents['intents']:
        tag = intent['tag']
        tags.append(tag)
        for pattern in intent['patterns']:
            tokenized_pattern = tokenize(pattern)
            vocabulary.extend(tokenized_pattern)
            patterns_and_tags.append((tokenized_pattern, tag))

    ignore_words = ['?', ".", ","]
    vocabulary = [stem(w) for w in vocabulary if w not in ignore_words]
    vocabulary = sorted(set(vocabulary))
    tags = sorted(set(tags))

    X_train = []
    y_train = []

    for (pattern_sentence, tag) in patterns_and_tags:
        vectorized_pattern = bag_of_words(pattern_sentence, vocabulary)
        X_train.append(vectorized_pattern)
        label = tags.index(tag)
        y_train.append(label)  # CrossEntropyLoss

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    dataset = ChatDataset(X_train, y_train)
    train_loader = DataLoader(dataset=dataset, batch_size=hyperparameters['batch_size'],
                              shuffle=hyperparameters['shuffle'], num_workers=hyperparameters['num_workers'])

    input_size = len(X_train[0])
    num_classes = len(tags)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = NeuralNet(input_size=input_size, hidden_size=hyperparameters['hidden_size'], num_classes=num_classes)
    model = train_loop(model, train_loader, device, learning_rate=hyperparameters['learning_rate'],
                       num_epochs=hyperparameters['num_epochs'])

    model_info = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "num_classes": num_classes,
        "hidden_size": hyperparameters['hidden_size'],
        "vocabulary": vocabulary,
        "tags": tags
    }

    torch.save(model_info, model_info_file_path)


def train_loop(model, train_loader, device, learning_rate, num_epochs):
    # loss and optimizer

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    for epoch in range(num_epochs):
        for (words, labels) in train_loader:
            words = words.to(dtype=torch.float32).to(device)
            labels = labels.to(dtype=torch.long).to(device)
            y_pred = model(words)
            loss = criterion(y_pred, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model

train_model(intents_file_path)