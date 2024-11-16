import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
import time
from tqdm import tqdm
import json
from argparse import ArgumentParser

unk = '<UNK>'

class FFNN(nn.Module):
    def __init__(self, input_dim, h):
        super(FFNN, self).__init__()
        self.h = h
        self.W1 = nn.Linear(input_dim, h)
        self.activation = nn.ReLU()  # The rectified linear unit; one valid choice of activation function
        self.output_dim = 5
        self.W2 = nn.Linear(h, self.output_dim)
        self.softmax = nn.LogSoftmax(dim=-1)  # Adjust dimension to match last dimension of z
        self.loss = nn.NLLLoss()  # Cross-entropy/negative log likelihood loss

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, input_vector):
        # First hidden layer
        h = self.W1(input_vector)  # Linear transformation
        h = self.activation(h)     # Apply ReLU activation

        # Output layer
        z = self.W2(h)             # Linear transformation to output space

        # Apply LogSoftmax for probability distribution
        predicted_vector = self.softmax(z)

        return predicted_vector


def make_vocab(data):
    vocab = set()
    for document, _ in data:
        for word in document:
            vocab.add(word)
    return vocab 


def make_indices(vocab):
    vocab_list = sorted(vocab)
    vocab_list.append(unk)
    word2index = {}
    index2word = {}
    for index, word in enumerate(vocab_list):
        word2index[word] = index 
        index2word[index] = word 
    vocab.add(unk)
    return vocab, word2index, index2word 


def convert_to_vector_representation(data, word2index):
    vectorized_data = []
    for document, y in data:
        vector = torch.zeros(len(word2index)) 
        for word in document:
            index = word2index.get(word, word2index[unk])
            vector[index] += 1
        vectorized_data.append((vector, y))
    return vectorized_data


def load_data(train_data, val_data, test_data=None):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)

    # Preprocess the data and adjust stars to range [0, 4]
    tra = [(elt["text"].split(), int(elt["stars"] - 1)) for elt in training]
    val = [(elt["text"].split(), int(elt["stars"] - 1)) for elt in validation]

    # Load test data if provided and adjust stars
    test = []
    if test_data:
        with open(test_data) as test_f:
            test_json = json.load(test_f)
        test = [(elt["text"].split(), int(elt["stars"] - 1)) for elt in test_json]
    
    return tra, val, test


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required=True, help="hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required=True, help="num of epochs to train")
    parser.add_argument("--train_data", required=True, help="path to training data")
    parser.add_argument("--val_data", required=True, help="path to validation data")
    parser.add_argument("--test_data", default=None, help="path to test data")  # Set default to None
    parser.add_argument('--do_train', action='store_true')
    args = parser.parse_args()

    # Fix random seeds
    random.seed(42)
    torch.manual_seed(42)

    # Load data
    print("========== Loading data ==========")
    train_data, valid_data, test_data = load_data(args.train_data, args.val_data, args.test_data)
    vocab = make_vocab(train_data)
    vocab, word2index, index2word = make_indices(vocab)

    print("========== Vectorizing data ==========")
    train_data = convert_to_vector_representation(train_data, word2index)
    valid_data = convert_to_vector_representation(valid_data, word2index)

    # Convert test data if provided
    if test_data:
        test_data = convert_to_vector_representation(test_data, word2index)

        # Check unique labels in the test data
        test_labels = [label for _, label in test_data]
        print("Unique labels in test data:", set(test_labels))

    model = FFNN(input_dim=len(vocab), h=args.hidden_dim)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    print("========== Training for {} epochs ==========".format(args.epochs))
    
    # Training loop with early stopping
    stopping_condition = False
    last_validation_accuracy = 0
    last_train_accuracy = 0

    for epoch in range(args.epochs):
        if stopping_condition:
            break

        model.train()
        optimizer.zero_grad()
        loss = None
        correct = 0
        total = 0
        start_time = time.time()
        print("Training started for epoch {}".format(epoch + 1))
        random.shuffle(train_data)  # Shuffle training data
        minibatch_size = 16
        N = len(train_data)
        
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_vector, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                predicted_vector = model(input_vector)
                predicted_label = torch.argmax(predicted_vector)
                correct += int(predicted_label == gold_label)
                total += 1
                example_loss = model.compute_Loss(predicted_vector.view(1, -1), torch.tensor([gold_label]))
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss
            loss = loss / minibatch_size
            loss.backward()
            optimizer.step()
        
        train_accuracy = correct / total
        print("Training completed for epoch {}".format(epoch + 1))
        print("Training accuracy for epoch {}: {}".format(epoch + 1, train_accuracy))
        print("Training time for this epoch: {}".format(time.time() - start_time))

        # Validation phase
        model.eval()
        correct = 0
        total = 0
        for input_vector, gold_label in valid_data:
            predicted_vector = model(input_vector)
            predicted_label = torch.argmax(predicted_vector)
            correct += int(predicted_label == gold_label)
        validation_accuracy = correct / len(valid_data)
        print("Validation accuracy for epoch {}: {}".format(epoch + 1, validation_accuracy))

        if validation_accuracy < last_validation_accuracy:
            stopping_condition = True
            print("Stopping early to prevent overfitting.")
            break
        else:
            last_validation_accuracy = validation_accuracy

    # Test phase with error logging
    if test_data:
        print("========== Testing on Test Data ==========")
        model.eval()
        correct = 0
        total = 0
        misclassified_examples = []

        for input_vector, gold_label in test_data:
            predicted_vector = model(input_vector)
            predicted_label = torch.argmax(predicted_vector)
            if predicted_label != gold_label:
                misclassified_examples.append({
                    "input_vector": input_vector,
                    "true_label": gold_label,
                    "predicted_label": predicted_label.item()
                })
            correct += int(predicted_label == gold_label)
        test_accuracy = (correct / len(test_data)) * 100
        print(f"Test Accuracy: {test_accuracy}%")
        print("Misclassified Examples:")
        for example in misclassified_examples[:5]:  # Show up to 5 misclassified examples
            print(f"True Label: {example['true_label']}, Predicted Label: {example['predicted_label']}")
