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
import matplotlib.pyplot as plt

unk = '<UNK>'
# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html

class FFNN(nn.Module):
    def __init__(self, input_dim, h):
        super(FFNN, self).__init__()
        self.h = h
        self.W1 = nn.Linear(input_dim, h)
        self.activation = nn.ReLU()
        # self.W_hidden1 = nn.Linear(h, h)  # First additional hidden layer
        # self.W_hidden2 = nn.Linear(h, h)  # Second additional hidden layer
        self.W2 = nn.Linear(h, 5)
        self.softmax = nn.LogSoftmax(dim=0)
        self.loss = nn.NLLLoss()

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, input_vector):
        # Forward pass through the first layer
        hidden_layer_output = self.activation(self.W1(input_vector))
        # Forward pass through the first additional hidden layer (commented out)
        # hidden_layer_output = self.activation(self.W_hidden1(hidden_layer_output))
        # Forward pass through the second additional hidden layer (commented out)
        # hidden_layer_output = self.activation(self.W_hidden2(hidden_layer_output))
        # Final output layer
        output_layer_output = self.W2(hidden_layer_output)
        # Obtain probability distribution
        predicted_vector = self.softmax(output_layer_output)
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

def load_data(train_data, val_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)

    tra = []
    val = []
    for elt in training:
        tra.append((elt["text"].split(), int(elt["stars"]-1)))
    for elt in validation:
        val.append((elt["text"].split(), int(elt["stars"]-1)))
    return tra, val

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required=True, help="hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required=True, help="num of epochs to train")
    parser.add_argument("--train_data", required=True, help="path to training data")
    parser.add_argument("--val_data", required=True, help="path to validation data")
    parser.add_argument("--test_data", default=None, help="path to test data (optional)")
    parser.add_argument('--do_train', action='store_true')
    args = parser.parse_args()

    random.seed(42)
    torch.manual_seed(42)

    print("========== Loading data ==========")
    train_data, valid_data = load_data(args.train_data, args.val_data) # X_data is a list of pairs (document, y); y in {0,1,2,3,4}

    # Think about the type of function that an RNN describes. To apply it, you will need to convert the text data into vector representations.
    # Further, think about where the vectors will come from. There are 3 reasonable choices:
    # 1) Randomly assign the input to vectors and learn better embeddings during training; see the PyTorch documentation for guidance
    # 2) Assign the input to vectors using pretrained word embeddings. We recommend any of {Word2Vec, GloVe, FastText}. Then, you do not train/update these embeddings.
    # 3) You do the same as 2) but you train (this is called fine-tuning) the pretrained embeddings further.
    # Option 3 will be the most time consuming, so we do not recommend starting with this

    print("========== Vectorizing data ==========")
    vocab = make_vocab(train_data)
    vocab, word2index, index2word = make_indices(vocab)
    train_data = convert_to_vector_representation(train_data, word2index)
    valid_data = convert_to_vector_representation(valid_data, word2index)

    # First test with a new hidden dimension size (e.g., 64)
    model = FFNN(input_dim=len(vocab), h=64)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    print("========== Training for {} epochs ==========".format(args.epochs))

    train_losses = []
    val_accuracies = []

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        loss = None
        correct = 0
        total = 0
        start_time = time.time()
        print("Training started for epoch {}".format(epoch + 1))
        random.shuffle(train_data)
        minibatch_size = 32
        N = len(train_data)
        loss_total = 0
        loss_count = 0
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
            loss_total += loss.item()
            loss_count += 1
        avg_loss = loss_total / loss_count
        train_losses.append(avg_loss)
        print("Training accuracy for epoch {}: {}".format(epoch + 1, correct / total))

        # Validation phase
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for input_vector, gold_label in valid_data:
                predicted_vector = model(input_vector)
                predicted_label = torch.argmax(predicted_vector)
                correct += int(predicted_label == gold_label)
                total += 1
        validation_accuracy = correct / total
        val_accuracies.append(validation_accuracy)
        print("Validation accuracy for epoch {}: {}".format(epoch + 1, validation_accuracy))

    # Plotting the learning curve
    epochs = list(range(1, args.epochs + 1))
    plt.figure(figsize=(10, 5))

    # Plotting training loss
    plt.plot(epochs, train_losses, label='Training Loss', color='b')

    # Plotting validation accuracy
    plt.plot(epochs, val_accuracies, label='Validation Accuracy', color='g')

    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.title('Learning Curve')
    plt.legend()
    plt.show()
