# Some part of the code was referenced from below.
# https://github.com/pytorch/examples/tree/master/word_language_model
import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils import clip_grad_norm
from data_utils import Dictionary, Corpus

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
embed_size = 128
hidden_size = 1024
num_layers = 1
num_epochs = 5
num_samples = 1000  # number of words to be sampled
batch_size = 20
seq_length = 30
learning_rate = 0.002

# Load "Penn Treebank" dataset
corpus = Corpus()
ids = corpus.get_data('data/train.txt', batch_size)
vocab_size = len(corpus.dictionary)
num_batches = ids.size(1)//seq_length

class RNNLM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(RNNLM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h):
        # Embed word ids to vectors
        # x.shape (before embedding): [batch_size, sequence_length]
        # x.shape (after embedding): [batch_size, sequence_length, embed_size
        x = self.embed(x)

        # forward propagate LSTM
        out, (h,c) = self.lstm(x,h)

        # reshape output to (batch_size*sequence_length, hidden_size)
        out = out.reshape(out.size(0)*out.size(1), out.size(2))

        # decode hidden states of all time steps
        out = self.linear(out)
        return out, (h, c)

model = RNNLM(vocab_size, embed_size, hidden_size, num_layers).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# truncated back propagation
def detach(states):
    # Every variable has a .creator attribute that is an entry point to a
    # graph, that encodes the operation history. This allows autograd to
    # replay it and differentiate each op. So each hidden state will have
    # a reference to some graph node that has created it. In here we are
    # doing back propagation through time, so you never want to backprop
    # to it after you finish the sequence. To get rid of the reference,
    # you have to take out the tensor containing the hidden state h.data
    # and wrap it in a fresh Variable, that has no history (is a graph leaf).
    # This allows the previous graph to go out of scope and free up the memory
    # for next iteration.
    return [state.detach() for state in states]
    # var.detach(): Returns a new Tensor, detached from the current graph.

# training
for epoch in range(num_epochs):
    # initialise hidden and cell states
    states = (torch.zeros(num_layers, batch_size, hidden_size).to(device),
              torch.zeros(num_layers, batch_size, hidden_size).to(device))
    for i in range(0, ids.size(1)-seq_length, seq_length):
        # get minibatch inputs and targets
        inputs = ids[:, i:i+seq_length].to(device)
        targets = ids[:, (i+1):(i+1)+seq_length].to(device)

        # forward pass
        states = detach(states)
        outputs,states = model(inputs, states)
        loss = criterion(outputs, targets.reshape(-1))
        # shape:
        #       outputs: length x n_classes (vocab_size)
        #       targets: length
        # think about it as we are trying to find the probability of each
        # word in this sentence being any word in the whole vocabulary

        # backward and optimize
        model.zero_grad()
        loss.backward()
        clip_grad_norm(model.parameters(), 0.5)
        # This function ‘clips’ the norm of the gradients by scaling the
        # gradients down by the same amount in order to reduce the norm
        # to an acceptable level. In practice this places a limit on the
        # size of the parameter updates. The hope is that this will ensure
        # that your model gets reasonably sized gradients and that the
        # corresponding updates will allow the model to learn.
        optimizer.step()

        step = (i+1) // seq_length
        if step % 100 == 0:
            print('Epoch [{}/{}], Step[{}/{}], Loss: {:.4f}, Perplexity: {:5.2f}'
                  .format(epoch + 1, num_epochs, step, num_batches, loss.item(), np.exp(loss.item())))

with torch.no_grad():
    with open('data/sample.txt', 'w') as f:
        state = (torch.zeros(num_layers, 1, hidden_size).to(device),
                 torch.zeros(num_layers, 1, hidden_size).to(device))

        # randomly select a word id
        prob = torch.ones(vocab_size) # size: [1, vocab_size]
        input = torch.multinomial(prob, num_samples=1).unsqueeze(1).to(device)
        # multinomial: Returns a tensor where each row contains num_samples
        #              indices sampled from the multinomial probability
        #              distribution located in the corresponding row of
        #              tensor input.
        # unsqueeze: add dimention at specified dimension

        for i in range(num_samples):
            output, state = model(input, state)

            # sample a word id based on output
            prob = output.exp()
            word_id = torch.multinomial(prob, num_samples=1).item()

            # fill input with sampled word id for next time step
            input.fill_(word_id)
            print("input shape in %d iter:" % (i))
            print(input.shape)

            # File write
            word = corpus.dictionary.idx2word[word_id]
            word = '\n' if word == '<eos>' else word + ' '
            f.write(word)

            if (i + 1) % 100 == 0:
                print('Sampled [{}/{}] words and save to {}'.format(i + 1, num_samples, 'sample.txt'))

        # Save the model checkpoints
        torch.save(model.state_dict(), 'model/7NLP_rnn_yunjey.ckpt')







































