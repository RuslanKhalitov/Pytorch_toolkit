import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import numpy as np
import spacy
import random
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard
from attention_transformer.utils import translate_sentence, bleu, save_checkpoint, load_checkpoint
import warnings

warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

spacy_eng = spacy.load('en')
spacy_ger = spacy.load('de')


# Define tokenizers
def tokenize_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]


def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]


german = Field(tokenize=tokenize_ger, lower=True, init_token="<sos>", eos_token="<eos>")

english = Field(
    tokenize=tokenize_eng, lower=True, init_token="<sos>", eos_token="<eos>"
)

train_data, valid_data, test_data = Multi30k.splits(
    exts=(".de", ".en"), fields=(german, english)
)

german.build_vocab(train_data, max_size=15000, min_freq=2)
english.build_vocab(train_data, max_size=15000, min_freq=2)


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p_drop):
        super(Encoder, self).__init__()
        self.dropout = nn.Dropout(p=p_drop)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p_drop)

    def forward(self, x):
        # x[sequence_length, N], N-batch_size

        embedding = self.embedding(x)
        embedding = self.dropout(embedding)
        # embedding[sequence_length, N, embedding_size]

        outputs, (hidden, cell) = self.rnn(embedding)
        # outputs[sequence_length, N, hidden_size]

        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size,
                 output_size, num_layers, p_drop):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(p=p_drop)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p_drop)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        # x[N], N-batch_size, since we send in here only one word

        x = x.unsqueeze(0)
        # x[1, N]

        embedding = self.embedding(x)
        embedding = self.dropout(embedding)
        # embedding[1, N, embedding_size]

        outputs, (hidden, cell) = self.rnn(embedding)
        # outputs[1, N, hidden_size]

        predictions = self.fc(outputs)
        # predictions[1, N, length_target_vocabulary]
        # Because we output the probabilities of each word in the vacabulary
        # We want to send in loss function shape = [N, length_target_vocabulary]
        predictions = predictions.squeeze(0)

        return predictions, hidden, cell
        # So every step we reuse it


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0.5):
        # source[sequence_length, N]
        batch_size = source.shape[1]
        target_length = target.shape[0]
        target_vocab_size = len(english.vocab)
        # ger -> eng

        outputs = torch.zeros(target_length, batch_size, target_vocab_size).to(device)

        hidden, cell = self.encoder(source)

        x = target[0]
        for index in range(1, target_length):
            output, hidden, cell = self.decoder(x, hidden, cell)
            outputs[index] = output
            best = output.argmax(1)
            x = target[index] if random.random() < teacher_force_ratio else best

        return outputs


# Training
# Hyperparameters

num_epochs = 20
save_freq = 2
lr = 0.001
batch_size = 64
load_model = False
input_size_encoder = len(german.vocab)
input_size_decoder = len(english.vocab)
output_size = len(english.vocab)

encoder_embedding_size = 300
decoder_embedding_size = 300
hidden_size = 1024
num_layers = 2
p_drop_encoder = 0.5
p_drop_decoder = 0.5

# Tensorboard
writer = SummaryWriter(f"runs/loss_seq2seq")
step = 0

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=batch_size,
    sort_within_batch=True,
    sort_key=lambda x: len(x.src),
    device=device
)

encoder_ = Encoder(
    input_size_encoder,
    encoder_embedding_size,
    hidden_size,
    num_layers,
    p_drop_encoder
).to(device)

decoder_ = Decoder(
    input_size_decoder,
    decoder_embedding_size,
    hidden_size,
    output_size,
    num_layers,
    p_drop_decoder
).to(device)

model = Seq2Seq(encoder_, decoder_).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
pad_idx = english.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

# Single sentence to test the performance of the trained model
test_sentence = "Liebe ist Glück, aber auch Leid."
test_gt = "Love is happiness, but also suffering."
example = True

# Main training loop
for epoch in range(num_epochs):
    print(f"[Epoch {epoch} of {num_epochs}]")
    if epoch % save_freq == 1:
        checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        save_checkpoint(checkpoint)

    if example:
        model.eval()
        translation = translate_sentence(
            model, test_sentence, german, english, device, max_length=50
        )
        print(f"Translation: \n {translation} GT: \n {test_gt}")
        model.train()

    for batch_idx, batch in enumerate(train_iterator):
        input_ = batch.src.to(device)
        target = batch.trg.to(device)

        # Forward
        output = model(input_, target)
        # (target_length, batch_size, output_dim)
        # Fit the input format of the criterion and remove the starting token
        output = output[1:].reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)


        optimizer.zero_grad()
        loss = criterion(output, target)

        # Backward
        loss.backward()

        # Gradient Clipping, since we use LSTM
        torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=1)

        # Grad descent step
        optimizer.step()

        #TB
        writer.add_scalar("Training loss", loss, global_step=step)
        step += 1


score = bleu(test_data[1:100], model, german, english, device)
print(f"BLEU Score: {score*100:.2f}")


