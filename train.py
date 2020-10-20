import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import IWSLT
from torchtext.data import Field, BucketIterator
import spacy
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from model import Transformer

#execute on gpu if available, else on cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#load spacy language modules to build vocabularies
spacy_fr = spacy.load("fr")
spacy_en = spacy.load("en")

#use spacy tokenization functions
def tokenize_fr(text):
    return [tok.text for tok in spacy_fr.tokenizer(text)]

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

french = Field(tokenize=tokenize_fr, lower=True, init_token="<sos>", eos_token="<eos>")
english = Field(tokenize=tokenize_en, lower=True, init_token="<sos>", eos_token="<eos>")

#use the ISWLT dataset of TED talks
train_data, valid_data, test_data = IWSLT.splits(exts=(".fr", ".en"), fields=(french, english))

#build vocabularies
french.build_vocab(train_data, max_size=10000, min_freq=2)
english.build_vocab(train_data, max_size=10000, min_freq=2)

#model hyperparameters
embed_dim = 512
num_heads = 1
expansion_size = 2048
num_layers = 3
src_dict_size = len(french.vocab)
trg_dict_size = len(english.vocab)
max_len = 100
dropout = 0.1

#training hyperparameters
batch_size = 32
num_epochs = 10
learn_rate = 0.0003

load_model = False
save_model = True

#use Tensorboard summary writer to plot loss and accuracy during training
step = 0
writer = SummaryWriter("runs/loss-plot")

#use the BucketIterator module to split data into batches sorted by sentence length
train_it, valid_it, test_it = BucketIterator.splits((train_data, valid_data, test_data), batch_size=batch_size, sort_within_batch=True, sort_key= lambda x: len(x.src), device=device)

#make instance of tranformer model
model = Transformer(embed_dim, num_heads, expansion_size, num_layers, src_dict_size, trg_dict_size, max_len=max_len, drop=dropout).to(device)

#make optimizer
optimizer = optim.Adam(model.parameters(), lr=learn_rate)

#get padding index and ignore it when calculating loss
pad_idx = english.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

#can load saved model and optimizer
if load_model:

    print("=> loading checkpoint")
    checkpoint = torch.load("my_checkpoint.pth.tar")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

#loop over epochs
for epoch in range(num_epochs):

    losses = []
    print(f"epoch {epoch + 1}/{num_epochs}")

    #can save model and optimizer in ".tar" file
    if save_model:

        print("=> saving checkpoint")
        checkpoint = {"state_dict":model.state_dict(), "optimizer":optimizer.state_dict()}
        torch.save(checkpoint, "my_checkpoint.pth.tar")

    #loop over batches (use tqdm to make a progress bar)
    for batch in tqdm(train_it):

        #ignore batches with sentence length over 100
        if (batch.src.size(0) > 100) or (batch.trg.size(0) > 100):

          continue
        
        #convert the input and the target into the adquate shape
        inp = batch.src.transpose(0, 1).to(device)
        trg = batch.trg.transpose(0, 1).to(device)

        #pass them through the model
        out = model(inp, trg[:, :-1])
        out = out.reshape(-1, out.shape[2])
        trg = trg[:, 1:].reshape(-1)

        #zero the gradients
        optimizer.zero_grad()

        #calculate the loss backward it
        loss = criterion(out, trg)
        losses.append(loss.item())
        loss.backward()

        #clip the gradients
        torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=1)

        optimizer.step()
        
        #add summary writer step
        writer.add_scalar("Training Loss", loss, global_step=step)
        step += 1
    
    #calculate the average loss over one epoch
    mean_loss = sum(losses)/len(losses)
    print("loss :",mean_loss)