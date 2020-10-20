import torch
import spacy
from model import Transformer
from torchtext.data import Field
from torchtext.datasets import IWSLT

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
num_heads = 8
expansion_size = 2048
num_layers = 3
src_dict_size = len(french.vocab)
trg_dict_size = len(english.vocab)
max_len = 100
dropout = 0.1

#make instance of transformer model
model = Transformer(embed_dim, num_heads, expansion_size, num_layers, src_dict_size, trg_dict_size, max_len=max_len, drop=dropout).to(device)

#load model
print("LOADING MODEL...")
checkpoint = torch.load("my_checkpoint.pth.tar")
model.load_state_dict(checkpoint["state_dict"])

#test sentence
sentence = "un chat noir et blanc est mignon."

#tokenize sentence and convert it to indices
tokens = [tok.text for tok in spacy_fr(sentence)]
tokens.insert(0, french.init_token)
tokens.append(french.eos_token)
indices = [french.vocab.stoi[tok] for tok in tokens]
inp = torch.LongTensor(indices).unsqueeze(0).to(device)

#second param will be start of sentence index
outs = [english.vocab.stoi["<sos>"]]

#for each loop the model will predict the next word in the translated sentence
for i in range(max_len):

    #convert target to tensor
    trg = torch.LongTensor(outs).unsqueeze(0).to(device)

    #predict next word
    with torch.no_grad():
        out = model(inp, trg)
        best_guess = out.argmax(2)[:, -1].item()
        outs.append(best_guess)

        if best_guess == english.vocab.stoi["<eos>"]:
            break

#convert indices into words
translated_sentence = [english.vocab.itos[idx] for idx in outs]

#print sentence without "start" and "end" tokens
print(translated_sentence[1:-1])