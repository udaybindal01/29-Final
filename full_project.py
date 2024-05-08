import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import spacy
from transformers import BertTokenizer
from transformers import XLMRobertaModel
import csv
import re
from math import log
from tqdm import tqdm
# filename = 'HinGE.pkl'

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
vocab = tokenizer.get_vocab()
# print(list(vocab.keys())[list(vocab.values()).index(1020)])

# with open(filename, 'rb') as f:
#   df = pickle.load(f)
hin = []
with open("hin_data.csv", mode='r') as file:
    reader = csv.reader(file)
    c = 1
    for row in reader:
        # print(row)
        # exit()
        if c:
            c = 0
            continue
        hin.append(row[1])

# print(hin[0])
# print(hin[1])
# print(len(hin))
# exit()

# print(hin[1])
# print(df)
def load_data(f):
    train = []
    with open(f, mode='r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader, None)
        for row in csv_reader:
            for j in range(3):
                if row[j][-1] == '\n':
                    row[j] = row[j][:-1]  
            train.append(row)
    return train
train = load_data('hin_data.csv')
for i in range(len(train)):
    train[i][0] = re.sub(r"[.,/?;':~`\"!@#$%&*()_=+{}\[\]\-“”<>,!‘’]","", train[i][0])

# print(len(train))
# exit()

ne_recognizer = spacy.load('en_core_web_sm')
eng_bpe_embed = torch.load('eng_bpe_embed.pt')
hin_bpe_embed = torch.load('hin_bpe_embed.pt')
# print(type(hin_bpe_embed))
# print(hin_bpe_embed(torch.tensor(100)))
# exit(0)
# Assuming BPE tokenizer and BPE vocab are loaded from somewhere like HuggingFace transformers


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
attention_hidden=12
linear1_len=16
decoder_hidden=32
batch_size=1

class Attention(torch.nn.Module):
    def __init__(self,input_len,hidden_dim, batch_size):
        super().__init__()
        self.hidden_dim=hidden_dim
        self.input_len=input_len
        self.batch_size = batch_size
        self.feed_forward = torch.nn.Sequential(
            torch.nn.Linear(self.input_len, self.hidden_dim,device=device),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, 1, device=device)
        )
    
    def forward(self, hidden_state, input_entries):
        hidden_state = torch.permute(hidden_state,(1,0))
        input_entries=torch.permute(input_entries,(1,0))
        new_input=torch.permute(torch.cat((hidden_state,input_entries)),(1,0))
        outalpha=self.feed_forward(new_input)
        outalpha = torch.permute(outalpha,(1,0))[0]
        return outalpha

class LSTM_Attention(torch.nn.Module):
    def __init__(self,input_size, output_size, hidden_dim, batch_size, embed_len, vocabulary):
        super(LSTM_Attention,self).__init__()
        self.hidden_dim=hidden_dim
        self.vocabulary=vocabulary
        self.embed_len=embed_len   #embed_len = size of embeddings recieved from encoder step
        self.input_len=input_size  #input_size = size of word embeddings
        self.batch_size=batch_size
        self.output_size=output_size
        self.lstm=torch.nn.LSTMCell(input_size,hidden_dim,device=device)
        self.fc = torch.nn.Linear(hidden_dim, output_size, device=device)
        self.attention = Attention(hidden_dim + embed_len, attention_hidden, batch_size)
        self.wh = torch.nn.Parameter(torch.rand((self.embed_len),device=device))
        self.ws = torch.nn.Parameter(torch.rand((self.hidden_dim),device=device))
        self.wx = torch.nn.Parameter(torch.rand((self.input_len),device=device))
        
        self.linear_prob_gen = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim + embed_len, linear1_len, device=device),
            torch.nn.Linear(linear1_len, output_size, device=device),
            torch.nn.Softmax(dim=1)
        )
        
    def forward(self, x, true_outputs=None, true_word=None):
        x = torch.permute(x,(1,0,2))
        h = self.init_hidden(self.batch_size)
        c = self.init_hidden(self.batch_size)
        outputs=[]
        
        if type(true_outputs[0])==list:
            print(true_outputs)
            print("huh")
            exit()
        true_outputs = torch.stack(true_outputs)
        # total_loss=torch.tensor(0,device=device, dtype=float, requires_grad=True)
        losses=torch.zeros(batch_size,device=device, dtype=float, requires_grad=False)
        if true_outputs!=None:
            transposed_outputs=torch.permute(true_outputs,(1,0,2))
        if true_outputs == None:
            wordcount=len(x)
        else:
            wordcount=len(transposed_outputs)
            
        for time_step in range(wordcount):
            e_i=[]
            for word_index in x:
                e_ij = self.attention(h, word_index)
                e_i.append(e_ij)
            e_i=torch.permute(torch.stack(e_i),(1,0))
            denominator=torch.sum(e_i,(1))
            alpha_i=torch.zeros((self.batch_size,len(e_i[0])))
            for row in range(self.batch_size):
                alpha_i[row]=e_i[row]/denominator[row]
                
            c_i=torch.zeros((self.batch_size,self.embed_len),device=device)
            x = torch.permute(x,(1,0,2))               #Realigning batches
            for sentence in range(self.batch_size):    #Calculating context vector for each element in the batch
                for word in range(len(x[sentence])):
                    c_i[sentence]+=alpha_i[sentence][word]*x[sentence][word]
            x = torch.permute(x,(1,0,2))
            
            # If true_outputs = None then we are on test step and inputs are previous output. Else take the given values
            if true_outputs!=None:
                lstm_in = transposed_outputs[time_step]
                
            if time_step==0:
                lstm_in = torch.rand((self.batch_size,self.input_len),device=device)  # Replace with embedding for start token OR add start token to the start of the output
            (h,c) = self.lstm(lstm_in, (h,c))
            
            u_t = torch.rand((self.batch_size,self.input_len),device=device)            # replace with embedding of output found using h
            pvocab=self.linear_prob_gen(torch.cat((h,c_i),1))
            val=torch.matmul(c_i,self.wh) + torch.matmul(h,self.ws) + torch.matmul(u_t,self.wx)
            pgen = torch.sigmoid(val)
            
            # pvocab = torch.permute(pvocab,(1,0))
            # for word in range(self.output_size):
            #     pvocab[word]=torch.dot(pvocab[word],pgen)
            # implement probability of copying
            # print(true_outputs.shape)
            # print(pvocab.shape)
            transposed_vocab=torch.permute(pvocab,(1,0))
            if true_outputs!=None:
                for sentence in range(self.batch_size):
                    myword=true_word[sentence][time_step]
                    if myword < 0 or myword >= len(transposed_vocab):
                        # print("Invalid index:", myword)
                        continue
                    if transposed_vocab[myword] <= 0:
                        # print("Invalid probability:", transposed_vocab[myword])
                        continue
                    losses[sentence]-= log(transposed_vocab[myword])
            
            out = torch.argmax(pvocab,dim=0)
            outputs.append(out)
            if true_outputs == None:
                lstm_in = [torch.rand(self.input_len,device=device) for i in range(self.batch_size)]    # replace with embedding for the element
                lstm_in = torch.stack(lstm_in)
        total_loss=sum(losses)/len(losses)
        total_loss = torch.tensor(total_loss, requires_grad=True)
        return(torch.permute(torch.stack(outputs),(1,0)), total_loss)
            
    def init_hidden(self, batch_size):
        hidden = torch.zeros(batch_size, self.hidden_dim,device=device)
        return hidden

    
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, pos_tags, ne_recognizer, hin):
        self.texts = texts
        self.tokenizer = tokenizer
        self.pos_tags = pos_tags
        self.ne_recognizer = ne_recognizer
        self.hin = hin

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        # print(text)
        # print(len(text))
        # print(len(hin))
        # print(hin)
        # print(idx)
        curr_hin = self.hin[idx]
        # print(hin[3])
        doc = self.ne_recognizer(text)
        # print(len(text))
        # print("type:" ,type(doc))
        # print("LENdoc:", len(doc))
        tokens = [token.text.lower() for token in doc]
        # print("text:",text)
        # print(len(tokens))
        pos = [token.pos_ for token in doc]
        entities = list(doc.ents)
       
        entities = [str(entity).lower() for entity in entities]
        # print(len(entities))

        entity_status = {token: 0 for token in tokens}
        # print(len(entity_status))
        for entity in entities:
            entity_status[entity] = 1
        # print(entity_status)
        
        ent_stat=[]
        for token in tokens:
            ent_stat.append(entity_status[token])
        
        # ent_stat = list(entity_status.values())
        # print(ent_stat)
        # exit(0)

        token_ids =  self.tokenizer.encode(text)
        # print(token_ids)
        # print(curr_hin)
        hin_ids = self.tokenizer.encode(curr_hin)
        hin_ids = hin_ids[1:-1]
        # print(hin_ids)
        # exit()
        Hin_ids = []
        for i in range (len(hin_ids)):
            Hin_ids.append(hin_bpe_embed(torch.tensor(hin_ids[i])))
        # print(Hin_ids)
        Hin_ids = torch.stack(Hin_ids)
        # exit()
        bpe_tokens = tokenizer.convert_ids_to_tokens(token_ids)
        bpe_tokens = bpe_tokens[1:-1]
        token_ids = token_ids[1:-1]
        # print("BPE: ",len(bpe_tokens))
        # print("TOKEN", len(token_ids))
        # print(bpe_tokens)

        bpe_pos = []
        bpe_ne = []
        curr_pos = ""
        curr_ne = 0
        curr_ind = 0
        i = 0
        # print("POS:",len(pos))
        # print("ENTSTAT:",len(ent_stat))
        while curr_ind < len(bpe_tokens) :
            if bpe_tokens[curr_ind].startswith("##"):
                while curr_ind < len(bpe_tokens) and bpe_tokens[curr_ind].startswith("##"):
                    bpe_pos.append(curr_pos)
                    bpe_ne.append(curr_ne)
                    curr_ind += 1
            else:
                if i<len(pos):
                    bpe_pos.append(pos[i])
                    bpe_ne.append(ent_stat[i])
                    curr_pos = pos[i]
                    curr_ne = ent_stat[i]
                else:
                    bpe_pos.append(pos[len(pos)-1])
                    bpe_ne.append(ent_stat[len(pos)-1])
                    # curr_pos = pos[i]
                    # curr_ne = ent_stat[i]
                curr_ind += 1
                
                i += 1
        # print(len(bpe_pos),len(bpe_ne))
        # print(bpe_pos)
        bpe_pos = [self.pos_tags.get(tag, self.pos_tags['VERB']) for tag in bpe_pos]
        # print(bpe_pos)
        
        return torch.tensor(token_ids,device=device), torch.tensor(bpe_pos,device=device), torch.tensor(bpe_ne,device=device), Hin_ids, hin_ids
    
def collate_fn(batch):
    if batch != None:
        token_ids, pos_tags, ne_tags , curr_hin ,hin= zip(*batch)
        token_ids = pad_sequence(token_ids, batch_first=True, padding_value=0)
        pos_tags = pad_sequence(pos_tags, batch_first=True, padding_value=POS_tags['<PAD>'])
        # print("ne tags:",ne_tags)
        ne_tags = pad_sequence(ne_tags, batch_first=True, padding_value=2)
        
        # print("curr_hin: ",curr_hin)
        # exit(0)
        # curr_hin = pad_sequence(curr_hin,batch_first=True, padding_value=0)

    
    return token_ids, pos_tags, ne_tags, curr_hin,hin
# Example use of the dataset

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, pos_size, ne_size, embed_dim, hidden_dim):
        super(TextEncoder, self).__init__()
        self.word_embed = nn.Embedding(vocab_size, embed_dim, device=device)
        self.pos_embed = nn.Embedding(pos_size, embed_dim, device=device)
        self.ne_embed = nn.Embedding(ne_size, embed_dim,device=device)
        self.lstm = nn.LSTM(embed_dim * 3, hidden_dim, bidirectional=True, batch_first=True,device=device)
        # self.linear = nn.Linear(hidden_dim * 2, 768)  #added this

    def forward(self, tokens, pos_tags, ne_tags):
        # Embedding lookups
        word_embeddings = self.word_embed(tokens)
        # print(tokens)
        # print(pos_tags)
        # print(ne_tags)
        pos_embeddings = self.pos_embed(pos_tags)
        ne_embeddings = self.ne_embed(ne_tags)

        # Concatenate embeddings
        # print(word_embeddings.shape)
        # print(pos_embeddings.shape)
        # print(ne_embeddings.shape)
        embeddings = torch.cat([word_embeddings, pos_embeddings, ne_embeddings], dim=-1)

        # Pass through LSTM
        output, _ = self.lstm(embeddings)
        # output = self.linear(output)
        return output, word_embeddings

# Example usage

POS_tags = {
    'NOUN': 2,  # Noun
    'VERB': 3,  # Verb
    'ADJ': 4,    # Adjective
    'ADV': 5,  # Adverb
    'PRON': 6,   # Pronoun
    'DET': 7,   # Determiner
    'ADP': 8,   # Adposition (prepositions and postpositions)
    'PROPN': 9, # Proper noun
    'PART': 10,  # Particle
    'INTJ': 11,  # Interjection
    'SYM': 12,   # Symbol
    'NUM': 13,   # Numeral
    'CONJ': 14,  # Conjunction
    'SCONJ': 15, # Subordinating conjunction
    'PUNCT': 16, # Punctuation
    'SPACE': 17, # Space (whitespace, tabs, newline, etc.)
    'AUX': 18,   # Auxiliary verb
    'CCONJ': 19, # Coordinating conjunction
    'POS': 20,   # Possessive ending
    '<PAD>': 22
}

# for i in range(len(train)):
#     print(train[i][0])
# print(len(hin))
# exit()
dataset = TextDataset([train[i][0] for i in range(len(train))], tokenizer, POS_tags, ne_recognizer, hin)
# for i in dataset:
#     print(i)
#     exit(0)
loader = DataLoader(dataset, batch_size, shuffle=False, collate_fn=collate_fn)

vocab_size = tokenizer.vocab_size  # Vocabulary size from tokenizer
embed_dim = 32  # Size of each embedding vector
hidden_dim = 32  # Size of LSTM hidden states
pos_size = len(POS_tags)
# print("pos_len", pos_size)

ne_size = 2

model = TextEncoder(vocab_size, pos_size, ne_size, embed_dim, hidden_dim)

class XLMEncoder(nn.Module):
    def __init__(self):
        super(XLMEncoder, self).__init__()
        self.xlm_model = XLMRobertaModel.from_pretrained('xlm-roberta-base')
        self.xlm_model.to(device)
        self.linear = nn.Linear(768, 64,device=device)

    def forward(self, tokens):
        outputs = self.xlm_model(tokens)
        last_hidden_state = outputs.last_hidden_state
        reshaped_output = self.linear(last_hidden_state)
        return reshaped_output

class GatedFeatureFusion(nn.Module):
    def __init__(self, text_encoder, xlm_encoder, hidden_dim):
        super(GatedFeatureFusion, self).__init__()
        self.text_encoder = text_encoder
        self.xlm_encoder = xlm_encoder
        self.gate = nn.Linear(hidden_dim * 2, 1, device=device)
        self.dropout = nn.Dropout(0.1)
        self.hidden_dim = hidden_dim

    def forward(self, tokens, pos_tags, ne_tags):
        text_output, word_embeddings = self.text_encoder(tokens, pos_tags, ne_tags)
        # print(text_output.shape)
        tokens.to(device)
        xlm_output = self.xlm_encoder(tokens)
        # print(xlm_output.shape)
        # Concatenate the outputs
        combined_output = torch.cat((text_output, xlm_output), dim=-1)
        # print(combined_output.shape)
        # Apply gating mechanism
        gate_values = torch.sigmoid(self.gate(combined_output))

        # Apply gating to the text and XLM outputs
        text_output_gated = text_output * gate_values
        xlm_output_gated = xlm_output * (1 - gate_values)

        # Concatenate gated outputs
        fused_output = torch.cat((text_output_gated, xlm_output_gated), dim=-1)


        fused_output = self.dropout(fused_output)

        return fused_output

# Initialize XLM model and Gated Feature Fusion model
xlm_model = XLMEncoder()
gated_model = GatedFeatureFusion(model, xlm_model, hidden_dim * 2)  # hidden_dim * 2 because of bidirectional LSTM
decoder_model = LSTM_Attention(embed_dim, vocab_size, decoder_hidden, batch_size, embed_len=4*hidden_dim, vocabulary=eng_bpe_embed)

NUM_EPOCHS = 5

# Expected shape: (batch_size, seq_length, 2 * hidden_dim)

# Define optimizer and loss function
optimizer = torch.optim.Adam(list(gated_model.parameters()) + list(decoder_model.parameters()), lr=0.001)
# criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in tqdm(range(NUM_EPOCHS)):
    total_loss = 0
    for data in loader:
        token_ids, pos_tags, ne_tags, Hindi_ids, hin = data
        # print(Hindi_ids)
        # exit()
        # Hindi_ids = Hindi_ids[0]
       
        # print(Hindi_ids)
        # exit()

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass through the model
        # print(hin_ids)
        # print()
        # print(token_ids)
        # exit()
        fused_output = gated_model(token_ids, pos_tags, ne_tags)
        decoder_output, loss = decoder_model(fused_output, true_outputs=Hindi_ids, true_word = hin)
        
        # print(loss)
        # print(decoder_output.shape)
        # print(token_ids.shape)
        # print(token_ids.view(-1).shape)
        

        # Calculate the loss
        # loss = criterion(decoder_output.view(-1, vocab_size), token_ids.view(-1))

        # Backpropagation
        loss.backward()
 
        # Update model parameters
        optimizer.step()

        total_loss += loss.item()
        # print(total_loss)
    # Print average loss for the epoch
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(loader)}")

# torch.save({
#     'gated_model_state_dict': gated_model.state_dict(),
#     'decoder_model_state_dict': decoder_model.state_dict(),
# }, 'model.pt')

checkpoint = torch.load('model.pt')
gated_model.load_state_dict(checkpoint['gated_model_state_dict'])
decoder_model.load_state_dict(checkpoint['decoder_model_state_dict'])


import nltk
# nltk.download('all')
from nltk.translate.bleu_score import corpus_bleu
# from nltk_translate import Translator

test_data = load_data('test.csv')

def generate_translation(model, input_text, max_length=100):

    # token_ids = tokenizer.encode(input_text, return_tensors='pt')
    translation = ""

    model.eval()
    
    with torch.no_grad():

        fused_output = gated_model(token_ids, pos_tags, ne_tags)
        print(fused_output)
        translation_output, _ = model(fused_output,true_outputs=Hindi_ids, true_word = hin)
        print(translation_output)

    for token_id in translation_output[0]:
        token = tokenizer.decode(token_id.item())
        if token == '[PAD]':  # Stop decoding at special tokens
            break
        translation += token + " "
        # Stop decoding if the maximum length is reached
        if len(translation.split()) >= max_length:
            break
    
    return translation.strip()

translations = []
reference_texts = []

for data in loader:
    token_ids, pos_tags, ne_tags, Hindi_ids, hin = data
    fused_output = gated_model(token_ids, pos_tags, ne_tags)
    # Generate translations
    for i in range(len(token_ids)):
        input_text = " ".join([tokenizer.decode(token_ids[i].tolist())])
        reference_text = hin[i]
        translation = generate_translation(decoder_model, fused_output[i])
        print(translation)
        exit()
        translations.append(translation)
        reference_texts.append(reference_text)

# Compute BLEU score
bleu_score = corpus_bleu(reference_texts, translations)
print("BLEU Score:", bleu_score)

# Compute METEOR score
# translator = Translator()
# meteor_score = translator.corpus_meteor(reference_texts, translations)
# print("METEOR Score:", meteor_score)

