import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size, dropout=0.5):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet101(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)

        # Remove final linear layer because we are not doing classification
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)

        # Create embedding from a fully connected layer
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

        # initilize weights
        torch.nn.init.xavier_normal_(self.embed.weight)
        self.embed.bias.data.fill_(0.0)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)

        return features

    
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=2, dropout=0.25):
        super(DecoderRNN, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        # image captioning tends to overfit badly so dropout is here to help
        self.dropout = dropout

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

        # initialize weights
        torch.nn.init.xavier_normal_(self.fc.weight)
        self.fc.bias.data.fill_(0.0)

    def forward(self, features, captions):
        assert captions.shape[0] == features.shape[0], "Features and captions should have the same batch size."

        # clean hidden and cell from previous batch
        self.hidden_cell = (torch.zeros(self.num_layers, 1, self.hidden_size).to(self.device),
                            torch.zeros(self.num_layers, 1, self.hidden_size).to(self.device))

        # remove the 'END' token to satisfy output shape without information loss 
        embeds = self.embedding(captions[:,:-1])

        # stack features and captions along horizontal axis
        input_tensor = torch.cat((features.view(captions.shape[0],1,-1), embeds), 1)

        # here we go
        lstm_out, self.hidden_cell = self.lstm(input_tensor)
        drop_out = self.dropout(lstm_out)
        out = self.fc(drop_out)

        return out

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        # check that the input has expected shape
        assert inputs.shape[0] == 1 and inputs.shape[1] == 1 and \
               inputs.shape[2] == self.embed_size, f"Input shape should be (1,1,{self.embed_size})"

        # initialize the output sequence
        sequence = list()

        # initialize hidden and cell
        hidden_cell = (torch.zeros(self.num_layers, 1, self.hidden_size).to(self.device),
                       torch.zeros(self.num_layers, 1, self.hidden_size).to(self.device))

        # Inference
        for i in range(max_len):
            # forward pass with word and hidden cell from previos step
            features, hidden_cell = self.lstm(inputs, hidden_cell)
            features = self.dropout(features)
            features = self.fc(features)

            word_tensor = features.argmax(dim=2)
            sequence.append(word_tensor.squeeze().cpu().item())

            inputs = self.embedding(word_tensor)

        return sequence
