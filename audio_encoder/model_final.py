import torch 
import timm
import clip # 1024로 바꾸기 위해
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List
from tqdm import tqdm

import json

class FrozenCLIPTextEmbedder(torch.nn.Module):
    """
    Uses the CLIP transformer encoder for text.
    available models = ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
    """
    def __init__(self, version='RN50', device="cuda", max_length=77, n_repeat=1, normalize=True):
        super().__init__()
        self.model, _ = clip.load(version, device=device)
        self.max_length = max_length
        self.n_repeat = n_repeat
        self.normalize = normalize

    def forward(self, text: Union[str, List[str]]):
        device = next(self.model.parameters()).device
        tokens = clip.tokenize(text, context_length=self.max_length).to(device)
        z = self.model.encode_text(tokens)
        if self.normalize:
            z = z / z.norm(dim=-1, keepdim=True)
        return z

    def encode(self, text: Union[str, List[str]]):
        z = self(text)
        if z.ndim == 2:
            z = z[:, None, :]
        z = z.repeat(1, self.max_length, 1) # 두 번째 인자가 n.repeat 이었던 것을 max_length로 수정
        return z
        

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError
    
    
def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


    
def load_annotations(json_file):
    with open(json_file, 'r') as f:
        annotations = json.load(f)
    return annotations



def process_annotations(annotations, clip_text_embedder):
    for video_id, video_info in tqdm(annotations['database'].items(), desc="Processing GT annotations"):
        for annotation in video_info['annotations']:
            start_time, end_time = annotation['segment']
            label = annotation['label']
            text_description = label 
            text_embedding = clip_text_embedder.encode(text_description)
            # print(f'Label: {label}, Embedding: {text_embedding.shape}') # 어느 동영상에 해당되는 label인지도 알 수 있게 코드 수정
    
    
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_text_embedder = FrozenCLIPTextEmbedder(device=device)

json_file = '/home/broiron/Desktop/TPoS/dataset/unav100_annotations.json' # UnAV-100 annotations.json 가져오기
annotations = load_annotations(json_file)
process_annotations(annotations, clip_text_embedder)


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError
    
def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


'''
# 원본
class Mapping_Model(nn.Module):
    def __init__(self, max_length=77):
        super().__init__()
        self.max_length = max_length-1
        self.linear1 = torch.nn.Linear(768,self.max_length//7*768)
        self.linear2 = torch.nn.Linear(self.max_length//7*768,self.max_length*768)
        self.act = torch.nn.GELU()
        self.drop = torch.nn.Dropout(0.2)
        
    def forward(self, x):
        return self.act(self.drop(self.linear2(self.act(self.drop(self.linear1(x)))))).reshape(x.shape[0],self.max_length,768)
'''   

class Mapping_Model(nn.Module):
    def __init__(self, sequence_length=77, input_dim=1024, output_dim=1024):
        super().__init__()
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.linear1 = nn.Linear(input_dim, input_dim * 2)
        self.linear2 = nn.Linear(input_dim * 2, sequence_length * output_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.drop(self.act(self.linear1(x))) 
        x = self.drop(self.act(self.linear2(x)))
        x = x.view(-1, self.sequence_length, self.output_dim)  
        return x


# LSTM을 일단 사용하지 않으니까 관련 부분 제거
class Audio_Encoder(nn.Module):
    def __init__(self, sequence_length=5, input_size=768, backbone_name="resnet18", batch_size=320, ngpus=4):
        super(Audio_Encoder, self).__init__()

        self.sequence_length = sequence_length
        self.input_size = input_size
        self.batch_size = batch_size
        self.ngpus = ngpus
        self.size = int(self.batch_size / self.ngpus)

        self.conv = nn.Conv2d(1, 3, (3, 3), padding=1)
        self.feature_extractor = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        self.final_linear = nn.Linear(self.feature_extractor.num_features, 1024)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, 1, x.size(2), x.size(3))
        x = self.conv(x)

        features = self.feature_extractor(x)
        output = self.final_linear(features)

        # Normalize
        output = F.normalize(output, p=2, dim=1)
        output = output.view(self.batch_size, self.sequence_length, -1)

        return output[:, -1, :]
    

'''
# 원본
class Audio_Encoder(nn.Module):

    def __init__(self, sequence_length=5, lstm_hidden_dim=768, input_size=768, hidden_size=768, num_layers=1,backbone_name="resnet18",batch_size=320, ngpus = 4):

        super(Audio_Encoder,self).__init__()

        self.sequence_length = sequence_length
        self.lstm_hidden_dim=lstm_hidden_dim
        
        
        self.T_A = nn.Linear(sequence_length*lstm_hidden_dim, 512)
        self.T_A2 = nn.Linear(self.sequence_length*lstm_hidden_dim, self.sequence_length*512)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.backbone_name = backbone_name
        self.num_layers = num_layers
        self.input_size = input_size
    
        self.hidden_size = hidden_size
        self.conv = torch.nn.Conv2d(1, 3, (3, 3))
        self.conv2 = torch.nn.Conv2d(1,77,(1,1)) 
        self.feature_extractor = timm.create_model(self.backbone_name, num_classes=self.input_size, pretrained=True)
    
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,num_layers=num_layers, batch_first=True)
        self.ngpus=ngpus
        self.batch_size=batch_size
        self.size=int(self.batch_size / self.ngpus)
    
        self.cnn = nn.Conv1d(768,1, kernel_size=1)

    def forward (self,x):

        a=torch.zeros(self.size,self.sequence_length,768).cuda()
        for i in range(self.sequence_length):
            a[:,i,:] = self.feature_extractor(self.conv(x[:,i,:,:].reshape(self.size,1,128,self.hidden_size//self.sequence_length)))
        x=a
        h_0 = Variable(torch.zeros( self.num_layers,x.size(0), self.hidden_size)).cuda()
        c_0 = Variable(torch.zeros( self.num_layers,x.size(0),  self.hidden_size)).cuda()
        self.lstm.flatten_parameters()
        output, (hn, cn) = self.lstm(x, (h_0, c_0))
        output = output/output.norm(dim=-1,keepdim=True)
        
        output_permute = output.permute(0,2,1)

        beta_t = self.cnn(output_permute).squeeze()

        beta_t=self.softmax(beta_t)

        out=output[:,0,:].mul(beta_t[:,0].reshape(self.size,-1))

        out=out.unsqueeze(1)


        for i in range(1,self.sequence_length):
            next_z=output[:,i,:].mul(beta_t[:,i].reshape(self.size,-1) )
            out=torch.cat([out,next_z.unsqueeze(1)],dim=1)

        return output[:,-1,:], out, beta_t
'''