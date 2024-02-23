import sys
sys.path.append('/home/broiron/Desktop/TPoS/')
import clip
import random
import argparse

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from datasets_final import UnAVCurationDataset,UnAVCurationTestDataset
from model_final import Mapping_Model, Audio_Encoder, FrozenCLIPTextEmbedder, copyStateDict # text encoder: FrozenCLIPEmbedder
from models.model.audioclip import AudioCLIP

import math
import time
import os
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser(description="Audio Text Clip Implementation")

parser.add_argument("--epochs", default=50, type=int,
                help="epochs of training")
parser.add_argument("--batch_size", default=1, type=int,
                help="batch size of training") # default: 150
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.8685, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--step_size', default=1, type=float,
                    help='Step size for SGD')
parser.add_argument('--num_workers', default=16, type=int,
                    help='Number of workers used in dataloading')        

args = parser.parse_args()

os.makedirs("../pretrained_models/",exist_ok=True)

if __name__ == "__main__":
    random.seed(42)
    unav_dataset = UnAVCurationDataset()
    print(f"trainset length: {unav_dataset.__len__()}") # 8
    unav_test_dataset = UnAVCurationTestDataset()
    print(f"testset length: {unav_test_dataset.__len__()}") # 2

    train_dataset=unav_dataset
    validation_dataset=unav_test_dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ngpu=len(device)
    #clip_model, _ = clip.load("ViT-L/14", device=device) # 768
    clip_model, _ = clip.load("RN50", device=device) # 1024

    # 필요 없음
    # model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    # tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True)

    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True)

    
    audioencoder = Audio_Encoder(batch_size = args.batch_size, ngpus = torch.cuda.device_count())
    # audioencoder = AudioCLIP() # __init__ default로
    audioencoder = nn.DataParallel(audioencoder).to(device)
    map_model = Mapping_Model()
    map_model = nn.DataParallel(map_model).to(device)
    mse_loss = torch.nn.MSELoss()
    clip_1024 = FrozenCLIPTextEmbedder() # annotation에 있는 text label을 encode하기 위해 사용
    optimizer = optim.SGD(audioencoder.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    map_optimizer = optim.Adam(map_model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.1, step_size_up=5, mode="triangular")
    ce = torch.nn.CrossEntropyLoss()
    min_validation_loss_value = 50000

    for epoch in range(args.epochs):
        start = time.time()
        train_loss_value, validation_loss_value = 0, 0
        audioencoder.train()
        map_model.train()
        result_loss = 0

        for idx, (batch_audio, audio_aug, batch_text) in tqdm(enumerate(train_dataloader), desc='Training: '):
            audio_embedding = audioencoder(batch_audio.cuda())
            # print(f'audio embedding shape: {audio_embedding.shape}') # [1, 1024]

            text_tokens = torch.cat([clip.tokenize(text) for text in batch_text])

            with torch.no_grad():
                clip_1024_data = torch.cat([clip_1024(text) for text in batch_text])
                text_embedding1 = clip_model.encode_text(text_tokens.to(device)).float()
                text_embedding1 = text_embedding1 / text_embedding1.norm(dim=-1, keepdim=True)

            optimizer.zero_grad()
            map_optimizer.zero_grad()

            map_result = map_model(audio_embedding.clone().unsqueeze(1)) # text embedding과 유사해지게 차원 조정
            print(f'MLP 통과 후: {map_result.shape}')

            loss = 0
            loss_list = []

            label = torch.arange(args.batch_size, dtype=torch.long).cuda()
            
            # torch.cuda.amp.autocast 나중에 적용하기
            map_result = map_result.float()
            clip_1024_data = clip_1024_data.float()
            
            result_loss = mse_loss(map_result, clip_1024_data) # MSE
            loss += result_loss

            loss_list.append(result_loss.item())

            loss.backward()
            
            optimizer.step()
            map_optimizer.step()
            
            train_loss_value += loss.item()

        
            if idx % 100 == 0:
                #print("VGG, Batch : {:3d} , total loss : {:.3f}, ".format(idx, loss.item()))
                print(f'Batch : {idx}')

                for i,loss_value in enumerate(loss_list):
                    print(f"loss_{i} : {loss_value:.6f}")
        scheduler.step()

        audioencoder.eval()
        map_model.eval()
        
        
        print("Validation !")
        for idx, (batch_audio, audio_aug, batch_text) in tqdm(enumerate(validation_dataloader), desc='Validation: '):
            
            with torch.no_grad():
                audio_embedding_val = audioencoder(batch_audio.cuda())
                print(f'audio embedding: {audio_embedding.shape}') 
                # 일단 augmentation은 따로 하지 않으니까 아래 코드 주석
                # audio_embedding_aug1, audio_embedding_aug2, beta_aug_t  = audioencoder(batch_audio_aug.cuda())

                text_tokens = torch.cat([clip.tokenize(text) for text in batch_text])

                clip_1024_data = torch.cat([clip_1024(text) for text in batch_text])
                text_embedding1 = clip_model.encode_text(text_tokens.to(device)).float()
                text_embedding1 = text_embedding1 / text_embedding1.norm(dim=-1, keepdim=True)

                optimizer.zero_grad()
                map_optimizer.zero_grad()

                map_result = map_model(audio_embedding.clone().unsqueeze(1))
                #print(f'MLP 통과 후: {map_result.shape}')

                torch.autograd.set_detect_anomaly(True)
                loss = 0
                loss_list = []

                label = torch.arange(args.batch_size, dtype=torch.long).cuda()

                map_result = map_result.float() # add
                clip_1024_data = clip_1024_data.float() # add

                result_loss = mse_loss(map_result, clip_1024_data) # MSE
                loss += result_loss 

                loss_list.append(result_loss.item())
                                        
            validation_loss_value += loss.item()
            if idx % 100 == 0:
                print("VGG, Batch : {:3d} , total loss : {:.3f}".format(idx, loss.item()))
        
        print("Epoch : {:2d} , train loss : {:.5f}, validation loss : {:.5f}, Time : {}".format(epoch, train_loss_value / len(train_dataloader), validation_loss_value / len(validation_dataloader), time.time() - start))
        with open("../pretrained_models/loss.txt", "a") as f:
                    f.write("\n\nEpoch : {:2d} , train loss : {:.5f}, validation loss : {:.5f}, Time : {}".format(epoch, train_loss_value / len(train_dataloader), validation_loss_value / len(validation_dataloader), time.time() - start))
        
        if min_validation_loss_value > validation_loss_value:
            save_path = "../pretrained_models/audio_encoder_" + str(epoch) + ".pth"
            torch.save(audioencoder.state_dict(), save_path)
            save_path2 = "../pretrained_models/map_model_" + str(epoch) + ".pth"
            torch.save(map_model.state_dict(), save_path2)
            min_validation_loss_value = validation_loss_value
