import math
import os
from functools import partial
from typing import List, Tuple

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from tqdm import tqdm
from datetime import datetime, timedelta
from datasets import ImageTextDataset
from vocabulary import Vocabulary
from metrics import bleu, meteor, rouge_l
from torchvision.models import swin_t, Swin_T_Weights
import torch.nn.functional as F


class ImageCaptioningModel(nn.Module):
    """img2seq模型"""

    def __init__(self, img_size, in_channels, patch_size, emb_size, target_vocab_size, num_layers, num_heads, expansion,
                 dropout, pretrained_embeddings=None):
        super(ImageCaptioningModel, self).__init__()
        
        # 替换ViT编码器为Swin Transformer编码器
        self.encoder = SwinTransformerEncoder(pretrained=True, emb_size=emb_size)

        # Transformer解码器
        self.decoder = TransformerDecoder(
            emb_size=emb_size,
            num_heads=num_heads,
            expansion=expansion,
            dropout=dropout,
            num_layers=num_layers,
            target_vocab_size=target_vocab_size,
            pretrained_embeddings=pretrained_embeddings,
        )

    def forward(self, images, captions, src_mask=None, tgt_mask=None):
        """

        :param images: [batch_size, in_channels, img_size, img_size]
        :param captions: [seq_length, batch_size]
        """
        encoder_output = self.encoder(images)  # [batch_size, n_patches + 1, emb_size]
        decoder_output = self.decoder(captions, encoder_output, src_mask, tgt_mask)

        # Step 3: 计算词汇的概率分布（假设输出是 logits）
        logits = decoder_output  # [seq_length, batch_size, target_vocab_size]

        # Step 4: 计算每个单词的对数概率
        action_log_probs = F.log_softmax(logits, dim=-1)  # [seq_length, batch_size, target_vocab_size]

        return decoder_output, action_log_probs  # 返回解码器输出和 action_log_probs
        #return decoder_output  # [seq_length, batch_size, target_vocab_size]
    def visualize(self):
        att_weights = self.decoder.layers[-1].encoder_decoder_att
        if att_weights is not None:
            return att_weights[:, -1, 1:]
        
def train(model, train_loader, criterion, optimizer, mask_func, save_path, device, epochs=10,
          save_interval=120, pretrained_weights=None, experiment_name='experiment'):

    os.makedirs(os.path.split(save_path)[0], exist_ok=True)

    if pretrained_weights:
        model.load_state_dict(torch.load(pretrained_weights))

    # 记录每个iteration的loss
    all_losses = []

    # training loop
    p_bar = tqdm(range(epochs))
    model = model.to(device)
    save_interval = timedelta(seconds=save_interval)
    model.train()

    for epoch in p_bar:
        running_loss = 0.0
        last_save_time = datetime.now()

        for batch_idx, (image, seq, seq_len) in enumerate(train_loader):
            image = image.to(device)  # (batch, c, img_sz, img_sz)
            seq = seq.to(device)  # (batch, seq_len + 1)

            input_seq = seq[:, :-1]  # (batch, seq_len)
            target_seq = seq[:, 1:]  # (batch, seq_len)

            # 开始训练
            optimizer.zero_grad()
            tgt_mask = mask_func(input_seq)

            prediction, action_log_probs = model(image, input_seq, tgt_mask=tgt_mask)  # (batch, seq_len, vocabulary_size)
            batch_size, _, vocab_size = prediction.shape
            loss = criterion(prediction.view(-1, vocab_size), target_seq.contiguous().view(-1))
            loss.backward()
            optimizer.step()

            # 保存每个iteration的loss
            all_losses.append(loss.item())

            # autosave
            if datetime.now() - last_save_time > save_interval:
                last_save_time = datetime.now()
                torch.save(model.state_dict(), save_path)

            # 记录结果
            running_loss += loss.item()
            p_bar.set_postfix(progress=f'{(batch_idx + 1)} / {len(train_loader)}',
                              loss=f'{running_loss / (batch_idx + 1):.4f}',
                              last_save_time=last_save_time)
    # 绘制损失图
    plt.figure(figsize=(10, 6))
    plt.plot(all_losses, label="Training Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.savefig(f'{experiment_name}_training_loss.png')
    plt.show()
