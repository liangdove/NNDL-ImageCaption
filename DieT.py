import math
import os
from functools import partial
from typing import List, Tuple
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from tqdm import tqdm
from datetime import datetime, timedelta
from datasets import ImageTextDataset
from vocabulary import Vocabulary
from metrics import bleu, meteor, rouge_l
from FUYUapi import get_file_content_as_base64,get_access_token
import base64
import urllib
import requests
import json
from openai import OpenAI
from tongyi import encode_image,initialize_client,process_image


API_KEY = "jSHDtwarvVTHtFVxQs8lkFF8"
SECRET_KEY = "CwF6YNuCF5zF4ReaIZphBBwi4oVkOY5S"
cibiao={"the": 107, "upper": 108, "clothing": 109, "has": 4, "long": 5, "sleeves": 6, ",": 7, "cotton": 8, "fabric": 9, "and": 10, "solid": 11, "color": 12, "patterns": 13, ".": 14, "neckline": 15, "of": 16, "it": 17, "is": 18, "v": 19, "-": 20, "shape": 21, "lower": 22, "length": 23, "denim": 24, "this": 25, "lady": 26, "also": 27, "wears": 28, "an": 29, "outer": 30, "with": 31, "complicated": 32, "female": 33, "wearing": 34, "a": 35, "ring": 36, "on": 37, "her": 38, "finger": 39, "neckwear": 40, "tank": 41, "shirt": 42, "no": 43, "chiffon": 44, "graphic": 45, "round": 46, "person": 47, "pants": 48, "are": 49, "top": 50, "woman": 51, "trousers": 52, "there": 53, "belt": 54, "accessory": 55, "wrist": 56, "sweater": 57, "lattice": 58, "three": 59, "point": 60, "pure": 61, "in": 62, "his": 63, "neck": 64, "sleeve": 65, "plaid": 66, "its": 67, "lapel": 68, "socks": 69, "shoes": 70, "suspenders": 71, "short": 72, "t": 73, "shorts": 74, "crew": 75, "sleeveless": 76, "floral": 77, "hat": 78, "pair": 79, "quarter": 80, "head": 81, "waist": 82, "leather": 83, "pattern": 84, "cut": 85, "off": 86, "medium": 87, "knitting": 88, "gentleman": 89, "other": 90, "mixed": 91, "stripe": 92, "skirt": 93, "striped": 94, "sunglasses": 95, "guy": 96, "stand": 97, "man": 98, "square": 99, "leggings": 100, "furry": 101, "block": 102, "glasses": 103, "hands": 104, "or": 105, "clothes": 106, "<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
class PatchEmbedding(nn.Module):
    """ViT嵌入层，通过将原始图像分为若干个小块，分别嵌入，然后展平为序列"""

    def __init__(self, in_channels: int, patch_size: int, emb_size: int, img_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.projection = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.projection(x)  # Shape: (batch_size, emb_size, n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)  # Shape: (batch_size, emb_size, n_patches)
        x = x.transpose(1, 2)  # Shape: (batch_size, n_patches, emb_size)
        return x


class FeedForward(nn.Module):
    """编码器、解码器点对点前馈层"""

    def __init__(self, emb_size: int, expansion: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(emb_size, expansion * emb_size)
        self.fc2 = nn.Linear(expansion * emb_size, emb_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = nn.ReLU()(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """ViT编码器层"""

    def __init__(self, emb_size: int, num_heads: int, expansion: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        self.attention = nn.MultiheadAttention(emb_size, num_heads, dropout)
        self.feed_forward = FeedForward(emb_size, expansion, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attention, _ = self.attention(x, x, x, mask)
        x = self.norm1(attention + x)
        forward = self.feed_forward(x)
        x = self.norm2(forward + x)
        return x


class VisionTransformerEncoder(nn.Module):
    """ViT编码器"""

    def __init__(self, in_channels: int, patch_size: int, img_size: int, emb_size: int, num_layers: int, num_heads: int,
                 expansion: int, dropout: float):
        super().__init__()
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, emb_size, img_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positional_embedding = nn.Parameter(torch.randn(1, 1 + self.patch_embedding.n_patches, emb_size))
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(emb_size, num_heads, expansion, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        x = self.patch_embedding(x)
        batch_size, _, _ = x.shape
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x += self.positional_embedding
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, mask)

        return x


def create_masks(target_seq, pad_idx, num_heads):
    """创建目标序列注意力掩码"""
    # 创建三角掩码
    seq_len = target_seq.size(1)
    triangular_mask = torch.triu(torch.ones((seq_len, seq_len), device=target_seq.device) * float('-inf'), diagonal=1)

    # 创建PAD掩码
    pad_mask = (target_seq == pad_idx).to(target_seq.device)  # [batch_size, seq_len]
    pad_mask = pad_mask.unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, seq_len]

    # 合并掩码
    tgt_mask = triangular_mask.unsqueeze(0).expand(pad_mask.size(0), -1, -1)  # [batch_size, seq_len, seq_len]
    tgt_mask = tgt_mask.masked_fill(pad_mask, float('-inf'))

    # 调整掩码形状以适应多头注意力
    tgt_mask = tgt_mask.repeat_interleave(num_heads, dim=0)  # [batch_size * num_heads, seq_len, seq_len]

    return tgt_mask


class DecoderLayer(nn.Module):
    """transformer解码器层"""

    def __init__(self, emb_size, num_heads, expansion, dropout):
        super(DecoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        self.norm3 = nn.LayerNorm(emb_size)
        self.self_attention = nn.MultiheadAttention(emb_size, num_heads, dropout=dropout)
        self.encoder_attention = nn.MultiheadAttention(emb_size, num_heads, dropout=dropout)
        self.feed_forward = FeedForward(emb_size, expansion, dropout)
        self.dropout = nn.Dropout(dropout)
        self.encoder_decoder_att = None  # (batch, seq_len, image_embed_size)

    def forward(self, x, enc_out, src_mask, trg_mask):
        # Self Attention
        x = x.transpose(0, 1)  # Change shape to [seq_length, batch_size, emb_size]
        enc_out = enc_out.transpose(0, 1)

        attention_output, _ = self.self_attention(x, x, x, attn_mask=trg_mask)
        query = self.dropout(self.norm1(attention_output + x))

        # Encoder-Decoder Attention
        attention_output, self.encoder_decoder_att = self.encoder_attention(query, enc_out, enc_out, attn_mask=src_mask)
        # print(self.encoder_decoder_att.shape)  # (batch, seq_len, image_embed_size)
        query = self.dropout(self.norm2(attention_output + query))

        # Change shape back to [batch_size, seq_length, emb_size]
        query = query.transpose(0, 1)

        # Feed Forward
        out = self.feed_forward(query)
        out = self.dropout(self.norm3(out + query))

        return out


class PositionalEncoding(nn.Module):
    """目标序列正余弦位置编码"""

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Inject position encoding"""
        x = x + self.pe[:x.size(0), :]
        return x


class TransformerDecoder(nn.Module):
    """Transformer解码器层"""

    def __init__(self, emb_size, num_heads, expansion, dropout, num_layers, target_vocab_size,
                 pretrained_embeddings=None):
        super(TransformerDecoder, self).__init__()
        self.emb_size = emb_size
        self.word_embedding = nn.Embedding(target_vocab_size, emb_size)

        if pretrained_embeddings is not None:
            assert pretrained_embeddings.shape == (target_vocab_size, emb_size), "预训练嵌入向量尺寸不匹配"
            self.word_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(pretrained_embeddings), freeze=False)
        else:
            self.word_embedding = nn.Embedding(target_vocab_size, emb_size)

        self.positional_encoding = PositionalEncoding(emb_size)
        self.layers = nn.ModuleList([DecoderLayer(emb_size, num_heads, expansion, dropout) for _ in range(num_layers)])
        self.fc_out = nn.Linear(emb_size, target_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        x = self.dropout(self.word_embedding(x))
        x = self.positional_encoding(x)

        for layer in self.layers:
            x = layer(x, enc_out, src_mask, trg_mask)

        out = self.fc_out(x)
        return out

class ImageCaptioningModel(nn.Module):
    """img2seq模型"""

    def __init__(self, img_size, in_channels, patch_size, emb_size, target_vocab_size, num_layers, num_heads, expansion,
                 dropout, pretrained_embeddings=None):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = VisionTransformerEncoder(in_channels=in_channels,
                                                patch_size=patch_size,
                                                img_size=img_size,
                                                emb_size=emb_size,
                                                num_layers=num_layers,
                                                num_heads=num_heads,
                                                expansion=expansion,
                                                dropout=dropout)
        self.decoder = TransformerDecoder(emb_size=emb_size,  # 简单起见，编码器图像块与解码器文本嵌入使用相同的嵌入维度
                                          num_heads=num_heads,
                                          expansion=expansion,
                                          dropout=dropout,
                                          num_layers=num_layers,
                                          target_vocab_size=target_vocab_size,
                                          pretrained_embeddings=pretrained_embeddings)

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

import timm
from torchvision import models
class ImageCaptioningModelT(nn.Module):
    """img2seq模型，加入DeiT知识蒸馏"""
    def __init__(self, img_size, in_channels, patch_size, emb_size, target_vocab_size, num_layers, num_heads, expansion,
                 dropout, pretrained_embeddings=None, teacher_model=None):
        super(ImageCaptioningModelT, self).__init__()
        
        self.teacher_model = teacher_model if teacher_model else timm.create_model('deit_base_patch16_224', pretrained=True)
        # Freezing the teacher model parameters to avoid training them
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        self.encoder = VisionTransformerEncoder(in_channels=in_channels,
                                                patch_size=patch_size,
                                                img_size=img_size,
                                                emb_size=emb_size,
                                                num_layers=num_layers,
                                                num_heads=num_heads,
                                                expansion=expansion,
                                                dropout=dropout)
        
        self.decoder = TransformerDecoder(emb_size=emb_size,  # 简单起见，编码器图像块与解码器文本嵌入使用相同的嵌入维度
                                          num_heads=num_heads,
                                          expansion=expansion,
                                          dropout=dropout,
                                          num_layers=num_layers,
                                          target_vocab_size=target_vocab_size,
                                          pretrained_embeddings=pretrained_embeddings)

    def forward(self, images, captions, src_mask=None, tgt_mask=None):
        encoder_output = self.encoder(images)  # [batch_size, n_patches + 1, emb_size]
        decoder_output = self.decoder(captions, encoder_output, src_mask, tgt_mask)

        # Step 3: 计算词汇的概率分布（假设输出是 logits）
        logits = decoder_output  # [seq_length, batch_size, target_vocab_size]

        # Step 4: 计算每个单词的对数概率
        action_log_probs = F.log_softmax(logits, dim=-1)  # [seq_length, batch_size, target_vocab_size]

        return decoder_output, action_log_probs

    def distill_loss(self, student_logits, teacher_logits, T=2.0, alpha=0.5):
        """
        定义蒸馏损失函数
        :param student_logits: 学生模型的logits (预测值)
        :param teacher_logits: 教师模型的logits (目标值)
        :param T: 温度系数
        :param alpha: 蒸馏损失与传统目标损失的加权系数
        """
        # 获取当前设备 (假设你使用的是 cuda:0)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 确保输入张量都在同一个设备上
        student_logits = student_logits.to(device)
        teacher_logits = teacher_logits.to(device)

        # 如果你需要将 teacher_logits 通过线性层映射到学生模型的目标词汇表大小
        # 假设教师模型的输出维度为 1000，学生模型的输出维度为 110
        teacher_linear = torch.nn.Linear(1000, 110).to(device)

        # 使用线性变换将教师模型输出从 1000 映射到 110（目标词汇表大小）
        teacher_soft_mapped = teacher_linear(teacher_logits)

        # 扩展后的教师输出形状：[batch_size, seq_length, target_vocab_size]
        teacher_soft_expanded = teacher_soft_mapped.unsqueeze(1).expand(-1, student_logits.size(1), -1)  # 扩展为与学生相同的 seq_length

        # 计算 softmax
        teacher_soft_expanded = F.softmax(teacher_soft_expanded / T, dim=-1)
        student_soft = F.softmax(student_logits / T, dim=-1)

        # 计算蒸馏损失 (KL散度)
        distillation_loss = F.kl_div(student_soft.log(), teacher_soft_expanded, reduction='batchmean') * (T * T)

        return distillation_loss
    def hard_distill_loss(self, student_logits, teacher_logits):
        """
        定义硬蒸馏损失函数
        :param student_logits: 学生模型的logits (预测值)
        :param teacher_logits: 教师模型的logits (目标值)
        """
        # 获取教师的硬标签（最大值对应的索引）
        teacher_labels = torch.argmax(teacher_logits, dim=-1)

        # 使用交叉熵损失计算学生模型与教师硬标签之间的差异
        distillation_loss = F.cross_entropy(student_logits, teacher_labels)
        
        return distillation_loss
    def visualize(self):
        att_weights = self.decoder.layers[-1].encoder_decoder_att
        if att_weights is not None:
            return att_weights[:, -1, 1:]


# def decode_caption(sequence, vocab):
#     """
#     将token序列解码为句子（字符串）
#     """
#     words = [vocab.inv[idx.item()] for idx in sequence]  # Assuming vocab.inv is a dictionary mapping index to word
#     return ' '.join(words)
from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score

def reward_func(prediction, target_seq, vocab):
    """
    计算生成文本与目标文本之间的评测指标（如 BLEU、ROUGE-L、METEOR），并返回每个样本的评测分数列表。
    """
    # 预测序列解码
    prediction_idx = prediction.argmax(dim=-1)  # shape: (batch_size, seq_len)
    decoded_pretexts = []
    for seq in prediction_idx:
        decoded_text = vocab.decode(seq.tolist())  # decode 函数接收的是 List[int]
        decoded_pretexts.append(decoded_text)
    
    # 真实目标序列解码
    decoded_tartexts = []
    for seq in target_seq:
        decoded_text = vocab.decode(seq.tolist())
        decoded_tartexts.append(decoded_text)
    
    # 计算 BLEU
    bleu_scores = []
    for ref, hyp in zip(decoded_tartexts, decoded_pretexts):
        bleu_score = corpus_bleu([[ref.split()]], [hyp.split()])
        bleu_scores.append(bleu_score)
    
    # 计算 METEOR
    meteor_scores = []
    for ref, hyp in zip(decoded_tartexts, decoded_pretexts):
        meteor_score_value = meteor_score([ref.split()], hyp.split())
        meteor_scores.append(meteor_score_value)
    
    # 计算 ROUGE-L
    rouge_l_scores = []
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    for ref, hyp in zip(decoded_tartexts, decoded_pretexts):
        rouge_l_score = scorer.score(ref, hyp)['rougeL'].fmeasure
        rouge_l_scores.append(rouge_l_score)

    # 返回包含每个样本的评测指标列表
    rewards = {
        "BLEU": bleu_scores,
        "ROUGE-L": rouge_l_scores,
        "METEOR": meteor_scores
    }
    #print(rewards)
    return rewards


def train(model, train_loader, criterion, optimizer, mask_func, vocab, reward_func, save_path, device, epochs=1,
                            save_interval=120, pretrained_weights=None, experiment_name='experiment'):
    if pretrained_weights:
        model.load_state_dict(torch.load(pretrained_weights))

    # training loop
    p_bar = tqdm(range(epochs))
    model = model.to(device)
    save_interval = timedelta(seconds=save_interval)
    model.train()

    # 用于存储每个iteration的损失
    iteration_losses = []

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

            # 获取学生模型的预测和教师模型的logits
            prediction, action_log_probs = model(image, input_seq, tgt_mask=tgt_mask)  # (batch, seq_len, vocabulary_size)
            batch_size, _, vocab_size = prediction.shape

            # 获取教师模型的logits
            with torch.no_grad():
                teacher_logits = model.teacher_model(image)  # 假设教师模型输出logits
            # 计算损失
            # 常规损失
            loss = criterion(prediction.view(-1, vocab_size), target_seq.contiguous().view(-1))

            # 蒸馏损失
            distillation_loss = model.distill_loss(prediction, teacher_logits)

            # 总损失
            total_loss = loss + distillation_loss

            # 记录每个iteration的损失
            iteration_losses.append(total_loss.item())

            # 反向传播
            total_loss.backward()
            optimizer.step()

            # 自动保存模型
            if datetime.now() - last_save_time > save_interval:
                last_save_time = datetime.now()
                torch.save(model.state_dict(), save_path)

            # 记录结果
            running_loss += total_loss.item()
            p_bar.set_postfix(progress=f'{(batch_idx + 1)} / {len(train_loader)}',
                              loss=f'{total_loss:.4f}',
                              last_save_time=last_save_time)

    # 绘制每个iteration的损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(iteration_losses, marker='o', label='Training Loss per Iteration')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Loss per Iteration')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{os.path.split(save_path)[0]}/diet2_loss_curve_per_iteration.png')  # 保存图片
    plt.show()


def generate_by_beam_search(model, image, vocab, device, max_length, mask_func, beam_width=5):
    model = model.to(device)
    image = image.to(device)  # (1, channel, img_size, img_size)

    # 初始候选序列和分数
    sequences = [([vocab.start], 0)]  # 每个序列是(token_list, score)
    attn_images = []
    model.eval()
    for _ in range(max_length):
        all_candidates = []
        for seq, score in sequences:
            if seq[-1] == vocab.end:
                all_candidates.append((seq, score))
                continue

            seq_tensor = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
            tgt_mask = mask_func(seq_tensor)

            with torch.no_grad():
                output,action = model(image, seq_tensor, tgt_mask=tgt_mask)
                attn_images.append((model.visualize(), vocab.inv[seq_tensor[:, -1].item()]))
                # 抽取最后一个decoder层、第一个batch、序列最后一个的交叉注意力权重

            # 考虑top k个候选
            top_k_probs, top_k_indices = torch.topk(torch.softmax(output, dim=-1), beam_width, dim=-1)
            top_k_probs = top_k_probs[0, -1]
            top_k_indices = top_k_indices[0, -1]

            for k in range(beam_width):
                next_seq = seq + [top_k_indices[k].item()]
                next_score = score - torch.log(top_k_probs[k])  # 使用负对数似然作为分数
                all_candidates.append((next_seq, next_score))

        # 按分数排序并选出前k个
        ordered = sorted(all_candidates, key=lambda tup: tup[1])
        sequences = ordered[:beam_width]

    # 选择分数最高的序列
    best_seq = sequences[0][0]
    best_score = sequences[0][1].item()
    text = vocab.decode(best_seq)
    return text, best_score, attn_images


def fuyu(image):
    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/image2text/fuyu_8b?access_token=" + get_access_token()
    #encode_image=get_file_content_as_base64("D:\神经网络深度学习\deepfashion\FashionDescription-main\FashionDescription\data\deepfashion-multimodal\images\WOMEN-Tees_Tanks-id_00007979-04_4_full.jpg",False)
    #print(encode_image)
    # image 可以通过 get_file_content_as_base64("C:\fakepath\MEN-Denim-id_00000080-01_7_additional.jpg",False) 方法获取
    # 单词列表
    word_list = [
        "the", "upper", "clothing", "has", "long", "sleeves", "cotton", "fabric", "and", 
        "solid", "color", "patterns", "neckline", "of", "it", "is", "v", "shape", "lower", 
        "length", "denim", "this", "lady", "also", "wears", "an", "outer", "with", "complicated", 
        "female", "wearing", "a", "ring", "on", "her", "finger", "neckwear", "tank", "shirt", 
        "no", "chiffon", "graphic", "round", "person", "pants", "are", "top", "woman", "trousers", 
        "there", "belt", "accessory", "wrist", "sweater", "lattice", "three", "point", "pure", 
        "in", "his", "neck", "sleeve", "plaid", "its", "lapel", "socks", "shoes", "suspenders", 
        "short", "t", "shorts", "crew", "sleeveless", "floral", "hat", "pair", "quarter", "head", 
        "waist", "leather", "pattern", "cut", "off", "medium", "knitting", "gentleman", "other", 
        "mixed", "stripe", "skirt", "striped", "sunglasses", "guy", "stand", "man", "square", 
        "leggings", "furry", "block", "glasses", "hands", "or", "clothes"
    ]
    
    # 动态生成 prompt 字符串
    prompt = f"Simply describe what the person is wearing using only words in this list: {', '.join(word_list)} ,Be sure to use the words listed and avoid using other words outside the list."
    payload = json.dumps({
            "prompt": prompt,
            "image": get_file_content_as_base64(image,False)
            })
    headers = {
            'Content-Type': 'application/json'
        }
    
    try:
        # 发起 POST 请求
        response = requests.post(url, headers=headers, data=payload)
        
        # 尝试解析返回的 JSON
        result = response.json()
        
        # 检查错误码，如果存在错误码则抛出异常
        if "error_code" in result and result["error_code"] != 0:
            error_code = result["error_code"]
            error_msg = result.get("error_msg", "No error message provided.")
            raise Exception(f"Error {error_code}: {error_msg}")
        
        # 如果没有错误，返回结果
        return result["result"]

    except requests.exceptions.RequestException as e:
            # 捕获 HTTP 请求的异常（网络问题等）
            print(f"HTTP request failed: {e}")
    except json.JSONDecodeError:
            # 捕获 JSON 解码错误
            print("Failed to decode JSON response.")
    except Exception as e:
            # 捕获其他异常
            print(f"An error occurred: {e}")

    return None  # 如果发生错误，返回 None

def evaluate(model, test_set, vocabulary, mask_func, device, max_length, beam_width=5):
    """评估模型性能"""
    model.eval()
    metrics = np.zeros(3)
    p_bar = tqdm(range(len(test_set)), desc='Deit知识蒸馏评估')
    #client = initialize_client()
    for i in p_bar:
        path, caption = test_set.get_pair(i)
        image, _, _ = test_set[i]
        caption_generated, score, _ = generate_by_beam_search(model, image.unsqueeze(0), vocabulary, device, max_length,
                                                               mask_func, beam_width)
        
        
        # caption_generated = fuyu(path)
        # caption_generated = process_image(client, path)
        # print("caption_generated:",caption_generated)
        #if(caption_generated):
        metrics += np.array([
                             bleu(caption_generated, caption, vocabulary),
                             rouge_l(caption_generated, caption, vocabulary),
                             meteor(caption_generated, caption, vocabulary)])
        value = metrics / (i + 1)
        p_bar.set_postfix(BLEU=value[0], ROUGE_L=value[1], METEOR=value[2])


if __name__ == "__main__":
    # =================== Example parameters for the model ===================
    img_size = 224  # Size of the input image
    in_channels = 3  # Number of input channels (for RGB images)
    patch_size = 16  # Size of each patch
    emb_size = 96  # Embedding size
    num_layers = 6  # Number of layers in both encoder and decoder
    num_heads = 8  # Number of attention headsOpenhimer
    expansion = 4  # Expansion factor for feed forward network

    # =================== Train Config ===================
    dropout = 0.1  # Dropout rate
    lr = 5e-4  # Learning rate
    epochs = 1
    batch_size = 64  # Batch size
    seq_length = 128  # Max length of the caption sequence
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    save_path = 'models/predeit_transformer.pth'
    experiment_name = 'fashion_description'
    vocabulary_path = 'vocabulary/vocab.json'
    word2vec_cache_path = 'vocabulary/word2vec.npy'
    dataset_root = 'data/deepfashion-multimodal'
    train_labels_path = 'data/deepfashion-multimodal/train_captions.json'
    test_labels_path = 'data/deepfashion-multimodal/test_captions.json'
    teacher_model= 'models/model_transformer.pth'
    # =================== Vocabulary and Image transforms ===================
    vocabulary = Vocabulary(vocabulary_path)
    transform = Compose([
        Resize((img_size, img_size)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    mask_func = partial(create_masks, pad_idx=vocabulary.pad, num_heads=num_heads)

    # =================== Initialize the model ===================
    # modeltt = ImageCaptioningModelT(img_size, in_channels, patch_size, emb_size, len(vocabulary), num_layers, num_heads,
    #                              expansion, dropout,
    #                              pretrained_embeddings=vocabulary.get_word2vec(cache_path=word2vec_cache_path),teacher_model=None)

    # modelt = ImageCaptioningModelT(img_size, in_channels, patch_size, emb_size, len(vocabulary), num_layers, num_heads,
    #                              expansion, dropout,
    #                              pretrained_embeddings=vocabulary.get_word2vec(cache_path=word2vec_cache_path),teacher_model=None)

    # if os.path.exists(save_path):
    #      modelt.load_state_dict(torch.load(save_path))
    model = ImageCaptioningModelT(img_size, in_channels, patch_size, emb_size, len(vocabulary), num_layers, num_heads,
                                 expansion, dropout,
                                 pretrained_embeddings=vocabulary.get_word2vec(cache_path=word2vec_cache_path),teacher_model= None)
    
    if os.path.exists(save_path):
         model.load_state_dict(torch.load(save_path))

    # =================== Prepare for Training ===================

    train_set = ImageTextDataset(dataset_root,
                                 train_labels_path,
                                 vocabulary=vocabulary,
                                 max_seq_len=seq_length,
                                 transform=transform,
                                 max_cache_memory=32 * 1024 ** 3)

    train_loader = DataLoader(train_set, batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # =================== Start Training ===================
    # train(model, train_loader, criterion, optimizer, mask_func, vocabulary, reward_func, save_path='models/predeit_transformer.pth',
    #          epochs=epochs, device=device, experiment_name=experiment_name)

    #  =================== Model Inference ===================
    test_set = ImageTextDataset(dataset_root,
                                test_labels_path,
                                vocabulary=vocabulary,
                                max_seq_len=seq_length,
                                transform=transform)


    evaluate(model, test_set, vocabulary, mask_func, device, seq_length, beam_width=5)  # 在测试集上评估模型