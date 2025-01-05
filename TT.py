import math
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from tqdm import tqdm
from datetime import datetime, timedelta
from datasets import ImageTextDataset
from vocabulary import Vocabulary
from metrics import bleu, meteor, rouge_l

# 超参数配置
IMG_SIZE = 256
IN_CHANNELS = 3
TEXT_EMB_SIZE = 96
IMG_EMB_SIZE = 64
HIDDEN_SIZE = 256
DROPOUT = 0.1
LEARNING_RATE = 5e-4
EPOCHS = 500
BATCH_SIZE = 64
SEQ_LENGTH = 128

# 束搜索算法
def beam_search(model, image, vocab, device, max_length, beam_width=5):
    model.to(device)
    image = image.to(device)
    sequences = [([vocab.start], 0, ())]

    for _ in range(max_length):
        all_candidates = []
        for seq, score, h in sequences:
            if seq[-1] == vocab.end:
                all_candidates.append((seq, score, h))
                continue

            inp = torch.tensor([seq[-1]], dtype=torch.long, device=device).unsqueeze(0)
            with torch.no_grad():
                output, h_ = model(image, inp, *h)

            top_k_probs, top_k_indices = torch.topk(torch.softmax(output, dim=-1), beam_width, dim=-1)
            top_k_probs = top_k_probs[0, -1]
            top_k_indices = top_k_indices[0, -1]

            for k in range(beam_width):
                next_seq = seq + [top_k_indices[k].item()]
                next_score = score - torch.log(top_k_probs[k])
                all_candidates.append((next_seq, next_score, h_))

        ordered = sorted(all_candidates, key=lambda tup: tup[1])
        sequences = ordered[:beam_width]

    best_seq = sequences[0][0]
    best_score = sequences[0][1].item()
    return vocab.decode(best_seq), best_score

# CNN特征提取器
class CNNFeatureExtractor(nn.Module):
    def __init__(self, in_channels, emb_size):
        super(CNNFeatureExtractor, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, emb_size // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(emb_size // 2, emb_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((16, 16))
        )

    def forward(self, x):
        return self.feature_extractor(x)

# 前馈网络
class FeedForward(nn.Module):
    def __init__(self, emb_size, expansion, dropout):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(emb_size, expansion * emb_size)
        self.fc2 = nn.Linear(expansion * emb_size, emb_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = nn.ReLU()(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Transformer编码器层
class TransformerEncoderLayer(nn.Module):
    def __init__(self, emb_size, num_heads, expansion, dropout):
        super(TransformerEncoderLayer, self).__init__()
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

# Vision Transformer编码器
class VisionTransformerEncoder(nn.Module):
    def __init__(self, emb_size, num_layers, num_heads, expansion, dropout):
        super(VisionTransformerEncoder, self).__init__()
        self.linear_projection = nn.Linear(emb_size, emb_size)
        self.positional_embedding = nn.Parameter(torch.randn(1, 16 * 16, emb_size))
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(emb_size, num_heads, expansion, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        batch_size, channels, height, width = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.linear_projection(x) + self.positional_embedding
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, mask)

        return x

# 解码器层
class DecoderLayer(nn.Module):
    def __init__(self, emb_size, num_heads, expansion, dropout):
        super(DecoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        self.norm3 = nn.LayerNorm(emb_size)
        self.self_attention = nn.MultiheadAttention(emb_size, num_heads, dropout)
        self.encoder_attention = nn.MultiheadAttention(emb_size, num_heads, dropout)
        self.feed_forward = FeedForward(emb_size, expansion, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        x = x.transpose(0, 1)
        enc_out = enc_out.transpose(0, 1)
        attention_output, _ = self.self_attention(x, x, x, attn_mask=trg_mask)
        query = self.dropout(self.norm1(attention_output + x))
        attention_output, _ = self.encoder_attention(query, enc_out, enc_out, attn_mask=src_mask)
        query = self.dropout(self.norm2(attention_output + query))
        query = query.transpose(0, 1)
        out = self.feed_forward(query)
        out = self.dropout(self.norm3(out + query))
        return out

# Transformer解码器
class TransformerDecoder(nn.Module):
    def __init__(self, emb_size, num_heads, expansion, dropout, num_layers, target_vocab_size, pretrained_embeddings=None):
        super(TransformerDecoder, self).__init__()
        self.emb_size = emb_size
        self.word_embedding = nn.Embedding(target_vocab_size, emb_size)
        if pretrained_embeddings is not None:
            assert pretrained_embeddings.shape == (target_vocab_size, emb_size), "预训练嵌入向量尺寸不匹配"
            self.word_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(pretrained_embeddings), freeze=False)
        self.positional_encoding = PostionalEncode(emb_size)
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

# 图像字幕模型
class ImageCaptioningModel(nn.Module):
    def __init__(self, img_size, in_channels, patch_size, emb_size, target_vocab_size, num_layers, num_heads, expansion, dropout, pretrained_embeddings=None):
        super(ImageCaptioningModel, self).__init__()
        self.feature_extractor = CNNFeatureExtractor(in_channels, emb_size)
        self.encoder = VisionTransformerEncoder(emb_size=emb_size, num_layers=num_layers, num_heads=num_heads, expansion=expansion, dropout=dropout)
        self.decoder = TransformerDecoder(emb_size=emb_size, num_heads=num_heads, expansion=expansion, dropout=dropout, num_layers=num_layers, target_vocab_size=target_vocab_size, pretrained_embeddings=pretrained_embeddings)

    def forward(self, images, captions, src_mask=None, tgt_mask=None):
        cnn_features = self.feature_extractor(images)
        encoder_output = self.encoder(cnn_features, mask=src_mask)
        decoder_output = self.decoder(captions, encoder_output, src_mask, tgt_mask)
        return decoder_output

# 训练函数
def train(model, train_loader, criterion, optimizer, save_path, device, epochs=10, wait_plt_time=120, pretrained_weights=None, experiment_name='experiment'):
    os.makedirs(os.path.split(save_path)[0], exist_ok=True)
    if pretrained_weights:
        model.load_state_dict(torch.load(pretrained_weights))

    p_bar = tqdm(range(epochs))
    model = model.to(device)
    wait_plt_time = timedelta(seconds=wait_plt_time)
    model.train()

    for epoch in p_bar:
        running_loss = 0.0
        last_save_time = datetime.now()
        for batch_idx, (image, seq, seq_len) in enumerate(train_loader):
            image = image.to(device)
            seq = seq.to(device)
            input_seq = seq[:, :-1]
            target_seq = seq[:, 1:]

            optimizer.zero_grad()
            prediction, _ = model(image, input_seq)
            _, _, vocab_size = prediction.shape
            loss = criterion(prediction.view(-1, vocab_size), target_seq.contiguous().view(-1))
            loss.backward()
            optimizer.step()

            if datetime.now() - last_save_time > wait_plt_time:
                last_save_time = datetime.now()
                torch.save(model.state_dict(), save_path)

            running_loss += loss.item()
            p_bar.set_postfix(progress=f'{(batch_idx + 1)} / {len(train_loader)}', loss=f'{running_loss / (batch_idx + 1):.4f}', last_save_time=last_save_time)

# 评估函数
def evaluate(model, test_set, vocabulary, device, max_length, beam_width=5):
    model.eval()
    metrics = np.zeros(4)
    p_bar = tqdm(range(len(test_set)), desc='evaluating')
    for i in p_bar:
        _, caption = test_set.get_pair(i)
        image, _, _ = test_set[i]
        caption_generated, score = beam_search(model, image.unsqueeze(0), vocabulary, device, max_length, beam_width)
        metrics += np.array([score, bleu(caption_generated, caption, vocabulary), rouge_l(caption_generated, caption, vocabulary), meteor(caption_generated, caption, vocabulary)])
        value = metrics / (i + 1)
        p_bar.set_postfix(score=value[0], bleu=value[1], rouge_l=value[2], meteor=value[3])

# 主函数
def main():
    save_path = 'xx/model_TT.pth'
    experiment_name = 'xx'
    vocabulary_path = 'xx/vocab.json'
    word2vec_cache_path = 'xx/word2vec.npy'
    dataset_root = 'xx/deepfashion-multimodal'
    train_labels_path = 'xx/deepfashion-multimodal/train_captions.json'
    test_labels_path = 'xx/deepfashion-multimodal/test_captions.json'

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    vocabulary = Vocabulary(vocabulary_path)
    transform = Compose([
        Resize((IMG_SIZE, IMG_SIZE)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model = ImageCaptioningModel(len(vocabulary), IN_CHANNELS, IMG_EMB_SIZE, TEXT_EMB_SIZE, HIDDEN_SIZE, pretrained_embeddings=vocabulary.get_word2vec(cache_path=word2vec_cache_path)).to(device)
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=torch.device('cpu')))

    train_set = ImageTextDataset(dataset_root, train_labels_path, vocabulary=vocabulary, max_seq_len=SEQ_LENGTH, transform=transform, max_cache_memory=32 * 1024 ** 3)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train(model, train_loader, criterion, optimizer, save_path=save_path, device=device, epochs=EPOCHS, experiment_name=experiment_name)

    test_set = ImageTextDataset(dataset_root, test_labels_path, vocabulary=vocabulary, max_seq_len=SEQ_LENGTH, transform=transform, max_cache_memory=32 * 1024 ** 3)
    evaluate(model, test_set, vocabulary, device, SEQ_LENGTH, beam_width=5)

if __name__ == '__main__':
    main()