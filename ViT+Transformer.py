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
IMAGE_SIZE = 256
IN_CHANNELS = 3
TEXT_EMBED_SIZE = 96
IMAGE_EMBED_SIZE = 64
HIDDEN_SIZE = 256
DROPOUT_RATE = 0.1
LEARNING_RATE = 5e-4
NUM_EPOCHS = 500
BATCH_SIZE = 64
SEQUENCE_LENGTH = 128

class PatchEmbedding(nn.Module):
    """将图像分割为小块并嵌入"""

    def __init__(self, in_channels, patch_size, embed_size, img_size):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.projection = nn.Conv2d(in_channels, embed_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x

class FeedForwardNetwork(nn.Module):
    """前馈网络"""

    def __init__(self, embed_size, expansion, dropout):
        super().__init__()
        self.fc1 = nn.Linear(embed_size, expansion * embed_size)
        self.fc2 = nn.Linear(expansion * embed_size, embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = nn.ReLU()(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""

    def __init__(self, embed_size, num_heads, expansion, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.attention = nn.MultiheadAttention(embed_size, num_heads, dropout)
        self.feed_forward = FeedForwardNetwork(embed_size, expansion, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attention, _ = self.attention(x, x, x, mask)
        x = self.norm1(attention + x)
        forward = self.feed_forward(x)
        x = self.norm2(forward + x)
        return x

class VisionTransformerEncoder(nn.Module):
    """Vision Transformer编码器"""

    def __init__(self, in_channels, patch_size, img_size, embed_size, num_layers, num_heads, expansion, dropout):
        super().__init__()
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, embed_size, img_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_size))
        self.positional_embedding = nn.Parameter(torch.randn(1, 1 + self.patch_embedding.num_patches, embed_size))
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_size, num_heads, expansion, dropout)
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

def create_attention_masks(target_sequence, pad_index, num_heads):
    """创建目标序列的注意力掩码"""
    sequence_length = target_sequence.size(1)
    triangular_mask = torch.triu(torch.ones((sequence_length, sequence_length), device=target_sequence.device) * float('-inf'), diagonal=1)
    pad_mask = (target_sequence == pad_index).to(target_sequence.device)
    pad_mask = pad_mask.unsqueeze(1).expand(-1, sequence_length, -1)
    target_mask = triangular_mask.unsqueeze(0).expand(pad_mask.size(0), -1, -1)
    target_mask = target_mask.masked_fill(pad_mask, float('-inf'))
    target_mask = target_mask.repeat_interleave(num_heads, dim=0)
    return target_mask

class TransformerDecoderLayer(nn.Module):
    """Transformer解码器层"""

    def __init__(self, embed_size, num_heads, expansion, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.norm3 = nn.LayerNorm(embed_size)
        self.self_attention = nn.MultiheadAttention(embed_size, num_heads, dropout=dropout)
        self.encoder_attention = nn.MultiheadAttention(embed_size, num_heads, dropout=dropout)
        self.feed_forward = FeedForwardNetwork(embed_size, expansion, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, source_mask, target_mask):
        x = x.transpose(0, 1)
        encoder_output = encoder_output.transpose(0, 1)
        attention_output, _ = self.self_attention(x, x, x, attn_mask=target_mask)
        query = self.dropout(self.norm1(attention_output + x))
        attention_output, _ = self.encoder_attention(query, encoder_output, encoder_output, attn_mask=source_mask)
        query = self.dropout(self.norm2(attention_output + query))
        query = query.transpose(0, 1)
        out = self.feed_forward(query)
        out = self.dropout(self.norm3(out + query))
        return out

class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, embed_size, max_length=5000):
        super().__init__()
        pe = torch.zeros(max_length, embed_size)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class TransformerDecoder(nn.Module):
    """Transformer解码器"""

    def __init__(self, embed_size, num_heads, expansion, dropout, num_layers, target_vocab_size, pretrained_embeddings=None):
        super().__init__()
        self.embed_size = embed_size
        self.word_embedding = nn.Embedding(target_vocab_size, embed_size)
        if pretrained_embeddings is not None:
            assert pretrained_embeddings.shape == (target_vocab_size, embed_size), "预训练嵌入向量尺寸不匹配"
            self.word_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(pretrained_embeddings), freeze=False)
        self.positional_encoding = PositionalEncoding(embed_size)
        self.layers = nn.ModuleList([TransformerDecoderLayer(embed_size, num_heads, expansion, dropout) for _ in range(num_layers)])
        self.fc_out = nn.Linear(embed_size, target_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, source_mask, target_mask):
        x = self.dropout(self.word_embedding(x))
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, encoder_output, source_mask, target_mask)
        out = self.fc_out(x)
        return out

class ImageCaptioningModel(nn.Module):
    """图像字幕模型"""

    def __init__(self, img_size, in_channels, patch_size, embed_size, target_vocab_size, num_layers, num_heads, expansion, dropout, pretrained_embeddings=None):
        super().__init__()
        self.encoder = VisionTransformerEncoder(in_channels, patch_size, img_size, embed_size, num_layers, num_heads, expansion, dropout)
        self.decoder = TransformerDecoder(embed_size, num_heads, expansion, dropout, num_layers, target_vocab_size, pretrained_embeddings)

    def forward(self, images, captions, source_mask=None, target_mask=None):
        encoder_output = self.encoder(images)
        decoder_output = self.decoder(captions, encoder_output, source_mask, target_mask)
        return decoder_output
    
    
# class Vocabulary:

def train_model(model, train_loader, criterion, optimizer, save_path, device, num_epochs=10, save_interval=120, pretrained_weights=None, experiment_name='experiment'):
    os.makedirs(os.path.split(save_path)[0], exist_ok=True)
    if pretrained_weights:
        model.load_state_dict(torch.load(pretrained_weights))

    progress_bar = tqdm(range(num_epochs))
    model = model.to(device)
    save_interval = timedelta(seconds=save_interval)
    model.train()

    for epoch in progress_bar:
        running_loss = 0.0
        last_save_time = datetime.now()
        for batch_idx, (image, sequence, sequence_length) in enumerate(train_loader):
            image = image.to(device)
            sequence = sequence.to(device)
            input_sequence = sequence[:, :-1]
            target_sequence = sequence[:, 1:]

            optimizer.zero_grad()
            prediction, _ = model(image, input_sequence)
            _, _, vocab_size = prediction.shape
            loss = criterion(prediction.view(-1, vocab_size), target_sequence.contiguous().view(-1))
            loss.backward()
            optimizer.step()

            if datetime.now() - last_save_time > save_interval:
                last_save_time = datetime.now()
                torch.save(model.state_dict(), save_path)

            running_loss += loss.item()
            progress_bar.set_postfix(progress=f'{(batch_idx + 1)} / {len(train_loader)}', loss=f'{running_loss / (batch_idx + 1):.4f}', last_save_time=last_save_time)

def beam_search(model, image, vocab, device, max_length, beam_width=5):
    model.to(device)
    image = image.to(device)
    sequences = [([vocab.start], 0, ())]

    for _ in range(max_length):
        all_candidates = []
        for sequence, score, hidden_state in sequences:
            if sequence[-1] == vocab.end:
                all_candidates.append((sequence, score, hidden_state))
                continue

            input_tensor = torch.tensor([sequence[-1]], dtype=torch.long, device=device).unsqueeze(0)
            with torch.no_grad():
                output, new_hidden_state = model(image, input_tensor, *hidden_state)

            top_k_probs, top_k_indices = torch.topk(torch.softmax(output, dim=-1), beam_width, dim=-1)
            top_k_probs = top_k_probs[0, -1]
            top_k_indices = top_k_indices[0, -1]

            for k in range(beam_width):
                next_sequence = sequence + [top_k_indices[k].item()]
                next_score = score - torch.log(top_k_probs[k])
                all_candidates.append((next_sequence, next_score, new_hidden_state))

        ordered = sorted(all_candidates, key=lambda tup: tup[1])
        sequences = ordered[:beam_width]

    best_sequence = sequences[0][0]
    best_score = sequences[0][1].item()
    return vocab.decode(best_sequence), best_score

def evaluate_model(model, test_set, vocab, device, max_length, beam_width=5):
    model.eval()
    metrics = np.zeros(4)
    progress_bar = tqdm(range(len(test_set)), desc='evaluating')
    for i in progress_bar:
        _, caption = test_set.get_pair(i)
        image, _, _ = test_set[i]
        generated_caption, score = beam_search(model, image.unsqueeze(0), vocab, device, max_length, beam_width)
        metrics += np.array([score, bleu(generated_caption, caption, vocab), rouge_l(generated_caption, caption, vocab), meteor(generated_caption, caption, vocab)])
        value = metrics / (i + 1)
        progress_bar.set_postfix(score=value[0], bleu=value[1], rouge_l=value[2], meteor=value[3])

def main():
    save_path = 'xx/model_ViT_T.pth'
    experiment_name = 'xx'
    vocab_path = 'xx/vocab.json'
    word2vec_cache_path = 'xx/word2vec.npy'
    dataset_root = 'xx/deepfashion-multimodal'
    train_labels_path = 'xx/deepfashion-multimodal/train_captions.json'
    test_labels_path = 'xx/deepfashion-multimodal/test_captions.json'

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    vocab = Vocabulary(vocab_path)
    transform = Compose([
        Resize((IMAGE_SIZE, IMAGE_SIZE)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model = ImageCaptioningModel(len(vocab), IN_CHANNELS, IMAGE_EMBED_SIZE, TEXT_EMBED_SIZE, HIDDEN_SIZE, pretrained_embeddings=vocab.get_word2vec(cache_path=word2vec_cache_path)).to(device)
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=torch.device('cpu')))

    train_set = ImageTextDataset(dataset_root, train_labels_path, vocab, max_seq_len=SEQUENCE_LENGTH, transform=transform, max_cache_memory=32 * 1024 ** 3)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_model(model, train_loader, criterion, optimizer, save_path, device, NUM_EPOCHS, experiment_name=experiment_name)

    test_set = ImageTextDataset(dataset_root, test_labels_path, vocab, max_seq_len=SEQUENCE_LENGTH, transform=transform, max_cache_memory=32 * 1024 ** 3)
    evaluate_model(model, test_set, vocab, device, SEQUENCE_LENGTH, beam_width=5)

if __name__ == '__main__':
    main()