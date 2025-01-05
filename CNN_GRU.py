from datetime import datetime, timedelta
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from tqdm import tqdm
from metrics import bleu, meteor, rouge_l
from datasets import ImageTextDataset # 从datasets.py中导入ImageTextDataset
from vocabulary import Vocabulary


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

class ConvolutionalNetwork(nn.Module):
    """卷积网络，用于提取图像特征"""

    def __init__(self, in_channels, out_dim):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.fc = nn.Linear(256 * 8 * 8, out_dim)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class ImageTextModel(nn.Module):
    """图像文本模型，结合卷积网络和GRU"""

    def __init__(self, vocab_size, in_channels, image_embed_dim, text_embed_dim, hidden_size=256, pretrained_embeddings=None):
        super().__init__()
        self.image_embed = ConvolutionalNetwork(in_channels, image_embed_dim)
        if pretrained_embeddings is not None:
            assert pretrained_embeddings.shape == (vocab_size, text_embed_dim), "预训练嵌入向量尺寸不匹配"
            self.text_embed = nn.Embedding.from_pretrained(torch.FloatTensor(pretrained_embeddings), freeze=False)
        else:
            self.text_embed = nn.Embedding(vocab_size, text_embed_dim)
        self.gru = nn.GRU(image_embed_dim + text_embed_dim, hidden_size, num_layers=2, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, images, sequences, hidden_state=None, image_features=None):
        if image_features is None:
            image_features = self.image_embed(images)
        image_embed = image_features.unsqueeze(1).repeat(1, sequences.size(1), 1)
        text_embed = self.text_embed(sequences)
        embeddings = torch.cat((image_embed, text_embed), dim=2)
        gru_output, hidden_state = self.gru(embeddings, hidden_state)
        output = self.fc_out(gru_output)
        return output, (hidden_state, image_features)

def train_model(model, train_loader, criterion, optimizer, save_path, device, num_epochs=10, save_interval=120, pretrained_weights=None, experiment_name='experiment'):
    os.makedirs(os.path.split(save_path)[0], exist_ok=True)
    if pretrained_weights:
        model.load_state_dict(torch.load(pretrained_weights))

    progress_bar = tqdm(range(num_epochs))
    model = model.to(device)
    save_interval = timedelta(seconds=save_interval)
    model.train()

    for _ in progress_bar:
        running_loss = 0.0
        last_save_time = datetime.now()
        for batch_idx, (images, sequences, _) in enumerate(train_loader):
            images = images.to(device)
            sequences = sequences.to(device)
            input_sequences = sequences[:, :-1]
            target_sequences = sequences[:, 1:]

            optimizer.zero_grad()
            predictions, _ = model(images, input_sequences)
            _, _, vocab_size = predictions.shape
            loss = criterion(predictions.view(-1, vocab_size), target_sequences.contiguous().view(-1))
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
    save_path = 'xx/model_lstm.pth'
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
        Normalize(mean=[0.4, 0.4, 0.4], std=[0.2, 0.2, 0.2]),
    ])

    model = ImageTextModel(len(vocab), IN_CHANNELS, IMAGE_EMBED_SIZE, TEXT_EMBED_SIZE, HIDDEN_SIZE, pretrained_embeddings=vocab.get_word2vec(cache_path=word2vec_cache_path)).to(device)
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=torch.device('cpu')))

    train_set = ImageTextDataset(dataset_root, train_labels_path, vocab, max_seq_len=SEQUENCE_LENGTH, transform=transform, max_cache_memory=32 * 1024 ** 3)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_model(model, train_loader, criterion, optimizer, save_path, device, NUM_EPOCHS, experiment_name=experiment_name)

    test_set = ImageTextDataset(dataset_root, test_labels_path, vocab, max_seq_len=SEQUENCE_LENGTH, transform=transform, max_cache_memory=32 * 1024 ** 3)
    evaluate_model(model, test_set, vocab, device, SEQUENCE_LENGTH, beam_width=5)

if __name__ == "__main__":
    main()