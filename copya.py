from datetime import datetime, timedelta
import os
from PIL import Image
import torch.nn.functional as F
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
from tqdm import _get_clones,_get_activation_fn,Optional,Tensor
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt

from datasets import ImageTextDataset
from vocabulary import Vocabulary
from metrics import bleu, meteor, rouge_l


class CaptionGenerator(nn.Module):
    def __init__(self, feature_extractor, transformer, hidden_dim, vocab_size):
        """
        初始化模型：
        - feature_extractor: 图像特征提取模块（例如 CNN）
        - transformer: 用于处理序列数据的 Transformer
        - hidden_dim: Transformer 的特征维度
        - vocab_size: 词汇表大小
        """
        super().__init__()
        self.feature_extractor = feature_extractor
        # 使用卷积将特征通道调整为 Transformer 所需的隐藏维度
        self.feature_projection = nn.Conv2d(feature_extractor.num_channels, hidden_dim, kernel_size=1)
        self.transformer = transformer
        # 输出通过多层感知机映射到词汇表分布
        self.output_head = MultiLayerPerceptron(hidden_dim, 512, vocab_size, num_layers=3)

    def forward(self, images, captions, masks):
        """
        前向传播：
        - images: 输入图像数据
        - captions: 文本序列，用作目标描述
        - masks: 掩码，用于指示哪些部分有效
        """
        # 从特征提取器获取编码后的特征和位置信息
        features, positions = self.feature_extractor(images)
        src, src_mask = features[-1].decompose()
        assert src_mask is not None, "Mask is required for Transformer."

        # 使用 Transformer 处理特征序列
        encoded_features = self.transformer(
            self.feature_projection(src), src_mask, positions[-1], captions, masks
        )
        # 输出结果通过 MLP 映射为词汇分布
        output = self.output_head(encoded_features.permute(1, 0, 2))
        return output


# 多层感知机（MLP）
class MultiLayerPerceptron(nn.Module):
    """
    使用多层全连接网络，将输入特征映射到目标维度。
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        """
        初始化多层感知机：
        - input_dim: 输入的特征维度
        - hidden_dim: 隐藏层的维度
        - output_dim: 输出特征的维度
        - num_layers: 网络层数
        """
    def forward(self, x):
        """
        前向传播：
        - 在最后一层之前，激活函数使用 ReLU。
        """
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
# 构建完整的模型
def initialize_model(config):
    """
    根据配置创建模型和损失函数：
    - config: 包含模型参数的配置对象
    """
    # 实例化特征提取器和 Transformer
    feature_extractor = build_backbone(config)
    transformer = build_transformer(config)

    # 创建描述生成模型
    model = CaptionGenerator(feature_extractor, transformer, config.hidden_dim, config.vocab_size)
    # 定义交叉熵作为损失函数
    loss_fn = nn.CrossEntropyLoss()
    return model, loss_fn

# 初始化目标序列模板和掩码
def initialize_caption_and_mask(start_token, max_length):
    """
    初始化序列模板和掩码：
    - start_token: 序列的起始标记
    - max_length: 描述的最大长度
    """
    # 创建序列模板和掩码
    caption_seq = torch.zeros((1, max_length), dtype=torch.long)
    caption_mask = torch.ones((1, max_length), dtype=torch.bool)
    # 设置起始标记，并解锁第一个位置的掩码
    caption_seq[:, 0] = start_token
    caption_mask[:, 0] = False
    return caption_seq, caption_mask

# 评估函数：生成描述序列
def evaluate_caption(model, image, config, start_token, end_token):

    """
    模型评估逻辑，使用自回归方式生成描述：
    - model: 已训练的模型
    - image: 输入图像
    - config: 模型配置
    - start_token: 起始标记
    - end_token: 结束标记
    """
    model.eval()  # 进入评估模式
    caption_seq, caption_mask = initialize_caption_and_mask(start_token, config.max_position_embeddings)

    # 自回归生成序列
    for i in range(1, config.max_position_embeddings):
        # 获取当前时间步的预测分布
        predictions = model(image, caption_seq, caption_mask)
        # 提取当前时间步的预测结果
        step_predictions = predictions[:, i - 1, :]
        predicted_token = torch.argmax(step_predictions, dim=-1)

        # 如果检测到结束标记，提前终止
        if predicted_token[0] == end_token:
            break

        # 更新序列和掩码
        caption_seq[:, i] = predicted_token[0]
        caption_mask[:, i] = False

    return caption_seq

# 自定义 Transformer 模块，包含编码器和解码器
class CustomTransformer(nn.Module):
    def __init__(self, config, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        """
        初始化自定义 Transformer 模块:
        - config: 配置对象，包含模型相关的参数
        - d_model: 特征维度
        - nhead: 多头注意力机制的头数
        - num_encoder_layers: 编码器的层数
        - num_decoder_layers: 解码器的层数
        - dim_feedforward: 前馈网络的隐藏层维度
        - dropout: Dropout 比例
        - activation: 激活函数类型
        - normalize_before: 是否在归一化之前处理数据
        - return_intermediate_dec: 解码器是否返回中间结果
        """
        super().__init__()

        # 定义编码器层及整体编码器
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        # 定义解码器嵌入层
        self.embeddings = DecoderEmbeddings(config)

        # 定义解码器层及整体解码器
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,return_intermediate=return_intermediate_dec)

        # 初始化参数
        self._initialize_parameters()

        # 保存特征维度与多头注意力的头数
        self.d_model = d_model
        self.nhead = nhead

    def _initialize_parameters(self):
        """
        参数初始化：采用 Xavier 均匀分布初始化所有权重参数。
        """
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, src, mask, position_embeddings, tgt, tgt_mask):
        """
        前向传播逻辑:
        - src: 输入特征 (N, C, H, W)，图像或其他输入数据的编码
        - mask: 输入特征的掩码 (N, H*W)，标记哪些位置是有效的
        - position_embeddings: 位置编码，提供序列中每个位置的位置信息
        - tgt: 解码器目标输入 (N, L)，表示目标序列
        - tgt_mask: 解码器的目标序列掩码 (N, L)，标记目标序列的有效位置

        返回:
        - hs: 解码器输出结果，包含上下文特征。
        """


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder: 多层堆叠的编码器，用于处理输入特征并提取全局上下文表示。
    """
    def __init__(self, encoder_layer, num_layers, norm=None):
        """
        初始化 TransformerEncoder。

        参数:
        - encoder_layer: 单个编码器层的实例，定义了特征处理的基本逻辑。
        - num_layers: 编码器堆叠的层数。
        - norm: 可选的归一化层，对最终输出应用归一化（如 LayerNorm）。
        """
        super().__init__()
        # 克隆多个相同的编码器层，组成编码器的层堆叠
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm  # 保存归一化层

    def forward():
        """
        前向传播逻辑。

        参数:
        - src: 输入特征张量，形状为 [sequence_length, batch_size, feature_dim]。
        - mask: 可选的自注意力掩码，用于屏蔽特定位置的计算。
        - src_key_padding_mask: 可选的填充掩码，标记序列中需要忽略的填充值。
        - pos: 可选的位置嵌入，用于引入序列中的位置信息。

        返回:
        - output: 编码器生成的上下文特征，形状与输入相同。
        """

class TransformerDecoder(nn.Module):
    """
    Transformer Decoder: 多层堆叠的解码器模块，用于结合编码器输出（memory）和目标序列生成解码结果。
    """
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        """
        初始化 TransformerDecoder。

        参数:
        - decoder_layer: 单个解码器层实例，定义解码过程的基本逻辑。
        - num_layers: 解码器堆叠的层数。
        - norm: 可选的归一化层，对最终输出进行归一化。
        - return_intermediate: 是否返回所有中间层的输出，默认为 False。
        """
        super().__init__()
        # 克隆多个相同的解码器层，构建多层堆叠结构
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm  # 保存归一化层
        self.return_intermediate = return_intermediate  # 是否返回中间结果

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        """
        前向传播逻辑。

        参数:
        - tgt: 目标序列张量，形状为 [sequence_length, batch_size, feature_dim]。
        - memory: 编码器的输出（上下文特征），形状为 [sequence_length, batch_size, feature_dim]。
        - tgt_mask: 自注意力掩码，用于屏蔽目标序列的特定位置。
        - memory_mask: 编码器-解码器注意力掩码，用于控制 memory 中的计算范围。
        - tgt_key_padding_mask: 目标序列的填充掩码，标记需要忽略的填充位置。
        - memory_key_padding_mask: 编码器输出的填充掩码，屏蔽无效位置。
        - pos: 编码器特征的位置信息。
        - query_pos: 目标序列的查询位置信息。

        返回:
        - output: 解码器的最终输出，或者包含所有中间结果的张量。
        """

class TransformerEncoderLayer(nn.Module):
    """
    Transformer 编码器层：包含多头自注意力和前馈神经网络的基础结构单元。
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):

        super().__init__()
        # 多头自注意力机制
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # 前馈神经网络的两层线性变换
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # 两个 LayerNorm 用于归一化输入和输出
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # 两个 dropout 层用于防止过拟合
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # 激活函数（如 ReLU 或 GELU）
        self.activation = _get_activation_fn(activation)

        # 是否在前向传播时优先进行归一化
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        """
        将位置嵌入添加到输入特征上。
        参数:
        - tensor: 输入的特征张量。
        - pos: 位置嵌入张量，若为 None 则不进行位置信息添加。
        返回:
        - 结合了位置嵌入的特征张量。
        """
    def forward_post(self, src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        """
        后归一化版本的前向传播：先进行自注意力计算，再归一化。
        参数:
        - src: 输入特征张量。
        - src_mask: 自注意力掩码。
        - src_key_padding_mask: 填充掩码，屏蔽无效位置。
        - pos: 位置嵌入张量。
        返回:
        - 经过自注意力和前馈网络处理后的输出。
        """
    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        """
        前归一化版本的前向传播：先归一化，再进行自注意力计算。
        参数:
        - src: 输入特征张量。
        - src_mask: 自注意力掩码。
        - src_key_padding_mask: 填充掩码。
        - pos: 位置嵌入张量。

        返回:
        - 经过自注意力和前馈网络处理后的输出。
        """
    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        """
        前向传播逻辑，根据是否在注意力前进行归一化选择对应的方法。

        参数:
        - src: 输入特征张量。
        - src_mask: 自注意力掩码。
        - src_key_padding_mask: 填充掩码。
        - pos: 位置嵌入张量。

        返回:
        - 经过处理的输出特征。
        """
        if self.normalize_before:
            # 先归一化，再执行注意力和前馈网络
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        # 先执行注意力和前馈网络，再归一化
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

class TransformerDecoderLayer(nn.Module):
    """
    Transformer 解码器层：结合自注意力、交叉注意力和前馈网络的结构单元。
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        """
        初始化解码器层。
        参数:
        - d_model: 特征向量的维度大小。
        - nhead: 多头注意力中头的数量。
        - dim_feedforward: 前馈网络中隐藏层的维度。
        - dropout: dropout 比率，用于减少过拟合。
        - activation: 激活函数（默认 "relu"），用于前馈网络的非线性转换。
        - normalize_before: 是否在计算注意力和前馈网络之前应用归一化。
        """
        super().__init__()
        # 解码器自注意力层
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # 解码器-编码器交叉注意力层
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # 前馈神经网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = _get_activation_fn(activation)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        # 归一化层
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        # Dropout 层
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        # 是否先归一化后计算
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        """
        添加位置嵌入到输入张量中。
        参数:
        - tensor: 输入特征张量。
        - pos: 位置嵌入张量，若为 None 则直接返回输入张量。
        返回:
        - 包含位置信息的特征张量。
        """
        return tensor if pos is None else tensor + pos
    
    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        """
        后归一化模式的前向传播逻辑。

        参数:
        - tgt: 解码器输入张量。
        - memory: 编码器输出张量。
        - tgt_mask: 解码器自注意力掩码。
        - memory_mask: 编码器-解码器注意力掩码。
        - tgt_key_padding_mask: 解码器输入的填充掩码。
        - memory_key_padding_mask: 编码器输出的填充掩码。
        - pos: 编码器位置嵌入。
        - query_pos: 解码器查询位置嵌入。

        返回:
        - 解码器的输出张量。
        """
        # 自注意力机制
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # 交叉注意力机制，与编码器输出交互
        tgt2 = self.cross_attn(query=self.with_pos_embed(tgt, query_pos),
                               key=self.with_pos_embed(memory, pos),
                               value=memory, attn_mask=memory_mask,
                               key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # 前馈网络
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        """
        前归一化模式的前向传播逻辑。

        参数与 `forward_post` 相同。
        """
        # 先对输入归一化
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)

        # 自注意力
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)

        # 交叉注意力
        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn(query=self.with_pos_embed(tgt2, query_pos),
                               key=self.with_pos_embed(memory, pos),
                               value=memory, attn_mask=memory_mask,
                               key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)

        # 前馈网络
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)

        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        """
        解码器的前向传播，自动选择前归一化或后归一化模式。

        参数与 `forward_post` 相同。

        返回:
        - 解码器的输出张量。
        """

class DecoderEmbeddings(nn.Module):
    """
    解码器嵌入层：将输入 token 序列转换为嵌入向量，并结合位置编码。
    """

    def __init__(self, config):
        super().__init__()
        # 单词嵌入层：将 token ID 映射到嵌入向量
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_dim, padding_idx=config.pad_token_id
        )
        # 位置嵌入层：为序列中的每个位置生成嵌入向量
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_dim
        )
        # 归一化层：对嵌入向量进行归一化处理
        self.layer_norm = nn.LayerNorm(
            config.hidden_dim, eps=config.layer_norm_eps
        )
        # Dropout 层：用于防止过拟合
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x):
        """
        前向传播，生成带有位置信息的嵌入向量。

        参数:
        - x: 输入的 token 序列张量，形状为 (batch_size, seq_length)。

        返回:
        - 结合了单词嵌入和位置嵌入的张量，形状为 (batch_size, seq_length, hidden_dim)。
        """

