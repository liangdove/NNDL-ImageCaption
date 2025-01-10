import numpy as np
import torch
import torch.nn as nn


class ReinforcementLearningLoss(nn.Module):
    def __init__(self, vocab, reward_func, gamma=0.99):
        super(ReinforcementLearningLoss, self).__init__()
        self.vocab = vocab  # 词汇表
        self.reward_func = reward_func  # 回报计算函数（根据 BLEU、ROUGE-L 等评测指标）
        self.gamma = gamma  # 折扣因子，控制回报的时间延迟

    def forward(self, prediction, target_seq, seq_len, action_log_probs):
        """
        prediction: 模型的预测 [batch_size, seq_len, vocab_size]
        target_seq: 真实目标序列 [batch_size, seq_len]
        action_log_probs: 动作的对数概率 [batch_size, seq_len]，每个 token 的 log-probability
        seq_len: 每个样本的序列长度 [batch_size]
        """
        # Step 1: 使用评测指标计算回报
        rewards = self.reward_func(prediction, target_seq, self.vocab)

        # 选择一个合适的奖励策略，这里使用 BLEU、ROUGE-L 和 METEOR 的平均值
        total_rewards = []
        for i in range(len(rewards["BLEU"])):
            total_reward = rewards["METEOR"][i]
            total_rewards.append(total_reward)
        #print("totalscore:",totalscore)
        total_rewards = torch.tensor(total_rewards, dtype=torch.float32)

        # Step 2: 计算累计回报（可以使用折扣因子 gamma 来累积回报）
        cumulative_rewards = []
        for i in range(len(total_rewards)):
            cumulative_reward = 0
            for t in range(seq_len[i].item()):  # 使用每个样本的 seq_len
                cumulative_reward = total_rewards[i] + self.gamma * cumulative_reward
            cumulative_rewards.append(cumulative_reward)

        cumulative_rewards = torch.tensor(cumulative_rewards, dtype=torch.float32)

        # Step 3: 计算策略梯度损失（基于 REINFORCE 算法）
        loss = 0
        batch_size = seq_len.size(0)
        
        for b in range(batch_size):
            for t in range(seq_len[b].item()):  # 逐个样本，逐时间步处理
                # 对于每个样本的 t 步，使用它的 seq_len
                reward_at_t = cumulative_rewards[b]  # 获取该样本的累计回报
                log_prob_at_t = action_log_probs[b, t]  # 获取该样本的 t 时间步的对数概率
                loss -= log_prob_at_t * reward_at_t  # 计算该时间步的损失

        return loss.mean()
