from rouge import Rouge
from nltk.translate.bleu_score import corpus_bleu
from vocabulary import Vocabulary

__rouge = Rouge()


def bleu(text, standard, vocabulary):
    """计算BLEU指标"""
    candidate = vocabulary.encode(text)
    reference = [vocabulary.encode(standard)]  
    score = corpus_bleu([reference], [candidate])
    return score

__rouge = Rouge()
def rouge_l(text, standard, vocabulary):
    """计算Rouge-l指标"""
    score = __rouge.get_scores(text, standard)[0]['rouge-l']['f']
    return score


def _find_chunks(candidate, reference):
    """寻找chunks（连续匹配单词序列）"""
    candidate_chunks = []
    reference_chunks = []
    chunk = []

    for word in candidate:
        if word in reference:
            if not chunk:
                chunk_start = reference.index(word)
            chunk.append(word)
        else:
            if chunk:
                candidate_chunks.append(chunk)
                reference_chunks.append(reference[chunk_start:chunk_start+len(chunk)])
                chunk = []

    if chunk:  # 处理最后一个chunk
        candidate_chunks.append(chunk)
        reference_chunks.append(reference[chunk_start:chunk_start+len(chunk)])

    return candidate_chunks, reference_chunks


def meteor(candidate, reference, vocabulary):
    candidate_words = vocabulary.split(candidate)
    reference_words = vocabulary.split(reference)

    candidate_chunks, reference_chunks = _find_chunks(candidate_words, reference_words)

    # 计算匹配数和chunk数
    matches = sum(len(chunk) for chunk in candidate_chunks)
    num_chunks = len(candidate_chunks)

    # 计算精确率和召回率
    precision = matches / len(candidate_words)
    recall = matches / len(reference_words)

    f_score = 0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)

    # 计算惩罚因子
    penalty = 0.5 * ((num_chunks / matches) ** 3) if matches != 0 else 0

    # 计算最终的METEOR分数
    meteor = f_score * (1 - penalty)

    return meteor
