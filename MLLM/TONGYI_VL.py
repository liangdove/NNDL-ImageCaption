import os
import base64
from openai import OpenAI

# 基础64编码格式
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# 保存输出结果到文件
def save_output_to_file(output, file_path):
    with open(file_path, "a", encoding="utf-8") as output_file:  # 使用追加模式
        output_file.write(output + "\n" + "="*80 + "\n\n")  # 添加分隔符

# 初始化OpenAI客户端
def initialize_client():
    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
        api_key="sk-117e657b5eb54689b2fd0297a8f3fa24",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    return client


# 处理单张图像并调用API
def process_image(client, image_path):
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
    base64_image = encode_image(image_path)
    completion = client.chat.completions.create(
        model="qwen-vl-max-latest",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        # 根据图像格式选择合适的Content Type
                        # PNG图像：  f"data:image/png;base64,{base64_image}"
                        # JPEG图像： f"data:image/jpeg;base64,{base64_image}"
                        # WEBP图像： f"data:image/webp;base64,{base64_image}"
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                    {"type": "text", "text": f"Describe what the person is wearing using only words in this list: {', '.join(word_list)}"},
                ],
            }
        ],
    )
    return completion.choices[0].message.content

# 处理整个数据集
def process_dataset(dataset_path, output_file_path):
    client = initialize_client()
    for filename in os.listdir(dataset_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            image_path = os.path.join(dataset_path, filename)
            try:
                output_text = process_image(client, image_path)
                print(f"Processed {filename}: {output_text}")
                save_output_to_file(f"Image: {filename}\nDescription: {output_text}", output_file_path)
            except Exception as e:
                print(f"Failed to process {filename}: {e}")

# 数据集路径和输出文件路径
# dataset_path = "D:/Desktop/work/NNDL/design/tongyi/images"
# output_file_path = "D:/Desktop/work/NNDL/design/tongyi/output.txt"

# 清空或创建输出文件
# open(output_file_path, "w").close()
#print("后")
# 处理数据集
# process_dataset(dataset_path, output_file_path)
# client = initialize_client()
# path="data/deepfashion-multimodal\images\MEN-Tees_Tanks-id_00001467-01_4_full.jpg"
# print("通义LLM生成的描述：",process_image(client,path))