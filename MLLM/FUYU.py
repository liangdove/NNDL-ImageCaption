import base64
import urllib
import requests
import json

API_KEY = "jSHDtwarvVTHtFVxQs8lkFF8"
SECRET_KEY = "CwF6YNuCF5zF4ReaIZphBBwi4oVkOY5S"
cibiao=["the" , "upper" , "clothing" , "has" , "long" , "sleeves" , "cotton" , "fabric" , "and" , "solid" , "color" , "patterns"  , "neckline" , "of" , "it" , "is" , "v" , "shape" , "lower" , "length" , "denim" , "this" , "lady" , "also" , "wears" , "an" , "outer" , "with" , "complicated" , "female" , "wearing" , "a" , "ring" , "on" , "her" , "finger" , "neckwear" , "tank" , "shirt" , "no" , "chiffon" , "graphic" , "round" , "person" , "pants" , "are" , "top" , "woman" , "trousers" , "there" , "belt" , "accessory" , "wrist" , "sweater" , "lattice" , "three" , "point" , "pure" , "in" , "his" , "neck" , "sleeve" , "plaid" , "its" , "lapel" , "socks" , "shoes" , "suspenders" , "short" , "t" , "shorts" , "crew" , "sleeveless" , "floral" , "hat" , "pair" , "quarter" , "head" , "waist" , "leather" , "pattern" , "cut" , "off" , "medium" , "knitting" , "gentleman" , "other" , "mixed" , "stripe" , "skirt" , "striped" , "sunglasses" , "guy" , "stand" , "man" , "square" , "leggings" , "furry" , "block" , "glasses" , "hands" , "or" , "clothes" ]
def main():
        
    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/image2text/fuyu_8b?access_token=" + get_access_token()
    #encode_image=get_file_content_as_base64("D:\神经网络深度学习\deepfashion\FashionDescription-main\FashionDescription\data\deepfashion-multimodal\images\WOMEN-Tees_Tanks-id_00007979-04_4_full.jpg",False)
    #print(encode_image)
    # image 可以通过 get_file_content_as_base64("C:\fakepath\MEN-Denim-id_00000080-01_7_additional.jpg",False) 方法获取
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
    prompt = f"Describe what the person is wearing using only words in this list: {', '.join(word_list)}"

    payload = json.dumps({
        "prompt": prompt,
        "image": get_file_content_as_base64("data\deepfashion-multimodal\images\WOMEN-Blouses_Shirts-id_00002333-02_7_additional.jpg",False)
        })
    headers = {
        'Content-Type': 'application/json'
    }
    
    response = requests.request("POST", url, headers=headers, data=payload)
    
    print(response.json()["result"])
    

def get_file_content_as_base64(path, urlencoded=False):
    """
    获取文件base64编码
    :param path: 文件路径
    :param urlencoded: 是否对结果进行urlencoded 
    :return: base64编码信息
    """
    with open(path, "rb") as f:
        content = base64.b64encode(f.read()).decode("utf8")
        if urlencoded:
            content = urllib.parse.quote_plus(content)
    return content

def get_access_token():
    """
    使用 AK，SK 生成鉴权签名（Access Token）
    :return: access_token，或是None(如果错误)
    """
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY}
    return str(requests.post(url, params=params).json().get("access_token"))

if __name__ == '__main__':
    main()
