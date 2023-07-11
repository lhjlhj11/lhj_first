import requests
import base64
from collections import OrderedDict
from PIL import Image
import os

sd_api = "http://127.0.0.1:7860"
url = f"{sd_api}/tagger/v1/interrogate"

image_name = f"long1.jpg"
image_path = os.getcwd()
image_path = os.path.join(image_path, image_name)
# model = 'wd14-convnext-v2'
model = 'wd14-vit-v2-git'
threshold = 0.35

# 确认照片为上传照片
image = Image.open(image_path)
image.show()

#将图片转换为Base64字符串
with open(image_path, 'rb') as file:
    image_data = file.read()
    base64_image = base64.b64encode(image_data).decode('utf-8')

# 构建请求体的JSON数据
data = {
    "image": base64_image,
    "model": model,
    "threshold": threshold
}

# 发送POST请求
response = requests.post(url, json=data)

# 检查响应状态码
if response.status_code == 200:
    json_data = response.json()
    # 处理返回的JSON数据
    caption_dict = json_data['caption']
    print(caption_dict)
    sorted_items = sorted(caption_dict.items(), key=lambda x: x[1], reverse=True)
    print(sorted_items)
    print(sorted_items[0][0])
    #output = '\n'.join([f'{k}: {v}, {int(v * 100)}%' for k, v in sorted_items])
    #print(output)
    # image_caption = ""
    # for captions in sorted_items:
    #     if captions[1] >= 0.34:
    #         image_caption = image_caption + captions[0] + ','
    # with open("output.txt", "w") as file:
    #     file.write(image_caption)
    #     file.close()

else:
    print('Error:', response.status_code)
    print('Response body:', response.text)

