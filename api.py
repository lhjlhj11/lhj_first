import os
import time
from flask import Flask, request
from model import txt2img, img2img, decode_image, select_checkpoint, encode_image, generate_image_file_path, save_image
from baiduapi import get_standard_image, removebg
from changeface import change_face
app = Flask(__name__)

img2imgpayload_orgin = [
{
    #'init_images': [encoded_image],
    'prompt': "(masterpiece), best quality, boy, (((handsome))), gorgeous, portrait, jae lee style, yoshitaka amano style, ink painting, black and white, art, abstract, expressionism, wu guanzhong style , manga, line art",
    'negative_prompt': "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, mustache, EasyNegative , female, girl",
    'steps': 28,
    'sampler_name': "Euler a",
    'denoising_strength': 0.88,
    "alwayson_scripts": {
            "controlnet": {
                "args": [
                    {
                        # "input_image": controlnet_image,
                        "module": "depth_midas",
                        "model": "control_v11f1p_sd15_depth",
                        "weight": 0.66
                    },
                    {
                        # "input_image": controlnet_image,
                        "module": "lineart_standard",
                        "model": "control_v11p_sd15_lineart",
                        "weight": 0.66
                    }
                ]
            }
        }
},
{
    #'init_images': [encoded_image],
    'prompt': "(masterpiece), best quality, boy, (((handsome))), gorgeous, portrait , sky background",
    'negative_prompt': "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, mustache, EasyNegative , female, girl",
    'steps': 28,
    'sampler_name': "Euler a",
    'denoising_strength': 0.88,
    "alwayson_scripts": {
            "controlnet": {
                "args": [
                    {
                        # "input_image": controlnet_image,
                        "module": "depth_midas",
                        "model": "control_v11f1p_sd15_depth",
                        "weight": 0.66
                    },
                    {
                        # "input_image": controlnet_image,
                        "module": "lineart_standard",
                        "model": "control_v11p_sd15_lineart",
                        "weight": 0.66
                    }
                ]
            }
        }
},
{
    #'init_images': [encoded_image],
    'prompt': "(masterpiece), best quality, boy, (((handsome))), gorgeous, portrait, suit, stage light , godfather , series man, premium lounge , noir",
    'negative_prompt': "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, mustache, EasyNegative , female, girl",
    'steps': 28,
    'sampler_name': "Euler a",
    'denoising_strength': 0.88,
    "alwayson_scripts": {
            "controlnet": {
                "args": [
                    {
                        # "input_image": controlnet_image,
                        "module": "depth_midas",
                        "model": "control_v11f1p_sd15_depth",
                        "weight": 0.66
                    },
                    {
                        # "input_image": controlnet_image,
                        "module": "lineart_standard",
                        "model": "control_v11p_sd15_lineart",
                        "weight": 0.66
                    }
                ]
            }
        }
},
{
    #'init_images': [encoded_image],
    'prompt': "(masterpiece), best quality, boy, (((handsome))), gorgeous, portrait, warrior, helmet, armored , battlefield background",
    'negative_prompt': "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, mustache, EasyNegative , female, girl",
    'steps': 28,
    'sampler_name': "Euler a",
    'denoising_strength': 0.88,
    "alwayson_scripts": {
            "controlnet": {
                "args": [
                    {
                        # "input_image": controlnet_image,
                        "module": "depth_midas",
                        "model": "control_v11f1p_sd15_depth",
                        "weight": 0.66
                    },
                    {
                        # "input_image": controlnet_image,
                        "module": "lineart_standard",
                        "model": "control_v11p_sd15_lineart",
                        "weight": 0.66
                    }
                ]
            }
        }
},
{
    #'init_images': [encoded_image],
    'prompt': "(masterpiece), best quality, boy, (((handsome))), gorgeous, portrait, shikai makoto style, universe background",
    'negative_prompt': "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, mustache, EasyNegative , female, girl",
    'steps': 28,
    'sampler_name': "Euler a",
    'denoising_strength': 0.88,
    "alwayson_scripts": {
            "controlnet": {
                "args": [
                    {
                        # "input_image": controlnet_image,
                        "module": "depth_midas",
                        "model": "control_v11f1p_sd15_depth",
                        "weight": 0.66
                    },
                    {
                        # "input_image": controlnet_image,
                        "module": "lineart_standard",
                        "model": "control_v11p_sd15_lineart",
                        "weight": 0.66
                    }
                ]
            }
        }
},
{
    #'init_images': [encoded_image],
    'prompt': "(masterpiece), best quality, boy, (((handsome))), gorgeous, portrait, van gogh style, impressionism , starry night",
    'negative_prompt': "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, mustache, EasyNegative , female, girl",
    'steps': 28,
    'sampler_name': "Euler a",
    'denoising_strength': 0.88,
    "alwayson_scripts": {
            "controlnet": {
                "args": [
                    {
                        # "input_image": controlnet_image,
                        "module": "depth_midas",
                        "model": "control_v11f1p_sd15_depth",
                        "weight": 0.66
                    },
                    {
                        # "input_image": controlnet_image,
                        "module": "lineart_standard",
                        "model": "control_v11p_sd15_lineart",
                        "weight": 0.66
                    }
                ]
            }
        }
},
{
    #'init_images': [encoded_image],
    'prompt': "(masterpiece), best quality, boy, (((handsome))), gorgeous, portrait, cyberpunk,",
    'negative_prompt': "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, mustache, EasyNegative , female, girl",
    'steps': 28,
    'sampler_name': "Euler a",
    'denoising_strength': 0.88,
    "alwayson_scripts": {
        "controlnet": {
            "args": [
                {
                    # "input_image": controlnet_image,
                    "module": "depth_midas",
                    "model": "control_v11f1p_sd15_depth",
                    "weight": 0.66
                },
                {
                    # "input_image": controlnet_image,
                    "module": "lineart_standard",
                    "model": "control_v11p_sd15_lineart",
                    "weight": 0.66
                }
            ]
        }
    }
},
{
    #'init_images': [encoded_image],
    'prompt': "(masterpiece), best quality, boy, (((handsome))), gorgeous, portrait, cyberpunk, exoskeleton, cyan, skyscraper background , white skin, hud glasses",
    'negative_prompt': "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, mustache, EasyNegative , female, girl",
    'steps': 28,
    'sampler_name': "Euler a",
    'denoising_strength': 0.88,
    "alwayson_scripts": {
        "controlnet": {
            "args": [
                {
                    # "input_image": controlnet_image,
                    "module": "depth_midas",
                    "model": "control_v11f1p_sd15_depth",
                    "weight": 0.66
                },
                {
                    # "input_image": controlnet_image,
                    "module": "lineart_standard",
                    "model": "control_v11p_sd15_lineart",
                    "weight": 0.66
                }
            ]
        }
    }
},
{
    #'init_images': [encoded_image],
    'prompt': "(masterpiece), best quality, boy, (((handsome))), gorgeous, portrait, fashion, harajuku, street, tattoo, sun glasses, street art",
    'negative_prompt': "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, mustache, EasyNegative , female, girl",
    'steps': 28,
    'sampler_name': "Euler a",
    'denoising_strength': 0.88,
    "alwayson_scripts": {
            "controlnet": {
                "args": [
                    {
                        # "input_image": controlnet_image,
                        "module": "depth_midas",
                        "model": "control_v11f1p_sd15_depth",
                        "weight": 0.66
                    },
                    {
                        # "input_image": controlnet_image,
                        "module": "lineart_standard",
                        "model": "control_v11p_sd15_lineart",
                        "weight": 0.66
                    }
                ]
            }
        }
},
{
    #'init_images': [encoded_image],
    'prompt': "(masterpiece), best quality, boy, (((handsome))), gorgeous, portrait, vintage anime, kimono , 1960s , wabi sabi garden background",
    'negative_prompt': "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, mustache, EasyNegative , female, girl",
    'steps': 28,
    'sampler_name': "Euler a",
    'denoising_strength': 0.88,
    "alwayson_scripts": {
        "controlnet": {
            "args": [
                {
                    # "input_image": controlnet_image,
                    "module": "depth_midas",
                    "model": "control_v11f1p_sd15_depth",
                    "weight": 0.66
                },
                {
                    # "input_image": controlnet_image,
                    "module": "lineart_standard",
                    "model": "control_v11p_sd15_lineart",
                    "weight": 0.66
                }
            ]
        }
    }
},
]

my_secret_id = "AKIDCgBveK0dDiHWCzyNmKyZpzKo7eJ3gOB6"
my_secret_key = "vnmpkyzrgYhxFRbiihlx2ygFGRkFtbJO"
my_region = "ap-shanghai"
my_bucket_name = "qiyu-1318929734"

@app.route("/portrait", methods=["POST"])
def painting():
    command = request.form.get("command")
    if command == "txt2img":
        response = txt2img()
        timestamp = int(time.time())
        image_file = decode_image(response, timestamp)
        return image_file
    elif command == "img2img":
        print("----begin----")
        #get image
        image = request.files.get("images")
        image.save('./input/input_image.png')
        input_file_name = f"input/input_image.png"
        input_file = os.path.join(os.getcwd(), input_file_name)
        #change to white background
        get_standard_image(input_file)
        removebg(input_file)
        #encode image
        input_image = encode_image(input_file)
        output_file = []
        for payload in img2imgpayload_orgin:
            images = []
            images.append(input_image)
            img2imgpayload = payload
            img2imgpayload.setdefault("init_images", images)
            response = img2img(img2imgpayload)
            #local image
            timestamp = int(time.time())
            image_local_file = decode_image(response, timestamp)
            #change face
            # source_image = input_file.split(".")[0]
            # result_image = image_local_file.split(".")[0]
            # change_face(source_image, result_image, timestamp)
            #qcloud image
            image_cloud_file = save_image(secret_id=my_secret_id, secret_key=my_secret_key, region=my_region,
                                          bucket_name=my_bucket_name, local_path=image_local_file, time_stamp=timestamp)
            output_file.append(image_cloud_file)
        # img2imgpayload = img2imgpayload1
        # img2imgpayload.setdefault("init_images", input_image)
        # response = img2img(input_image)
        # image_file = decode_image(response)
        output = {
            "code": 200,
            "data": output_file,
            "msg": ''
        }
        print("----finish----")
        return output

if __name__ == "__main__":
    app.run(debug=False, host='172.16.10.116')
