import argparse
import os
import random
from collections import defaultdict

import cv2
import re

import numpy as np
from PIL import Image
import torch
import html
import gradio as gr

import torchvision.transforms as T
import torch.backends.cudnn as cudnn

from minigpt4.common.config import Config

from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Conversation, SeparatorStyle, Chat

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *
from ExtractVideos import *
from tqdm import tqdm
import setproctitle
setproctitle.setproctitle("wangxin")
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
bounding_box_size = 100

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", default='eval_configs/minigptv2_eval.yaml',
                        help="path to configuration file.")
    parser.add_argument("--save-path", default='/home/phdcv/wangxin/ActionPrompt/MiniGPT-4/testImages/testImages1',
                        help="path to configuration file.")
    parser.add_argument("--videos-path", default='/home/phdcv/wangxin/NTU-RGB-D/RGB/nturgb+d_rgb',
                        help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args

def chat_initialization(cfg):

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    cudnn.benchmark = False
    cudnn.deterministic = True

    print('Initializing Chat')

    device = 'cuda:{}'.format(args.gpu_id)

    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(device)

    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

    model = model.eval()

    CONV_VISION = Conversation(
        system="",
        roles=(r"<s>[INST] ", r" [/INST]"),
        messages=[],
        offset=2,
        sep_style=SeparatorStyle.SINGLE,
        sep="",
    )
    chat = Chat(model, vis_processor, device=device)
    print('Initialization Finished')
    return CONV_VISION, chat

def extract_substrings(string):
    # first check if there is no-finished bracket
    index = string.rfind('}')
    if index != -1:
        string = string[:index + 1]

    pattern = r'<p>(.*?)\}(?!<)'
    matches = re.findall(pattern, string)
    substrings = [match for match in matches]

    return substrings


def save_tmp_img(visual_img):
    file_name = "".join([str(random.randint(0, 9)) for _ in range(5)]) + ".jpg"
    file_path = "/tmp/gradio" + file_name
    visual_img.save(file_path)
    return file_path

def mask2bbox(mask):
    if mask is None:
        return ''
    mask = mask.resize([100, 100], resample=Image.NEAREST)
    mask = np.array(mask)[:, :, 0]

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if rows.sum():
        # Get the top, bottom, left, and right boundaries
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        bbox = '{{<{}><{}><{}><{}>}}'.format(cmin, rmin, cmax, rmax)
    else:
        bbox = ''

    return bbox

def escape_markdown(text):
    # List of Markdown special characters that need to be escaped
    md_chars = ['<', '>']

    # Escape each special character
    for char in md_chars:
        text = text.replace(char, '\\' + char)

    return text

def reverse_escape(text):
    md_chars = ['\\<', '\\>']

    for char in md_chars:
        text = text.replace(char, char[1:])

    return text

def image_upload_trigger(upload_flag, replace_flag, img_list):
    # set the upload flag to true when receive a new image.
    # if there is an old image (and old conversation), set the replace flag to true to reset the conv later.
    upload_flag = 1
    if img_list:
        replace_flag = 1
    return upload_flag, replace_flag

def gradio_ask(user_message, chatbot, chat_state, gr_img, img_list, upload_flag, replace_flag):

    if len(user_message) == 0:
        text_box_show = 'Input should not be empty!'
    else:
        text_box_show = ''

    if isinstance(gr_img, dict):
        gr_img, mask = gr_img['image'], gr_img['mask']
    else:
        mask = None

    if '[identify]' in user_message:
        # check if user provide bbox in the text input
        integers = re.findall(r'-?\d+', user_message)
        if len(integers) != 4:  # no bbox in text
            bbox = mask2bbox(mask)
            user_message = user_message + bbox

    if chat_state is None:
        chat_state = CONV_VISION.copy()

    if upload_flag:
        if replace_flag:
            chat_state = CONV_VISION.copy()  # new image, reset everything
            replace_flag = 0
            chatbot = []
        img_list = []
        llm_message = chat.upload_img(gr_img, chat_state, img_list)
        upload_flag = 0

    chat.ask(user_message, chat_state)

    chatbot = chatbot + [[user_message, None]]

    if '[identify]' in user_message:
        visual_img, _ = visualize_all_bbox_together(gr_img, user_message)
        if visual_img is not None:
            file_path = save_tmp_img(visual_img)
            chatbot = chatbot + [[(file_path,), None]]

    return text_box_show, chatbot, chat_state, img_list, upload_flag, replace_flag

def default_stream_answer(chatbot, chat_state, img_list, temperature):
    if len(img_list) > 0:
        if not isinstance(img_list[0], torch.Tensor):
            chat.encode_img(img_list)
    
    streamer = chat.stream_answer(conv=chat_state,
                                  img_list=img_list,
                                  temperature=temperature,
                                  max_new_tokens=500,
                                  max_length=2000)
                        
    output = ''
    for new_output in streamer:
        escapped = escape_markdown(new_output)
        output += escapped
        chatbot[-1][1] = output
        # yield chatbot, chat_state
    chat_state.messages[-1][1] = '</s>'
    
    return chatbot, chat_state


def default_crop_person(chatbot, gr_img):
    if isinstance(gr_img, dict):
        gr_img, mask = gr_img['image'], gr_img['mask']

    image_width, image_height = gr_img.size
    gr_img = gr_img.resize([960, int(960 / image_width * image_height)])
    image_width, image_height = gr_img.size

    string_list = extract_substrings(html.unescape(reverse_escape(chatbot[-1][1])))
    # string_list
    # ['a person</p> {<51><28><60><75>']
    # ['two women</p> {<49><27><61><79>}<delim>{<39><25><48><80>']
    if string_list:  # it is grounding or detection
        mode = 'all'
        entities = defaultdict(list)
        i = 0
        j = 0
        for string in string_list:
            try:
                obj, string = string.split('</p>')
            except ValueError:
                print('wrong string: ', string)
                continue
            bbox_list = string.split('<delim>')
            flag = False
            for bbox_string in bbox_list:
                integers = re.findall(r'-?\d+', bbox_string)
                # print(integers)
                if len(integers) == 4:
                    x0, y0, x1, y1 = int(integers[0]), int(integers[1]), int(integers[2]), int(integers[3])
                    left = x0 / bounding_box_size * image_width
                    bottom = y0 / bounding_box_size * image_height
                    right = x1 / bounding_box_size * image_width
                    top = y1 / bounding_box_size * image_height

                    # print(left, bottom, right, top)

                    entities[obj].append([left, bottom, right, top])

                    j += 1
                    flag = True
            if flag:
                i += 1
    else:
        raise ValueError(f"Wrong task attribute, should be [grounding] or [detection] !")


    new_image = gr_img.copy()

    for entity_idx, entity_name in enumerate(entities):

        if entity_idx == 0:

            bboxes = np.round(entities[entity_name]).astype(int).tolist()
            
            if len(bboxes) == 1:
                left = bboxes[0][0]
                top = 0#bboxes[0][1]
                right = bboxes[0][2]
                bottom = bboxes[0][3]
                bugflag = False
                bugNum = None
            elif len(bboxes) == 2:
                left = min(bboxes[0][0], bboxes[1][0])
                top = 0#min(bboxes[0][1], bboxes[1][1])
                right = max(bboxes[0][2], bboxes[1][2])
                bottom = max(bboxes[0][3], bboxes[1][3])
                bugflag = False
                bugNum = None
            elif len(bboxes) > 2:
                left = min(bboxes[0][0], bboxes[1][0])
                top = 0#min(bboxes[0][1], bboxes[1][1])
                right = max(bboxes[0][2], bboxes[1][2])
                bottom = max(bboxes[0][3], bboxes[1][3])
                bugflag = True
                bugNum = '001'  # raise ValueError(f"More than two identical targets detected, please check the data.")

            cropped_img = new_image.crop((left, top, right, bottom))
            cropped_img_width, cropped_img_height = cropped_img.size
            cropped_img = cropped_img.resize([int(200 / cropped_img_height * cropped_img_width), 200])

        else:
            left = 50
            top = 0
            right = 100
            bottom = 200
            bugflag = True
            bugNum = '002' # raise ValueError(f"Anomalous target detected, please check the data.")

            cropped_img = new_image.crop((left, top, right, bottom))
            cropped_img_width, cropped_img_height = cropped_img.size
            cropped_img = cropped_img.resize([int(200 / cropped_img_height * cropped_img_width), 200])
            

    return cropped_img, bugflag, bugNum

def chatReset(chatbot, hat_state, img_list):
    if chat_state is not None:
        chat_state.messages = []
    if img_list is not None:
        img_list = []
    if chatbot is not None:
        chatbot = []
    # print("Parameters have been reset")
    return chatbot, chat_state, img_list

def image_upload_trigger(upload_flag, replace_flag, img_list):
    # set the upload flag to true when receive a new image.
    # if there is an old image (and old conversation), set the replace flag to true to reset the conv later.
    upload_flag = 1
    if img_list:
        replace_flag = 1
    return upload_flag, replace_flag

def chatRebot(upload_flag, replace_flag, img_list, item, text_input, chatbot, chat_state):

    upload_flag, replace_flag = image_upload_trigger(upload_flag, replace_flag, img_list)
    image = item#Image.open(item).convert("RGB")
    # print(image)
    _, chatbot, chat_state, img_list, upload_flag, replace_flag = gradio_ask(text_input, chatbot, chat_state, image, img_list, upload_flag, replace_flag)
    llm_massage, _ = default_stream_answer(chatbot, chat_state, img_list, 0.6)
    chatbot, chat_state, img_list = chatReset(chatbot, chat_state, img_list)

    return llm_massage, image

def horizontal_concat_opencv(images, output_path):
    """
    Args:
        images (list): PIL.Image.Image 或 NumPy 数组列表
        output_path (str): 输出文件路径
    """
    # 转换为NumPy数组（BGR格式）
    np_images = [np.array(img)[:, :, ::-1] if isinstance(img, Image.Image) else img for img in images]
    
    concatenated = cv2.hconcat(np_images)
    
    # 保存结果（BGR转RGB）
    cv2.imwrite(output_path, concatenated)

def chatDataInit(cfg):
    '''
        '',
        '[grounding] describe this image in detail',
        '[refer] ',
        '[detection] ',
        '[identify] what is this ',
        '[vqa] '
    '''
    text_input = '[detection] person'
    chat_state = CONV_VISION.copy()
    img_list = []
    chatbot = []

    if not os.path.exists(cfg.args.save_path):
        os.makedirs(cfg.args.save_path)

    # fileList = [
    #     "/home/phdcv/wangxin/ActionPrompt/MiniGPT-4/testImages/A001/S001C001P001R001A001_rgb_frame_0.jpg",
    #     "/home/phdcv/wangxin/ActionPrompt/MiniGPT-4/testImages/A001/S001C001P001R001A001_rgb_frame_25.jpg",
    #     "/home/phdcv/wangxin/ActionPrompt/MiniGPT-4/testImages/A001/S001C001P001R001A001_rgb_frame_51.jpg",
    #     "/home/phdcv/wangxin/ActionPrompt/MiniGPT-4/testImages/A001/S001C001P001R001A001_rgb_frame_76.jpg",
    #     "/home/phdcv/wangxin/ActionPrompt/MiniGPT-4/testImages/A001/S001C001P001R001A001_rgb_frame_102.jpg"
    # ]

    return chatbot, text_input, chat_state, img_list, sorted(os.listdir(cfg.args.videos_path))


if __name__ == '__main__':

    args = parse_args()
    cfg = Config(args)

    CONV_VISION, chat = chat_initialization(cfg)

    # ========================================
    #             Default Inference
    # ========================================

    print('Inferring ...')

    chatbot, text_input, chat_state, img_list, videosList = chatDataInit(cfg)
    # print(videosList)

    print(f"The results save as: {cfg.args.save_path}")
    # videosList = [
    #     'S002C002P014R002A005_rgb.avi', 'S014C003P019R001A054_rgb.avi', 'S003C003P016R001A036_rgb.avi'
    # ]
    
    for filename in tqdm(videosList[45000:]):
        fileList = ExtractVideo(os.path.join(cfg.args.videos_path, filename))
    
        upload_flag, replace_flag = 0, 0
        crop_image_list = []

        for item in fileList:
            
            if '[detection]' in text_input:
                llm_massage, image = chatRebot(upload_flag, replace_flag, img_list, item, text_input, chatbot, chat_state)
                # llm_massage
                # [['[detection] person', '\\<p\\>a person\\</p\\> {\\<51\\>\\<28\\>\\<60\\>\\<75\\>}']]
                # [['[detection] person', '\\<p\\>two women\\</p\\> {\\<49\\>\\<27\\>\\<61\\>\\<79\\>}\\<delim\\>{\\<39\\>\\<25\\>\\<48\\>\\<80\\>}']]
                
                crop_image, bugflag, bugNum = default_crop_person(llm_massage, image)
                # print("crop_image:",crop_image)
                crop_image_list.append(crop_image)
            else:
                raise ValueError(f"Wrong task attribute, should be [detection] !")

        if bugflag:
            print(f"{filename} detection error, Error Code: {bugNum}")
            bugflag = False
            bugNum = None

        horizontal_concat_opencv(crop_image_list, os.path.join(cfg.args.save_path, filename.split('.')[0]+'.png'))

    print('Inference Done.')
