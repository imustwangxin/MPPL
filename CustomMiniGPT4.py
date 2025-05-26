import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr
from PIL import Image

from transformers import StoppingCriteriaList

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0, CONV_VISION_LLama2, StoppingCriteriaSub

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", default='eval_configs/minigpt4_llama2_eval.yaml', help="path to configuration file.")
    parser.add_argument("--images-path", default='/home/phdcv/wangxin/ActionPrompt/MiniGPT-4/testImages/testImages',
                        help="path to configuration file.")
    parser.add_argument("--save-path", default='/home/phdcv/wangxin/ActionPrompt/MiniGPT-4/testImages/testtexts',
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


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True

def chat_initialization(cfg):

    # ========================================
    #             Model Initialization
    # ========================================

    conv_dict = {'pretrain_vicuna0': CONV_VISION_Vicuna0,
                'pretrain_llama2': CONV_VISION_LLama2}

    print('Initializing Chat')

    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

    CONV_VISION = conv_dict[model_config.model_type]

    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

    stop_words_ids = [[835], [2277, 29937]]
    stop_words_ids = [torch.tensor(ids).to(device='cuda:{}'.format(args.gpu_id)) for ids in stop_words_ids]
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id), stopping_criteria=stopping_criteria)
    print('Initialization Finished')

    return CONV_VISION, chat

# ========================================
#             Default Inference
# ========================================

def chatReset(chat_state, img_list):
    if chat_state is not None:
        chat_state.messages = []
    if img_list is not None:
        img_list = []
    return chat_state, img_list

def chatRebot(image, text_input, chat_state, img_list):
    image = Image.open(image+'.png').convert("RGB")
    llm_massage = chat.upload_img(image, chat_state, img_list)
    chat.encode_img(img_list)
    chat.ask(text_input, chat_state)
    llm_massage = chat.answer(conv=chat_state, img_list=img_list)
    return llm_massage

def chatInit(cfg):
    # text_input = 'What is he/she(are they) trying to do? Answer the questions concisely'
    text_input = 'This is a single-person or two-person action sequence — please describe in one sentence what he/she/they are doing.'
    chat_state = CONV_VISION.copy()
    img_list = []
    # fileList = [
    #     "/home/phdcv/wangxin/ActionPrompt/MiniGPT-4/testImages/WIN_20240906_20_38_04_Pro.jpg",
    #     "/home/phdcv/wangxin/ActionPrompt/MiniGPT-4/testImages/drinkwater.png"
    # ]

    if not os.path.exists(cfg.args.save_path):
        os.makedirs(cfg.args.save_path)

    return text_input, chat_state, img_list, [x.split('.')[0] for x in sorted(os.listdir(cfg.args.images_path))], [x.split('.')[0] for x in os.listdir(cfg.args.save_path)]


if __name__ == '__main__':
    
    args = parse_args()
    cfg = Config(args)

    CONV_VISION, chat = chat_initialization(cfg)

    print('Inferring ...')

    text_input, chat_state, img_list, fileList, txtList = chatInit(cfg)

    for item in fileList:
        if item not in txtList:
            # print(item)
            textLines = []
            for count in range(10):
                llm_massage = chatRebot(os.path.join(cfg.args.images_path, item), text_input, chat_state, img_list)
                chat_state, img_list = chatReset(chat_state, img_list)
                textLines.append(llm_massage[0])
            with open(cfg.args.save_path+'/'+item+'.txt', "w", encoding="utf-8") as f:
                for line in textLines:
                    clean_line = line.replace("\n", "")  # 删除所有换行符
                    f.write(clean_line + "\n")
        else:
            print(f'{item} already exists')

    print('Inference Done.')