# ActionNegPrompts
**Positive-Negative Co-Prompt Learning for Skeleton-based Action Recognition**

Xin Wang, Xi'an University of Tecnology

---

## Download Datasets
1. **NTU-RGB+D 60/120** dataset from [https://rose1.ntu.edu.sg/dataset/actionRecognition/](https://rose1.ntu.edu.sg/dataset/actionRecognition/)
2. **NW-UCLA** dataset from [https://wangjiangb.github.io/my_data.html](https://wangjiangb.github.io/my_data.html)
3. **kinetics-skeleton** dataset from [https://github.com/yysijie/st-gcn](https://github.com/yysijie/st-gcn)


## Getting Started
**1. MiniGPT Installation**

For the minigpt installation tutorial, refer to [https://github.com/Vision-CAIR/MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4). If the server cannot connect to the external network, refer to [https://blog.csdn.net/kikiLQQ/article/details/135731250](https://blog.csdn.net/kikiLQQ/article/details/135731250).

```bash
git clone https://github.com/Vision-CAIR/MiniGPT-4.git
cd MiniGPT-4
conda env create -f environment.yml
conda activate minigptv
cp CustomMiniGPT-v2.py ./
cp CustomMiniGPT4.py ./
```

**2. Generating RGB images and language descriptions**

Considering the balance of efficiency and performance, we generate RGB images based on the [detection] of MiniGPT-V2, and then generate language descriptions of each skeleton action based on MiniGPT4 as the positive prompt information for the network.

```bash
# step 1
python CustomMiniGPT-v2.py --cfg-path eval_configs/minigptv2_eval.yaml --save-path "your save path for rgb images" --videos-path "your videos path" --gpu-id 0

# step 2
python CustomMiniGPT4.py --cfg-path eval_configs/minigpt4_eval.yaml --save-path "your save path for language descriptions" --images-path "your images path from step 1" --gpu-id 0
```

**3. Training ActionNegPrompts**

++*This part of the code will be released as soon as possible.*++

## Acknowledgements

Our project is based on the [MMCL-Action](https://github.com/liujf69/MMCL-Action), [BLIP2](https://huggingface.co/docs/transformers/main/model_doc/blip-2), [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4). Thanks to the original authors for their work!

## Contact

For any questions, feel free to contact: ***wangxin2168@163.com***

### The complete project code is continuously being updated.
