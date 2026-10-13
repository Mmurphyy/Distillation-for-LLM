#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试脚本：检查模型是否可以正常加载和运行
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 设置环境变量，避免tokenizers警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    print("加载模型和分词器...")
    
    # 加载教师模型
    teacher_model_path = "/home/zp/.cache/modelscope/hub/models/Qwen/Qwen3-4B"
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_path)
    
    # 使用Float16加载模型以节省内存
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        temperature=0.5,
        top_p=0.95,
        repetition_penalty=1.0,
    )
    
    print("模型加载成功，尝试生成文本...")
    
    # 准备输入
    prompt = "问题：地球是太阳系中第几颗行星？\n选项：\nA. 第一颗\nB. 第三颗\nC. 第五颗\nD. 第八颗\n答案："
    inputs = teacher_tokenizer(prompt, return_tensors="pt").to(teacher_model.device)

    print("==========这是inputs==========")
    print(inputs)
    print("==========这是inputs==========")

    # 进行推理
    with torch.no_grad():
        outputs = teacher_model.generate(
            inputs.input_ids,
            max_new_tokens=512,
            do_sample= False, # 是否使用采样，如果为True，则使用采样，如果为False，则使用贪心搜索
            num_beams=1, # 该参数数值越大，生成文本越多样
            temperature=0.2, # 该参数数值越大，则生成文本越随机
            top_p=0.6, #这个参数意思就是，在生成文本时，只考虑概率最高的top_p个词，其他词的概率为0
            repetition_penalty=0.5, #这个参数意思就是，在生成文本时，如果某个词重复出现，则降低其概率
        )
    
    # 解码输出
    print("==========这是outputs==========")
    print(outputs)
    print("==========这是outputs==========")

    print("==========这是outputs[0]==========")
    print(outputs[0])
    print("==========这是outputs[0]==========")


    print("==========这是teacher_tokenizer.special_tokens_map==========")
    print(teacher_tokenizer.special_tokens_map)
    print("==========这是teacher_tokenizer.special_tokens_map==========")

    for tok in teacher_tokenizer.all_special_tokens:
        print(tok, teacher_tokenizer.convert_tokens_to_ids(tok))

    result = teacher_tokenizer.decode(outputs[0], skip_special_tokens=False) 
    print(f"输入: {prompt}")
    print(f"模型输出: {result}")
    print("测试完成！")

if __name__ == "__main__":
    main() 