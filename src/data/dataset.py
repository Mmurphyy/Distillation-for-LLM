import json
import os
from typing import Dict, List, Optional, Union

import jsonlines
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer


class ExamDataset:
    """
    处理考试题目数据集的类
    支持加载ARC Challenge等格式的考试题目数据
    """
    def __init__(self, config):
        self.config = config
        self.cache_dir = config["data"]["cache_dir"]
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def load_jsonl(self, file_path: str) -> Dataset:
        """
        加载JSONL格式的考试题目数据
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
            
        records = []
        with jsonlines.open(file_path) as reader:
            for obj in reader:
                try:
                    question_data = obj
                    # 处理ARC格式
                    if isinstance(obj.get("question", None), str):
                        obj["question"] = json.loads(obj["question"])
                    
                    if "stem" not in obj and "question" in obj:
                        if isinstance(obj["question"], dict) and "stem" in obj["question"]:
                            question_data = obj["question"]
                            question_data["answerKey"] = obj.get("answerKey", "")
                            
                    # 提取并标准化数据
                    stem = question_data.get("stem", "")
                    choices = question_data.get("choices", [])
                    answer_key = question_data.get("answerKey", "")
                    
                    records.append({
                        "id": obj.get("id", ""),
                        "stem": stem,
                        "choices": choices,
                        "answer_key": answer_key
                    })
                except Exception as e:
                    print(f"Error processing record: {e}")
                    continue
                    
        return Dataset.from_pandas(pd.DataFrame(records))
    
    def preprocess_for_teacher_model(
        self,
        examples: Dict[str, List],
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 512,
    ) -> Dict[str, List]:
        """
        为教师模型准备输入数据
        """
        instructions = []
        for i in range(len(examples["stem"])):
            stem = examples["stem"][i]
            choices = examples["choices"][i]
            
            options_text = ""
            for choice in choices:
                label = choice.get("label", "")
                text = choice.get("text", "")
                options_text += f"{label}. {text}\n"
            
            instruction = f"问题：{stem}\n选项：\n{options_text.strip()}\n请分析这个问题并提供答案。"
            instructions.append(instruction)
        
        inputs = tokenizer(
            instructions,
            max_length=max_seq_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        return inputs
    
    def preprocess_for_student_model(
        self,
        examples: Dict[str, List],
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 512,
        is_training: bool = True,
    ) -> Dict[str, List]:
        """
        为学生模型准备输入数据
        """
        input_texts = []
        labels = []
        
        for i in range(len(examples["stem"])):
            stem = examples["stem"][i]
            choices = examples["choices"][i]
            answer_key = examples["answer_key"][i] if is_training else ""
            
            options_text = ""
            for choice in choices:
                label = choice.get("label", "")
                text = choice.get("text", "")
                options_text += f"{label}. {text}\n"
            
            input_text = f"问题：{stem}\n选项：\n{options_text.strip()}\n答案："
            input_texts.append(input_text)
            
            # 如果是训练状态且有答案，则添加答案作为标签
            if is_training and answer_key:
                labels.append(answer_key)
            else:
                labels.append("")
        
        model_inputs = tokenizer(
            input_texts,
            max_length=max_seq_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        if is_training:
            label_inputs = tokenizer(
                labels,
                max_length=8,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            model_inputs["labels"] = label_inputs["input_ids"]
        
        return model_inputs
    
    def create_dataloaders(
        self,
        tokenizer_teacher: Optional[PreTrainedTokenizer] = None,
        tokenizer_student: PreTrainedTokenizer = None,
    ) -> Dict[str, DataLoader]:
        """
        创建训练和验证数据加载器
        """
        train_path = self.config["data"]["train_file"]
        valid_path = self.config["data"].get("valid_file", "")
        test_path = self.config["data"].get("test_file", "")
        
        # 加载训练数据
        train_dataset = self.load_jsonl(train_path)
        
        # 加载或分割验证数据
        if valid_path and os.path.exists(valid_path):
            valid_dataset = self.load_jsonl(valid_path)
        else:
            # 如果没有验证集，从训练集中分割
            dataset_dict = train_dataset.train_test_split(test_size=0.1, seed=self.config["random_seed"])
            train_dataset = dataset_dict["train"]
            valid_dataset = dataset_dict["test"]
        
        # 加载测试数据（如果有）
        test_dataset = None
        if test_path and os.path.exists(test_path):
            test_dataset = self.load_jsonl(test_path)
        
        # 创建数据加载器
        dataloaders = {}
        
        max_seq_length = self.config["data"]["max_seq_length"]
        batch_size = self.config["data"]["batch_size"]
        valid_batch_size = self.config["data"]["valid_batch_size"]
        
        # 处理训练数据
        if tokenizer_student:
            # 预处理学生模型的数据
            train_encodings = self.preprocess_for_student_model(
                train_dataset, tokenizer_student, max_seq_length, is_training=True
            )
            train_dataset_student = Dataset.from_dict(train_encodings)
            
            dataloaders["train_student"] = DataLoader(
                train_dataset_student,
                batch_size=batch_size,
                shuffle=True,
                num_workers=self.config["data"]["num_workers"],
            )
            
            # 预处理验证数据
            valid_encodings = self.preprocess_for_student_model(
                valid_dataset, tokenizer_student, max_seq_length, is_training=True
            )
            valid_dataset_student = Dataset.from_dict(valid_encodings)
            
            dataloaders["valid_student"] = DataLoader(
                valid_dataset_student,
                batch_size=valid_batch_size,
                shuffle=False,
                num_workers=self.config["data"]["num_workers"],
            )
        
        # 处理教师模型的数据（如果提供了教师tokenizer）
        if tokenizer_teacher:
            train_encodings_teacher = self.preprocess_for_teacher_model(
                train_dataset, tokenizer_teacher, max_seq_length
            )
            train_dataset_teacher = Dataset.from_dict(train_encodings_teacher)
            
            # 确保教师模型的批大小至少为1
            teacher_batch_size = max(1, batch_size // 2)
            
            dataloaders["train_teacher"] = DataLoader(
                train_dataset_teacher,
                batch_size=teacher_batch_size,  # 教师模型通常更大，减小batch size
                shuffle=True,
                num_workers=self.config["data"]["num_workers"],
            )
        
        # 如果有测试数据，也创建测试数据加载器
        if test_dataset and tokenizer_student:
            test_encodings = self.preprocess_for_student_model(
                test_dataset, tokenizer_student, max_seq_length, is_training=False
            )
            test_dataset_student = Dataset.from_dict(test_encodings)
            
            dataloaders["test_student"] = DataLoader(
                test_dataset_student,
                batch_size=valid_batch_size,
                shuffle=False,
                num_workers=self.config["data"]["num_workers"],
            )
        
        return dataloaders 
