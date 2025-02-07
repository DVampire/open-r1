import os
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(verbose=True)

root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)

from litellm import completion
import json
import re
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

def pase_answer(answer: str):
    content_match = re.search(r'<answer>(.*?)</answer>', answer)
    answer = content_match.group(1).strip() if content_match else ""

    try:
        answer = float(answer)
        res_str = str(answer)
    except Exception as e:
        try:
            answer = answer.strip()
            pattern = r'\b[A-Z]\b'
            matches = re.findall(pattern, answer, re.DOTALL)
            matches = list(sorted(matches))
            if matches:
                res_str = ",".join(matches)
            else:
                res_str = ""
        except Exception as e:
            res_str = ""
    return res_str

def run(mode="deepseek", model=None, tokenizer=None):

    OUTPUT_PATH = os.path.join(root, f"workdir/Qwen2.5-1.5B-Instruct-Open-R1-GRPO/{mode}_results.json")
    PROMPT_PATH = os.path.join(root, "datasets/raw/ACCA/test.jsonl")
    BATCH_SIZE = 16

    # Load dataset
    with open(PROMPT_PATH, "r") as f:
        data = [json.loads(line) for line in f]

    def make_conversation(example):
        """Format the conversation"""
        # PROBLEM_FORMAT = "{problem} Output the thinking process in <think> </think> and final answer in <answer> </answer>."
        # return [
        #     {"role": "system", "content": SYSTEM_PROMPT},
        #     {"role": "user", "content": PROBLEM_FORMAT.format(problem=example["problem"])},
        # ]
        SYSTEM_PROMPT = (
            "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
            "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
            "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
            "<think> reasoning process here </think><answer> answer here </answer>"
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["problem"]},
        ]

    # Create batch messages
    messages = [make_conversation(item) for item in data]

    # Process in batches
    all_outputs = []

    if mode in ["deepseek", "qwen", "finreasoner"]:
        for i in tqdm(range(0, len(messages), BATCH_SIZE)):
            batch_messages = messages[i: i + BATCH_SIZE]

            # Tokenize with padding
            inputs = tokenizer.apply_chat_template(
                batch_messages, return_tensors="pt", padding=True, truncation=True
            ).to("cuda")

            # Generate output
            output = model.generate(
                inputs,
                max_new_tokens=2048,
                temperature=0.6,
                return_dict_in_generate=True,
                top_p=0.95,
                output_scores=True
            )

            # Decode batch outputs
            batch_decoded_texts = tokenizer.batch_decode(output.sequences[:, inputs.shape[1]:], skip_special_tokens=True)
            all_outputs.extend(batch_decoded_texts)
    elif mode == "gpt-4o":
        for message in messages:
            response = completion(model="openai/gpt-4o", messages=message)

            text = response.choices[0].message.content
            print(text)
            all_outputs.append(text)

    # Store results
    final_output = []
    correct_number = 0

    for input_example, model_output in zip(data, all_outputs):
        problem = input_example['problem']
        correct_answer = input_example['answer']
        extracted_answer = pase_answer(model_output)

        result = {
            'problem': problem,
            'answer': correct_answer,
            'model_output': model_output,
            'extracted_answer': extracted_answer
        }
        final_output.append(result)

        if extracted_answer is not None and extracted_answer == correct_answer:
            correct_number += 1

    # Compute accuracy
    accuracy = correct_number / len(data) * 100
    print(f"Mode: {mode} \nAccuracy: {accuracy:.2f}%")

    # Save results
    with open(OUTPUT_PATH, "w") as f:
        json.dump({'accuracy': accuracy, 'results': final_output}, f, indent=2)
    print(f"Results saved to {OUTPUT_PATH}")

if __name__ == '__main__':
    model_name = "zwt963/Qwen2.5-1.5B-Instruct-Open-R1-GRPO"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16,
                                                 trust_remote_code=True)
    run(mode = "finreasoner", model=model, tokenizer=tokenizer)
    del model
    del tokenizer
    #
    # model_name = "Qwen/Qwen2.5-32B"
    # tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16,
    #                                              trust_remote_code=True)
    # run(mode = "qwen", model=model, tokenizer=tokenizer)
    # del model
    # del tokenizer
    #
    # model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    # tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16,
    #                                              trust_remote_code=True)
    # run(mode = "deepseek", model=model, tokenizer=tokenizer)
    # del model
    # del tokenizer

    # run(mode = "gpt-4o", model=None, tokenizer=None)