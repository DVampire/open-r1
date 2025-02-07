import os

from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor
import torch
import random

class InsertTokenLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

        end_tokens = [". "]
        self.end_tokens = [tokenizer.encode(token, add_special_tokens=False)[0] for token in end_tokens]
        insert_tokens = [" Wait", " Hmm", " Alternatively"]
        self.insert_tokens = [tokenizer.encode(token, add_special_tokens=False)[0] for token in insert_tokens]

    def __call__(self, input_ids, scores):
        last_token_id = input_ids[:, -1].cpu().item()
        if last_token_id in self.end_tokens:
            if random.random() < 0.3:
                insert_token_id = random.choice(self.insert_tokens)
                scores[:, insert_token_id] = 30.0
        return scores

def run(mode="deepseek", turn = 0, model=None, tokenizer=None):

    SYSTEM_PROMPT = (
        "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
        "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
        "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
        "<think> reasoning process here </think><answer> answer here </answer>"
    )

    PROBLEM = """
    If $a > 1$, then the sum of the real solutions of $\sqrt{a - \sqrt{a+x}} = x$ is equal to
    """

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": PROBLEM}
    ]

    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")

    logits_processor = [InsertTokenLogitsProcessor(tokenizer)]

    if mode == "deepseek":
        output = model.generate(
            inputs,
            max_new_tokens=2048,
            temperature=1.2,
            return_dict_in_generate=True,
            top_p = 0.95,
            output_scores=True
        )
    elif mode == "qwen":
        output = model.generate(
            inputs,
            max_new_tokens=2048,
            temperature=1.2,
            return_dict_in_generate=True,
            top_p = 0.95,
            output_scores=True,
            logits_processor=logits_processor
        )
    else:
        raise ValueError("Invalid mode")

    generated_tokens = output.sequences[0][inputs.shape[1]:]
    decoded_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    res_str = "Generated text:\n" + decoded_text + "\n\n"

    check_words = [
        "Wait",
        "wait",
        "Hmm",
        "hmm",
    ]

    scores = output.scores
    for i, logits in enumerate(scores):

        res_str += "Step " + str(i) + ":" + "\n"
        res_str += "Generated token: " + tokenizer.decode([generated_tokens[i]]) + " (logit: " + str(logits[0, generated_tokens[i]].item()) + ")" + "\n"

        topk_logits, topk_indices = torch.topk(logits, k=10, dim=-1)
        topk_logits = topk_logits[0].cpu().numpy()
        topk_indices = topk_indices[0].cpu().numpy()
        logits = logits[0].cpu().numpy()

        # check words logits
        res_str += "Check words logits:\n"
        for j, word in enumerate(check_words):
            if mode == "deepseek":
                word_id = tokenizer.encode(word)[1]
            elif mode == "qwen":
                word_id = tokenizer.encode(word)[0]
            else:
                raise ValueError("Invalid mode")
            logit_val = logits[word_id]
            res_str += f"    {j+1}. {word} (logit: {logit_val:.4f})" + "\n"

        res_str += "Top 10 tokens:\n"
        for j in range(10):
            token_id = topk_indices[j]
            token_str = tokenizer.decode([token_id])
            logit_val = topk_logits[j]
            res_str += f"    {j + 1}. {token_str} (logit: {logit_val:.4f})" + "\n"
        res_str += "-" * 50 + "\n"

    with open(f"run/{mode}_token_{turn}.txt", "w") as f:
        f.write(res_str)

if __name__ == '__main__':
    os.makedirs("run", exist_ok=True)

    # model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    # tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16,
    #                                              trust_remote_code=True)
    # for i in range(5):
    #     run(mode="deepseek", turn = i, model=model, tokenizer=tokenizer)
    # del model
    # del tokenizer

    model_name = "Qwen/Qwen2.5-32B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16,
                                                 trust_remote_code=True)
    for i in range(5):
        run(mode="qwen", turn = i, model=model, tokenizer=tokenizer)
    del model
    del tokenizer