import torch
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification
)
from datasets import load_dataset, Dataset
from textblob import TextBlob
import matplotlib.pyplot as plt
import copy
import random
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rlhf_model_name = "sehyun66/Tiny-lama-1.3B-chat-ppo"
rlhf_tokenizer = AutoTokenizer.from_pretrained(rlhf_model_name)
rlhf_model = AutoModelForCausalLM.from_pretrained(rlhf_model_name).to(device)
rlhf_pipe = pipeline(
    "text-generation",
    model=rlhf_model,
    tokenizer=rlhf_tokenizer,
    device=0 if torch.cuda.is_available() else -1
)

base_model_name = "gpt2"
base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base_model = AutoModelForCausalLM.from_pretrained(base_model_name).to(device)
base_pipe = pipeline(
    "text-generation",
    model=base_model,
    tokenizer=base_tokenizer,
    device=0 if torch.cuda.is_available() else -1
)

reward_model_name = "distilbert-base-uncased-finetuned-sst-2-english"
reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)
reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_name).to(device)
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model=reward_model,
    tokenizer=reward_tokenizer,
    device=0 if torch.cuda.is_available() else -1
)

evaluation_prompts = [
    "What is the capital of France?",
    "Explain the theory of relativity.",
    "How does photosynthesis work?",
    "What are the benefits of regular exercise?",
    "Can you describe the process of making bread?",
    "What is the significance of the Mona Lisa painting?",
    "How does blockchain technology function?",
    "What causes thunderstorms?",
    "Describe the impact of social media on society.",
    "What are the key features of Python programming language?"
]

def compute_reward(text, pipeline):
    result = pipeline(text)[0]
    label = result['label']
    if label == 'POSITIVE':
        return 1.0
    elif label == 'NEGATIVE':
        return -1.0
    else:
        return 0.0

def generate_single_response(pipe, prompt, max_length=100, num_return_sequences=1):
    response = pipe(
        prompt,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        do_sample=True,
        top_p=0.95,
        top_k=50,
        temperature=0.7,
        pad_token_id=pipe.tokenizer.eos_token_id
    )
    return response[0]['generated_text']

rlhf_responses = []
base_responses = []
rlhf_rewards = []
base_rewards = []

for prompt in evaluation_prompts:
    rlhf_resp = generate_single_response(rlhf_pipe, prompt)
    rlhf_responses.append(rlhf_resp)
    base_resp = generate_single_response(base_pipe, prompt)
    base_responses.append(base_resp)
    rlhf_reward = compute_reward(rlhf_resp, sentiment_pipeline)
    base_reward = compute_reward(base_resp, sentiment_pipeline)
    rlhf_rewards.append(rlhf_reward)
    base_rewards.append(base_reward)

print("\n=== Evaluation Results ===\n")
for i, prompt in enumerate(evaluation_prompts):
    print(f"Prompt: {prompt}")
    print(f"RLHF Response: {rlhf_responses[i]} | Reward: {rlhf_rewards[i]}")
    print(f"Base Response: {base_responses[i]} | Reward: {base_rewards[i]}")
    print("-" * 80)

indices = range(len(evaluation_prompts))
plt.figure(figsize=(14, 7))
plt.bar([i - 0.2 for i in indices], rlhf_rewards, width=0.4, label='RLHF Model', color='skyblue')
plt.bar([i + 0.2 for i in indices], base_rewards, width=0.4, label='Base Model', color='salmon')
plt.xticks(indices, [f"Prompt {i+1}" for i in indices], rotation=45)
plt.xlabel('Evaluation Prompts')
plt.ylabel('Reward Scores')
plt.title('Comparison of RLHF Fine-Tuned Model vs. Base Model')
plt.legend()
plt.tight_layout()
plt.show()
