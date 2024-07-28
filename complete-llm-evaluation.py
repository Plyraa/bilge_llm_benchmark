import argparse
import os
import re
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Dict

import google.generativeai as genai
import numpy as np
import pandas as pd
from anthropic import Anthropic
from groq import Groq
from openai import OpenAI

choices = ["A", "B", "C", "D"]

results_directory = "placeholder"


class LLMWrapper(ABC):
    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt

    @abstractmethod
    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        pass


class LlamaWrapper(LLMWrapper):
    def __init__(self, model: str, system_prompt: str):
        super().__init__(system_prompt)
        self.model_name = model
        self.client = Groq()

    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        # Prepend the system prompt to the messages
        full_messages = [{"role": "system", "content": self.system_prompt}] + messages

        try:
            chat_completion = self.client.chat.completions.create(
                messages=full_messages,
                model=self.model_name,
                temperature=0
                # top_k = 0.1
                # We observed repetition issues while testing
                # https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct/discussions/32
                # https://www.reddit.com/r/SillyTavernAI/comments/1cvzv32/llama_3_just_keeps_repeating_the_same_description/
                # even after playing with temperature/top_k/prompts, it is still there and affects precision
            )
            ai_response = chat_completion.choices[0].message.content
            print('Question: ' + messages.pop()["content"])
            print('LLM RESPONSE: ' + ai_response)
            return ai_response
        except Exception as e:
            print(f"Error: {e}. Retrying in 1 second...")
            time.sleep(1)
            return self.generate_response(messages)  # Retry the request


class ChatGPTWrapper(LLMWrapper):
    # MODELS:
    # gpt-4o = gpt-4o-2024-05-13
    # gpt-4o-mini = gpt-4o-mini-2024-07-18
    # gpt-4-turbo = gpt-4-turbo-2024-04-09

    def __init__(self, model: str, system_prompt: str):
        super().__init__(system_prompt)
        self.client = OpenAI()
        self.model = model

    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        messages = [{"role": "system", "content": self.system_prompt}] + messages
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0,
                )
                print('Question: ' + messages.pop()["content"])
                print('LLM RESPONSE:' + response.choices[0].message.content)
                return response.choices[0].message.content
            except Exception as e:
                print(f"Error: {e}. Retrying in 1 second...")
                time.sleep(1)


class OpenRouterWrapper(LLMWrapper):
    def __init__(self, model: str, system_prompt: str):
        super().__init__(system_prompt)
        self.model = model
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )

    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        full_messages = [{"role": "system", "content": self.system_prompt}] + messages

        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=full_messages,
                    temperature=0,
                )
                ai_response = response.choices[0].message.content
                print('Question: ' + messages[-1]["content"])
                print('LLM RESPONSE: ' + ai_response)
                return ai_response
            except Exception as e:
                print(f"Error: {e}. Retrying in 1 second...")
                time.sleep(1)


class ClaudeWrapper(LLMWrapper):
    # claude-3-haiku-20240307
    # claude-3-opus-20240229
    # claude-3-5-sonnet-20240620
    def __init__(self, model: str, system_prompt: str):
        super().__init__(system_prompt)
        self.client = Anthropic()
        self.model = model

    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        while True:
            try:
                response = self.client.messages.create(
                    model=self.model,
                    messages=messages,
                    system=self.system_prompt,
                    max_tokens=1000,
                    temperature=0,
                )
                print('Question: ' + messages.pop()["content"])
                print('LLM RESPONSE: ' + response.content[0].text)
                return response.content[0].text
            except Exception as e:
                print(f"Error: {e}. Retrying in 1 second...")
                time.sleep(1)


class GeminiWrapper(LLMWrapper):

    def __init__(self, model: str, system_prompt: str):
        super().__init__(system_prompt)
        self.model_name = model
        genai.configure()
        self.model = genai.GenerativeModel(
            self.model_name,
            system_instruction=system_prompt,
            generation_config=genai.GenerationConfig(
                max_output_tokens=2000,
                temperature=0  # Set to 0 for deterministic output
            )
        )

    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        # Convert messages to the format expected by Gemini
        gemini_messages = [
            {"role": msg["role"], "parts": [msg["content"]]}
            for msg in messages
        ]

        try:
            response = self.model.generate_content(contents=gemini_messages)
            ai_response = response.text
            print('Question: ' + messages.pop()["content"])
            print('LLM RESPONSE: ' + ai_response)
            return ai_response
        except Exception as e:
            print(f"Error: {e}. Retrying in 1 second...")
            time.sleep(1)
            return self.generate_response(messages)  # Retry the request


def get_llm_wrapper(provider: str, model: str, system_prompt: str) -> LLMWrapper:
    if provider == "chatgpt":
        return ChatGPTWrapper(model, system_prompt)
    elif provider == "claude":
        return ClaudeWrapper(model, system_prompt)
    elif provider == "google":
        return GeminiWrapper(model, system_prompt)
    elif provider == "llama":
        return LlamaWrapper(model, system_prompt)
    elif provider == "openrouter":
        return OpenRouterWrapper(model, system_prompt)
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def format_example(row, include_answer=True):
    prompt = f"Question: {row['Question']}\n"
    for choice in choices:
        prompt += f"{choice}. {row[choice]}\n"
    if include_answer:
        prompt += f"Answer: {row['Answer']}\n\n"
    return prompt


def gen_prompt(df, k):
    prompt = "The following are multiple choice questions (with answers). Please answer new questions based on this format.\n\n"
    for _, row in df.head(k).iterrows():
        prompt += format_example(row)
    return prompt


def eval(timestamp, args, llm_wrapper: LLMWrapper, df):
    cors = []
    all_preds = []

    for i, row in df.iterrows():
        prompt_end = format_example(row, include_answer=False)
        # train_prompt = gen_prompt(df.iloc[:args.ntrain], args.ntrain)
        # In context learning doesn't work with Claude. Doesn't even bother trying.
        # It cost my money and sanity.

        # messages = [
        #    {"role": "user", "content": train_prompt + prompt_end + "Answer:"}
        # ]

        messages = [
            {"role": "user", "content": prompt_end + "Answer:"}
        ]
        label = row['Answer']

        pred = llm_wrapper.generate_response(messages)
        save_llm_answer(timestamp, results_directory, args.provider, args.model, pred)
        pred = parse_llm_response(pred)

        cor = pred == label
        print("prediction:" + str(cor) + "correct label:" + label, "predicted label:" + pred)
        cors.append(cor)
        all_preds.append(pred)

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1} questions. Current accuracy: {np.mean(cors):.3f}")

    acc = np.mean(cors)
    print(f"Final accuracy: {acc:.3f}")

    return cors, all_preds


def main(args):
    df = pd.read_csv(args.input_file)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if args.provider == "chatgpt":
        system_prompt = "You are a helpful assistant that answers multiple choice questions accurately."
    elif args.provider == "claude":
        system_prompt = "You are Claude, an AI assistant created by Anthropic. Answer multiple choice questions accurately."
    elif args.provider == "google":
        system_prompt = "You are a helpful assistant that answers multiple choice questions accurately."
    elif args.provider == "llama":
        system_prompt = "You are a helpful assistant that answers multiple choice questions and clearly specify which choice is correct. Do not repeat your descriptions."
    elif args.provider == "openrouter":
        system_prompt = "You are a helpful assistant that answers multiple choice questions and clearly specify which choice is correct. Do not repeat your descriptions."
    else:
        system_prompt = "You are a helpful assistant that answers multiple choice questions accurately."

    llm_wrapper = get_llm_wrapper(args.provider, args.model, system_prompt)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cors, preds = eval(timestamp, args, llm_wrapper, df)

    df['correct'] = cors
    df['prediction'] = preds

    output_file = os.path.join(args.save_dir, f"results_{args.provider}_{args.model}.csv")
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")


def parse_llm_response(pred):
    # Convert to lowercase for case-insensitive matching
    pred_lower = pred.lower()

    # List of possible answer indicators in English and Turkish
    answer_indicators = [
        r'answer\s*[is:]*\s*([a-d])',  # "answer is X" or "answer: X"
        r'cevap\s*[:\s]*([a-d])',  # "cevap X" or "cevap: X"
        r'\b([a-d])\s*is correct',  # "X is correct"
        r'\b([a-d])\s*doğru',  # "X doğru" (Turkish)
        r'correct answer\s*[is:]*\s*([a-d])',  # "correct answer is X"
        r'doğru cevap\s*[:\s]*([a-d])'  # "doğru cevap X" (Turkish)
    ]
    print("parser:" + pred_lower)
    parsed_choice = ''
    # Check for each pattern
    for pattern in answer_indicators:
        match = re.search(pattern, pred_lower)
        if match:
            parsed_choice = match.group(1).upper()
    else:
        # If no pattern matched, look for standalone A, B, C, or D
        standalone_match = re.search(r'\b([a-d])\b', pred_lower)
        if standalone_match:
            parsed_choice = standalone_match.group(1).upper()

    # If still no match found
    if parsed_choice == '':
        parsed_choice = 'X'

    return parsed_choice


def save_llm_answer(timestamp, dir, provider, model, pred):
    """
    Save the LLM's answer to a file.

    Args:
    args (argparse.Namespace): Command-line arguments containing provider and model
    pred (str): The LLM's prediction/answer
    """
    # edge case for windows naming
    if model == "meta-llama/llama-3.1-8b-instruct:free":
        model = "llama-3.1-8b-instruct"

    filename = f"answers_{provider}_{model}_{timestamp}.txt"
    os.path.join(dir)
    # Ensure the directory exists
    os.makedirs(os.path.dirname(os.path.join(dir, "answers")), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.join(dir, filename)), exist_ok=True)

    # Append the prediction to the file
    with open(filename, 'a', encoding='utf-8') as f:
        f.write(pred + '\n' + "---\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", "-i", type=str, default="questions.csv",
                        help="Path to the input CSV file")
    parser.add_argument("--ntrain", "-k", type=int, default=0, help="Number of examples to use for in-context learning")
    parser.add_argument("--save_dir", "-s", type=str, default="results", help="Directory to save results")
    parser.add_argument("--provider", "-p", choices=["chatgpt", "claude", "google", "llama", "openrouter"],
                        help="LLM provider to use")
    parser.add_argument("--model", "-m", type=str, help="Model name for the chosen provider")
    args = parser.parse_args()
    results_directory = args.save_dir
    main(args)
