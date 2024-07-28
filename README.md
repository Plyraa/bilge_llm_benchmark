BİLGE (Türkçe Bilgi ve İdrak Eğitimi) is a benchmark designed to evaluate the performance of large language models (LLMs) in Turkish. It addresses the gap in resources for assessing Turkish language understanding and factual knowledge. BİLGE includes diverse categories, mainly STEM, literature, history, geography, and philosophy, utilizing multiple-choice questions to challenge LLMs. BİLGE introduces elements such as intentionally badly-typed questions and common misconceptions to test model robustness.

This repo is supposed to store evaluation script and our raw results.

We tested BİLGE on most relevant models available:
- GPT-4 Turbo
- GPT-4o mini
- GPT-4o
- Claude 3 Haiku
- Claude 3.5 Sonnet
- Gemini 1.5 Flash
- Llama 3.1 7B

To evaluate BİLGE on those models, we propose a script (complete-llm-evaluation.py) too!

To run the script, you will need to install needed libraries and API keys. We are using Groq for Llama 3.1 7B.

```
pip install requirements.txt
setx ANTHROPIC_API_KEY='$YOUR_API_KEY'
setx GOOGLE_API_KEY='$YOUR_API_KEY'
setx OPENAI_API_KEY='$YOUR_API_KEY'
setx GROQ_API_KEY='$YOUR_API_KEY'
```

Running following commands will give you our results, as all models are set to run at 0 temperature.
```
python .\complete-llm-evaluation.py -i diff_questions.csv -p chatgpt -m gpt-4-turbo-2024-04-09
python .\complete-llm-evaluation.py -i diff_questions.csv -p chatgpt -m gpt-4o-2024-05-13
python .\complete-llm-evaluation.py -i diff_questions.csv -p chatgpt -m gpt-4o-mini-2024-07-18
python .\complete-llm-evaluation.py -i diff_questions.csv -p claude -m claude-3-haiku-20240307
python .\complete-llm-evaluation.py -i diff_questions.csv -p claude -m claude-3-5-sonnet-20240620
python .\complete-llm-evaluation.py -i diff_questions.csv -p google -m gemini-1.5-flash-001
python .\complete-llm-evaluation.py -i diff_questions.csv -p llama -m llama-3.1-8b-instant
```

Our Python script supports OpenRouter so you can evaluate BİLGE on any model via OpenRouter. For instance, following command would evaluate BİLGE on Qwen 2 72B. <br/>
`python .\complete-llm-evaluation.py -i diff_questions.csv -p openrouter -m qwen/qwen-2-72b-instruct`
You need to setup OPENROUTER_API_KEY before using OpenRouter as a provider. Also our implementation is calling OpenRouter via its OpenAI integration.





Evaluation script uses RegEx to parse LLM output, if model failed to provide a choice or denied to answer, it evaluates to False and choice 'X' in results file.
