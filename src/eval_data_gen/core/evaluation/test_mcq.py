#!/usr/bin/env python3
import os
import re
import json
import glob
import time
import asyncio
import aiohttp
import google.generativeai as genai
import ollama

from typing import Dict, Any, List, Optional
from pathlib import Path
from tqdm.asyncio import tqdm # Use asyncio-compatible tqdm
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

QUESTIONS_DIR = os.getenv("QUESTIONS_DIR", "tmp/questions")
OUTPUT_FILE = os.getenv("EVAL_OUT", "tmp/eval_results_models.json")
QUESTION_FILE = os.getenv("EVAL_QUESTIONS_OUT", "tmp/eval_per_question.json")
DEBUG_DUMP_FILE = os.getenv("EVAL_DEBUG_OUT", "tmp/eval_debug_per_question.json")
DEBUG_DUMP = bool(int(os.getenv("DEBUG_DUMP", "0")))
SLEEP_BETWEEN_QUESTIONS = float(os.getenv("EVAL_SLEEP", "0.5"))


GEMINI_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_KEY = os.getenv("GROQ_API_KEY")
TOGETHER_KEY = os.getenv("TOGETHERAI_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")
PERPLEXITY_KEY = os.getenv("PERPLEXITY_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")


if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)
    _gemini_model = genai.GenerativeModel("gemini-2.5-flash")
if HF_API_KEY:
    _hf_client = InferenceClient(token=HF_API_KEY)


SYS_INSTRUCTIONS = (
    "You are a law expert. Your task is to select the single correct option for the given multiple-choice question.\n"
    "CRITICAL: Don't return your thinking, rather only the final answer. Your entire response must be ONLY one line in the format 'Answer: X', where X is the option number.\n"
    "DO NOT output any other text, reasoning, explanations, or think blocks. Your entire response must be 'Answer: X'. "
)


def build_prompt(mcq_text: str) -> str:
    lines = [line for line in mcq_text.splitlines() if not line.strip().lower().startswith("answer:")]
    return f"{SYS_INSTRUCTIONS}\n" + "\n".join(lines) + "\n\nYour answer:"


def extract_correct_index(mcq_text: str) -> int:
    ANS_RE = re.compile(r"Answer:\s*([A-F1-8])", re.IGNORECASE)
    LETTER2NUM = {"A": "1", "B": "2", "C": "3", "D": "4", "E": "5", "F": "6", "G": "7", "H": "8"}
    m = ANS_RE.search(mcq_text)
    if not m: raise ValueError("Answer tag missing.")
    token = m.group(1).strip().upper()
    return int(LETTER2NUM.get(token, token))


def extract_model_answer(text: str) -> Optional[int]:
    PATTERNS = [re.compile(p, re.M | re.I) for p in [
        r"Answer:\s*([1-8])", r"<final_answer>\s*([1-8])\s*</final_answer>",
        r"The correct option is \s*\(*([1-8])", r"The final answer is \s*\(*([1-8])",
        r"My choice is \s*\(*([1-8])", r"^\s*\(*([1-8])\)"
    ]]
    for pat in PATTERNS:
        m = pat.search(text)
        if m: return int(m.group(1))
    m = re.search(r"\b([1-8])\b", text)
    return int(m.group(1)) if m else None


def difficulty_from_ratio(r: float) -> str:
    if r >= 0.75:
        return "easy"
    if r >= 0.50:
        return "medium"
    if r >= 0.25:
        return "hard"
    return "very_hard"


# ASYNC Model Call Wrappers 

async def ask_gemini_async(session: aiohttp.ClientSession, prompt: str) -> str:
    try:
        resp = await _gemini_model.generate_content_async(prompt)
        return resp.text.strip()
    except Exception as e: return f"Error: {e}"


async def ask_async(session: aiohttp.ClientSession, prompt: str, model_name: str, url: str, key: str) -> str:
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    data = {"model": model_name, "messages": [{"role": "system", "content": SYS_INSTRUCTIONS}, {"role": "user", "content": prompt}], "temperature": 0.0, "max_tokens": 8192}
    try:
        async with session.post(url, headers=headers, json=data, timeout=20) as r:
            r.raise_for_status()
            res = await r.json()
            return res["choices"][0]["message"]["content"].strip()
    except Exception as e: return f"Error: {e}"


async def ask_cohere_async(session: aiohttp.ClientSession, prompt: str, model_name: str) -> str:
    url = "https://api.cohere.com/v1/chat"
    headers = {"Authorization": f"Bearer {COHERE_API_KEY}", "Content-Type": "application/json"}
    data = {"model": model_name, "preamble": SYS_INSTRUCTIONS, "message": prompt, "temperature": 0.0, "max_tokens": 128}
    try:
        async with session.post(url, headers=headers, json=data, timeout=20) as r:
            r.raise_for_status()
            res = await r.json()
            return res.get("text", "").strip()
    except Exception as e: return f"Error: {e}"


# async def ask_hf_async(session: aiohttp.ClientSession, prompt: str, model_name: str) -> str:
#     try:
#         msgs = [{"role": "system", "content": SYS_INSTRUCTIONS}, {"role": "user", "content": prompt}]
#         out = await _hf_client.chat_completion(model=model_name, messages=msgs, max_tokens=128, temperature=0.0)
#         return out.choices[0].message.content.strip()
#     except Exception as e: return f"Error: {e}"

# async def ask_ollama_async(session: aiohttp.ClientSession, prompt: str, model_name: str) -> str:
#     try:
#         response = await ollama.AsyncClient().chat(model=model_name, messages=[{"role": "system", "content": SYS_INSTRUCTIONS}, {"role": "user", "content": prompt}], options={"temperature": 0.0})
#         return response["message"]["content"].strip()
#     except Exception as e: return f"Error: {e}"




async def main() -> None:
    # get all questions 
    files = sorted(glob.glob(os.path.join(QUESTIONS_DIR, "*.json")))
    
    if not files:
        print("No question files found in", QUESTIONS_DIR)
        return
    
    all_mcqs = []
    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f: data = json.load(f)
        for q_raw in data.get("mcqs", []): 
            all_mcqs.append({
                "q_raw": q_raw, 
                "taxonomy_id": data.get("taxonomy_id", Path(fpath).stem),
                "file_path": fpath
            })

    print(f"Found {len(all_mcqs)} total questions to evaluate.")
    


    model_funcs = {}
    if GEMINI_KEY: 
        model_funcs["gemini_2.5_flash"] = lambda s, p: ask_gemini_async(s, p)
    if GROQ_KEY: 
        GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
        model_funcs["llama3_8b"] = lambda s, p: ask_async(s, p, "llama-3.1-8b-instant", GROQ_URL, GROQ_KEY)
        model_funcs["gemma2-9b-it"] = lambda s, p: ask_async(s, p, "gemma2-9b-it", GROQ_URL, GROQ_KEY)
        model_funcs["kimi-k2-instruct"] = lambda s, p: ask_async(s, p, "moonshotai/kimi-k2-instruct", GROQ_URL, GROQ_KEY)
        model_funcs["qwen3-32b"] = lambda s, p: ask_async(s, p, "qwen/qwen3-32b", GROQ_URL, GROQ_KEY)
        model_funcs["deepseek-r1"] = lambda s, p: ask_async(s, p, "deepseek-r1-distill-llama-70b", GROQ_URL, GROQ_KEY)
        # model_funcs["llama3_8b"] = lambda s, p: ask_groq_async(s, p, "llama-3.1-8b-instant")
        # model_funcs["gemma2-9b-it"] = lambda s, p: ask_groq_async(s, p, "gemma2-9b-it")
        # model_funcs["kimi-k2-instruct"] = lambda s, p: ask_groq_async(s, p, "moonshotai/kimi-k2-instruct")
        # model_funcs["qwen3-32b"] = lambda s, p: ask_groq_async(s, p, "qwen/qwen3-32b")
        # model_funcs["deepseek-r1"] = lambda s, p: ask_groq_async(s, p, "deepseek-r1-distill-llama-70b")
    if TOGETHER_KEY: 
        TOGETHER_URL = "https://api.together.xyz/v1/chat/completions"
        model_funcs["mistral7b"] = lambda s, p: ask_async(s, p, "mistralai/Mistral-7B-Instruct-v0.2", TOGETHER_URL, TOGETHER_KEY)
        # model_funcs["mistral7b"] = lambda s, p: ask_together_async(s, p, "mistralai/Mistral-7B-Instruct-v0.2")
    # if PERPLEXITY_KEY: 
        # PERPLEXITY_URL = "https://api.perplexity.ai/chat/completions"
        # model_funcs["perplexity_sonar"] = lambda s, p: ask_async(s, p, "sonar", PERPLEXITY_URL, PERPLEXITY_KEY)
    if COHERE_API_KEY: 
        model_funcs["cohere_command-a-03-2025"] = lambda s, p: ask_cohere_async(s, p, "command-a-03-2025")



    model_stats = {name: {"correct": 0, "incorrect": 0, "unanswered": 0, "total": 0} for name in model_funcs}
    question_rows = []
    debug_rows = []  # optional raw dump

    # Run evaluation concurrently
    async with aiohttp.ClientSession() as session:   #non blocking 
        for mcq_data in tqdm(all_mcqs, desc="Evaluating Questions"):
            q_raw = mcq_data["q_raw"]
            cur_fpath = mcq_data["file_path"]
            cur_taxonomy_id = mcq_data["taxonomy_id"]
            
            try:
                correct_idx = extract_correct_index(q_raw)
            except Exception as e:
                if DEBUG_DUMP:
                    debug_rows.append({
                        "file": cur_fpath,
                        "taxonomy_id": cur_taxonomy_id,
                        "question_raw": q_raw,
                        "error": str(e)
                    })
                continue
            prompt = build_prompt(q_raw)

            tasks = [func(session, prompt) for func in model_funcs.values()]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            print(cur_taxonomy_id, cur_fpath) #remove later

            q_correct = 0
            q_incorrect = 0
            q_unanswered = 0
            q_total = 0
            q_debug = {
                "file": cur_fpath, 
                "taxonomy_id": cur_taxonomy_id,
                "question_raw": q_raw, 
                "correct": correct_idx, 
                "models": {}
            }


            for i, model_name in enumerate(model_funcs.keys()):
                try:
                    model_out = results[i]
                    pred_idx = None # Reset for each model
                    corr = False    # Reset for each model
                    model_stats[model_name]["total"] += 1
                    if isinstance(model_out, Exception) or (isinstance(model_out, str) and model_out.startswith("Error:")):
                        model_stats[model_name]["unanswered"] += 1
                        q_unanswered += 1
                    else:
                        pred_idx = extract_model_answer(model_out)
                        if pred_idx is None:
                            model_stats[model_name]["unanswered"] += 1 
                            q_unanswered += 1
                        elif pred_idx == correct_idx:
                            model_stats[model_name]["correct"] += 1 
                            q_correct += 1
                        else:
                            model_stats[model_name]["incorrect"] += 1 
                            q_incorrect += 1
                    if DEBUG_DUMP:
                        q_debug["models"][model_name] = {
                            "answer": pred_idx, 
                            "correct": correct_idx, 
                            "raw": model_out
                        }


                except Exception as e:
                    # error → unanswered
                    model_stats[model_name]["total"] += 1
                    model_stats[model_name]["unanswered"] += 1
                    q_total += 1
                    q_unanswered += 1
                    if DEBUG_DUMP:
                        q_debug["models"][model_name] = {"error": str(e)}


            q_total = q_correct + q_incorrect + q_unanswered
            ratio = (q_correct / q_total) if q_total else 0.0
            question_rows.append({
                "taxonomy_id": cur_taxonomy_id, #mcq_data["taxonomy_id"], 
                "question_raw": q_raw, 
                "correct": correct_idx, 
                "correct_models": q_correct, 
                "incorrect_models": q_incorrect, 
                "unanswered_models": q_unanswered, 
                "total_models": q_total, 
                "correct_ratio": ratio, 
                "difficulty_tag": difficulty_from_ratio(ratio)
    
            })

            if DEBUG_DUMP:
                debug_rows.append(q_debug)

            await asyncio.sleep(SLEEP_BETWEEN_QUESTIONS)


    for st in model_stats.values(): st["accuracy"] = st["correct"] / st["total"] if st["total"] else 0
    summary = {"total_questions": len(all_mcqs), "models": model_stats}
    with open(OUTPUT_FILE, "w") as f: json.dump(summary, f, indent=2)
    with open(QUESTION_FILE, "w") as f: json.dump(question_rows, f, indent=2)

    if DEBUG_DUMP:
        print(f"Dumping debug info for {len(debug_rows)} questions...")
        with open(DEBUG_DUMP_FILE, "w", encoding="utf-8") as f:
            json.dump(debug_rows, f, indent=2)


    print("\n--- Evaluation Complete ---")
    for m, st in model_stats.items(): print(f"  {m:22s}  acc={st['accuracy']:.2%}  (C:{st['correct']} I:{st['incorrect']} U:{st['unanswered']})")
    print(f"Saved model summary to {OUTPUT_FILE}")
    print(f"Saved question report to {QUESTION_FILE}")
    if DEBUG_DUMP:
        print("Saved debug dump      →", DEBUG_DUMP_FILE)



if __name__ == "__main__":
    asyncio.run(main())





