import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from tqdm import tqdm
import argparse
import time
import re
import gc

MODEL_NAME = "/data1/s3905993/ECRHMAS/src/models/mistral7b_instruct/model_files"   # Local model files

def create_mistral_prompt_ultra_fast(context, response):
    context_str = " ".join(context) if isinstance(context, list) else str(context)
    prompt = (
        "You are a strict evaluator. Rate this response using scores between 0.0 (worst) and 1.0 (best). "
        "Respond with ONLY the JSON scores, nothing else.\n\n"
        f"Context: {context_str}\n\n"
        f"Response: {response}\n\n"
        "Output this exact JSON format:\n"
        "{\n"
        '"empathy_score": [score],\n'
        '"informativeness_score": [score],\n'
        '"recommendation_score": [score],\n'
        '"engagement_score": [score]\n'
        "}"
    )
    return prompt

def parse_scores_ultra_fast(result):
    import re
    json_matches = re.finditer(r'\{.*?\}', result, re.DOTALL)
    for match in json_matches:
        try:
            scores = json.loads(match.group())
            required_keys = ["empathy_score", "informativeness_score", "recommendation_score", "engagement_score"]
            if all(k in scores for k in required_keys):
                return {k: float(scores[k]) for k in required_keys}
        except Exception as e:
            continue
    try:
        pattern = r'"([^"]+)_score":\s*([0-9]*\.?[0-9]+)'
        matches = re.findall(pattern, result)
        if len(matches) >= 4:
            scores = {}
            for key, value in matches:
                if key in ["empathy", "informativeness", "recommendation", "engagement"]:
                    scores[f"{key}_score"] = float(value)
            required_keys = ["empathy_score", "informativeness_score", "recommendation_score", "engagement_score"]
            if all(k in scores for k in required_keys):
                return scores
    except Exception as e:
        pass
    try:
        numbers = re.findall(r'([0-9]*\.?[0-9]+)', result)
        if len(numbers) >= 4:
            scores = {
                "empathy_score": float(numbers[0]),
                "informativeness_score": float(numbers[1]),
                "recommendation_score": float(numbers[2]),
                "engagement_score": float(numbers[3])
            }
            return scores
    except Exception as e:
        pass
    return None

def get_mistral_scores_ultra_fast(prompt_string, model, tokenizer):
    chat_prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are a JSON-only response generator. Always respond with valid JSON only, no other text."},
            {"role": "user", "content": prompt_string}
        ],
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    output = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=True,
        temperature=0.1,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.1
    )
    result = tokenizer.decode(output[0], skip_special_tokens=True)
    assistant_reply = result[len(chat_prompt):].strip()
    scores = parse_scores_ultra_fast(assistant_reply)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return scores, assistant_reply

def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='llama2_prompts.jsonl', help='Input file')
    parser.add_argument('--output', type=str, default='mistral7b_scored_ultra_fast.jsonl', help='Output file')
    parser.add_argument('--max', type=int, default=None, help='Max number of prompts to process (for testing)')
    args = parser.parse_args()

    print("Loading Mistral-7B-Instruct model and tokenizer with 4-bit quantization...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Configure 4-bit quantization for memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    print(f"Model loaded. Device: {model.device}")
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB allocated")

    print("Counting total samples...")
    with open(args.input) as fin:
        total_lines = sum(1 for _ in fin)
    if args.max:
        total_lines = min(total_lines, args.max)
    print(f"Total samples to process: {total_lines}")

    start_time = time.time()
    all_scores = []
    successful_parses = 0
    total_processed = 0
    with open(args.input) as fin, open(args.output, "w") as fout:
        for i, line in enumerate(fin):
            if args.max is not None and i >= args.max:
                break
            total_processed += 1
            item = json.loads(line)
            context = item.get("context", "")
            response = item.get("response", item.get("resp", ""))
            prompt = create_mistral_prompt_ultra_fast(context, response)
            scores, full_output = get_mistral_scores_ultra_fast(prompt, model, tokenizer)
            if scores:
                successful_parses += 1
                overall = sum(scores.values()) / 4.0
                scores["overall_score"] = round(overall, 4)
                item["mistral7b_scores"] = scores
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                all_scores.append(scores)
            else:
                print(f"\n[PARSE FAIL] Raw model output for sample {i+1}:\n{full_output}\n")
                item["mistral7b_scores"] = {"error": "parsing failed"}
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            if i < 5:
                print(f"\n=== Sample {i+1} ===")
                if scores:
                    print(json.dumps(scores, indent=2))
                else:
                    print(json.dumps({"error": "parsing failed"}, indent=2))
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                parse_rate = (successful_parses / total_processed) * 100 if total_processed > 0 else 0
                remaining_samples = total_lines - (i + 1)
                eta_seconds = remaining_samples / rate if rate > 0 else 0
                eta_str = format_time(eta_seconds)
                gpu_mem = f"{torch.cuda.memory_allocated()/1024**3:.2f}GB" if torch.cuda.is_available() else "N/A"
                print(f"[PROGRESS] {i+1}/{total_lines} processed | Parse success: {parse_rate:.1f}% | Rate: {rate:.2f} samples/sec | ETA: {eta_str} | GPU Mem: {gpu_mem}")
                if (i + 1) % 50 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
    print(f"\n===== FINAL STATISTICS =====")
    print(f"Total processed: {total_processed}")
    print(f"Successful parses: {successful_parses}")
    print(f"Parse success rate: {(successful_parses/total_processed)*100:.1f}%")
    print(f"Total time: {format_time(time.time() - start_time)}")
    print(f"Average rate: {total_processed/(time.time() - start_time):.2f} samples/sec")
    print(f"\n===== SCORES FOR FIRST 5 RESPONSES =====")
    for idx, s in enumerate(all_scores[:5]):
        print(f"Sample {idx+1}: {s}")
    print("\nUltra-fast scoring complete.")

if __name__ == "__main__":
    main() 