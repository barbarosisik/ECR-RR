"""
LLM-Based Subjective Evaluation for ECR System
Replicates the methodology from the ECR paper (arXiv:2409.10527v1)
Evaluates responses on 5 dimensions: Emo Int, Emo Pers, Log Pers, Info, Life
"""

import json
import random
import time
import argparse
from typing import List, Dict, Any
import os

#optional OpenAI import (only if using API)
try:
    import openai  # type: ignore
except Exception:  # pragma: no cover
    openai = None

from transformers import AutoTokenizer, AutoModelForCausalLM
try:
    from transformers import BitsAndBytesConfig  # type: ignore
except Exception:  # pragma: no cover
    BitsAndBytesConfig = None  # type: ignore
import torch


class LLMSubjectiveEvaluator:
    def __init__(self, scorer_model="gpt-4-turbo", api_key=None, local_model=None,
                 load_in_8bit: bool = False, force_cpu: bool = False,
                 progress_interval: int = 50, dtype: str = "fp32"):
        """
        Initialize the LLM-based subjective evaluator
        
        Args:
            scorer_model: OpenAI model to use for scoring (gpt-4-turbo, gpt-3.5-turbo)
            api_key: OpenAI API key
            local_model: Local model path or HF id (e.g., "openai/gpt-oss-20b")
            load_in_8bit: If True, attempt 8-bit loading for large local models
        """
        self.scorer_model = scorer_model
        self.local_model = local_model
        self.load_in_8bit = load_in_8bit
        self.force_cpu = force_cpu
        self.progress_interval = max(1, progress_interval)
        self.dtype = dtype
        
        if api_key and openai is not None:
            openai.api_key = api_key
        
        #initializing local model if specified
        self.tokenizer = None
        self.model = None
        if local_model:
            load_kwargs = {"trust_remote_code": True}
            #ensuring a valid current CUDA device is selected when GPUs are present (IF NOT NECESSARY, PLEASE REMOVE WHILE REPRODUCING)
            try:
                if not self.force_cpu and torch.cuda.is_available():
                    torch.cuda.set_device(0)
            except Exception:
                pass
            if self.force_cpu:
                load_kwargs["device_map"] = "cpu"
                if self.dtype == "bf16":
                    load_kwargs["torch_dtype"] = torch.bfloat16
                    try:
                        torch.set_default_dtype(torch.bfloat16)
                    except Exception:
                        pass
                else:
                    load_kwargs["torch_dtype"] = torch.float32
            else:
                load_kwargs["device_map"] = "auto"
                #preferring BitsAndBytesConfig over deprecated load_in_8bit flag
                if load_in_8bit and BitsAndBytesConfig is not None:
                    load_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_enable_fp32_cpu_offload=True,
                        bnb_8bit_compute_dtype=(torch.float16 if self.dtype in ("fp16", "bf16") else torch.float32),
                    )
                    #keep dtype to help scheduler choose kernels
                    load_kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
                else:
                    if self.dtype == "bf16":
                        load_kwargs["torch_dtype"] = torch.bfloat16
                    elif self.dtype == "fp16":
                        load_kwargs["torch_dtype"] = torch.float16
                    else:
                        load_kwargs["torch_dtype"] = torch.float32
                load_kwargs["low_cpu_mem_usage"] = True
                try:
                    if torch.cuda.is_available():
                        num_gpus = torch.cuda.device_count()
                        if num_gpus and num_gpus > 0:
                            #accelerate expects integer device indices (0,1,...) and 'cpu'
                            max_memory = {i: "8GiB" for i in range(num_gpus)}
                            max_memory["cpu"] = "160GiB"
                            load_kwargs["max_memory"] = max_memory
                            load_kwargs["offload_folder"] = os.environ.get("HF_HOME", "/data1/s3905993/hf_cache")
                except Exception:
                    pass

            # Some open-weight models ship fast-tokenizer JSONs that fail to parse in certain versions.
            # Fallback to slow tokenizer to avoid JSON parsing errors.
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(local_model, use_fast=False, trust_remote_code=True)
            except Exception:
                self.tokenizer = AutoTokenizer.from_pretrained(local_model, use_fast=False)
            try:
                self.model = AutoModelForCausalLM.from_pretrained(local_model, **load_kwargs)
            except torch.cuda.OutOfMemoryError:
                #fallback: reload entirely on CPU
                load_kwargs["device_map"] = "cpu"
                load_kwargs["torch_dtype"] = torch.bfloat16 if self.dtype == "bf16" else torch.float32
                self.model = AutoModelForCausalLM.from_pretrained(local_model, **load_kwargs)
            #ensuring model dtype alignment if CPU bf16 requested
            if self.force_cpu and self.dtype == "bf16":
                try:
                    self.model = self.model.to(dtype=torch.bfloat16)
                except Exception:
                    pass
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Define evaluation prompts based on ECR paper methodology
        # Strengthen instruction: return only a single number (0-9)
        self.evaluation_prompts = {
            "emotional_intensity": (
                "You are evaluating a conversational recommender system's response for emotional intensity.\n"
                "Return only a single number from 0 to 9 with no text.\n\n"
                "Response to evaluate: \"{response}\"\n\n"
                "Score (0-9):"
            ),
            "emotional_persuasiveness": (
                "You are evaluating a conversational recommender system's response for emotional persuasiveness.\n"
                "Return only a single number from 0 to 9 with no text.\n\n"
                "Response to evaluate: \"{response}\"\n\n"
                "Score (0-9):"
            ),
            "logic_persuasiveness": (
                "You are evaluating a conversational recommender system's response for logic persuasiveness.\n"
                "Return only a single number from 0 to 9 with no text.\n\n"
                "Response to evaluate: \"{response}\"\n\n"
                "Score (0-9):"
            ),
            "informativeness": (
                "You are evaluating a conversational recommender system's response for informativeness.\n"
                "Return only a single number from 0 to 9 with no text.\n\n"
                "Response to evaluate: \"{response}\"\n\n"
                "Score (0-9):"
            ),
            "lifelikeness": (
                "You are evaluating a conversational recommender system's response for lifelikeness.\n"
                "Return only a single number from 0 to 9 with no text.\n\n"
                "Response to evaluate: \"{response}\"\n\n"
                "Score (0-9):"
            ),
        }
    
    def score_with_openai(self, response: str, dimension: str) -> float:
        """Score a response using OpenAI API"""
        if openai is None:
            print("OpenAI package not available; defaulting to 5.0")
            return 5.0
        prompt = self.evaluation_prompts[dimension].format(response=response)
        try:
            response_openai = openai.ChatCompletion.create(
                model=self.scorer_model,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator of conversational AI responses. Provide only the numerical score (0-9) as your response."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=4
            )
            score_text = response_openai.choices[0].message.content.strip()
            try:
                score = float(score_text)
                return max(0, min(9, score))
            except ValueError:
                return 5.0
        except Exception as e:
            print(f"Error scoring with OpenAI: {e}")
            return 5.0
    
    def score_with_local_model(self, response: str, dimension: str) -> float:
        """Score a response using local HF model"""
        if self.model is None or self.tokenizer is None:
            return 5.0
        prompt = self.evaluation_prompts[dimension].format(response=response)
        try:
            #use chat template if available (for chat-tuned models like Llama-2-Chat)
            if hasattr(self.tokenizer, "apply_chat_template"):
                messages = [
                    {"role": "system", "content": "You are an expert evaluator of conversational AI responses. Provide only the numerical score (0-9) as your response."},
                    {"role": "user", "content": prompt},
                ]
                templated = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
                inputs = self.tokenizer(templated, return_tensors="pt", truncation=True, max_length=512)
            else:
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            #ensuring input dtype matches model dtype
            if hasattr(self.model, 'dtype'):
                inputs = {k: v.to(dtype=self.model.dtype) for k, v in inputs.items()}
            input_len = inputs["input_ids"].shape[1]
            with torch.no_grad():
                #matching autocast with model dtype to avoid bf16/half mismatches
                try:
                    first_param = next(self.model.parameters())
                    if first_param.is_cuda and first_param.dtype == torch.bfloat16:
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            outputs = self.model.generate(
                                **inputs,
                                max_new_tokens=3,
                                do_sample=False,
                                pad_token_id=self.tokenizer.eos_token_id,
                                eos_token_id=self.tokenizer.eos_token_id,
                            )
                    elif first_param.is_cuda and first_param.dtype == torch.float16:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            outputs = self.model.generate(
                                **inputs,
                                max_new_tokens=3,
                                do_sample=False,
                                pad_token_id=self.tokenizer.eos_token_id,
                                eos_token_id=self.tokenizer.eos_token_id,
                            )
                    else:
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=3,
                            do_sample=False,
                            pad_token_id=self.tokenizer.eos_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                        )
                except StopIteration:
                outputs = self.model.generate(
                    **inputs,
                        max_new_tokens=3,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
            seq = outputs[0]
            gen_ids = seq[input_len:]
            gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            import re
            m = re.search(r"(?<!\\d)([0-9](?:\\.[0-9])?)(?!\\d)", gen_text)
            if m:
                score = float(m.group(1))
                return max(0, min(9, score))
            #if the model produced nothing numeric, return neutral 5.0
                return 5.0
        except Exception as e:
            print(f"Error scoring with local model: {e}")
            return 5.0
    
    def evaluate_response(self, response: str) -> Dict[str, float]:
        scores = {}
        for dimension in self.evaluation_prompts.keys():
            if self.local_model:
                score = self.score_with_local_model(response, dimension)
            else:
                score = self.score_with_openai(response, dimension)
            scores[dimension] = score
            time.sleep(0.05)
        return scores
    
    def evaluate_dataset(self, data: List[Dict[str, Any]], sample_size: int = 1000) -> Dict[str, Any]:
        if sample_size and len(data) > sample_size:
            data = random.sample(data, sample_size)
        total = len(data)
        print(f"Evaluating {total} responses...")
        all_scores = {k: [] for k in [
            "emotional_intensity", "emotional_persuasiveness", "logic_persuasiveness", "informativeness", "lifelikeness"
        ]}
        results = []
        start_ts = time.time()
        for i, item in enumerate(data):
            response = item.get('pred', '') or item.get('response', '')
            if not response.strip():
                continue
            if (i + 1) % self.progress_interval == 0:
                elapsed = time.time() - start_ts
                rate = (i + 1) / elapsed if elapsed > 0 else 0.0
                remaining = total - (i + 1)
                eta_sec = remaining / rate if rate > 0 else 0.0
                print(f"... {i+1}/{total} | {rate:.2f} it/s | ETA {eta_sec/60:.1f} min", flush=True)
            scores = self.evaluate_response(response)
            results.append({
                'input': item.get('input', ''),
                'response': response,
                'label': item.get('label', ''),
                'scores': scores
            })
            for dimension, score in scores.items():
                all_scores[dimension].append(score)
        averages = {k: (sum(v) / len(v) if v else 0.0) for k, v in all_scores.items()}
        return {'results': results, 'averages': averages, 'total_evaluated': len(results)}


def load_ecr_results(file_path: str) -> List[Dict[str, Any]]:
    results = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line.strip()))
    return results


def save_evaluation_results(results: Dict[str, Any], output_file: str):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def print_evaluation_summary(results: Dict[str, Any]):
    averages = results['averages']
    print("\n" + "="*80)
    print("LLM-BASED SUBJECTIVE EVALUATION RESULTS")
    print("="*80)
    print(f"Total responses evaluated: {results['total_evaluated']}")
    print("\nAverage Scores (0-9 scale):")
    print("-" * 50)
    print(f"Emotional Intensity (Emo Int):     {averages['emotional_intensity']:.3f}")
    print(f"Emotional Persuasiveness (Emo Pers): {averages['emotional_persuasiveness']:.3f}")
    print(f"Logic Persuasiveness (Log Pers):   {averages['logic_persuasiveness']:.3f}")
    print(f"Informativeness (Info):            {averages['informativeness']:.3f}")
    print(f"Lifelikeness (Life):               {averages['lifelikeness']:.3f}")
    print("-" * 50)
    overall_avg = sum(averages.values()) / len(averages)
    print(f"Overall Average:                   {overall_avg:.3f}")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="LLM-based subjective evaluation for ECR system")
    parser.add_argument("--input_file", default="save/redial_gen/emp_test.jsonl",
                       help="Path to ECR results file")
    parser.add_argument("--output_file", default="llm_evaluation_results.json",
                       help="Path to save evaluation results")
    parser.add_argument("--scorer_model", default="gpt-4-turbo",
                       help="OpenAI model for scoring (gpt-4-turbo, gpt-3.5-turbo)")
    parser.add_argument("--local_model", default=None,
                       help="Local model path or HF id for offline scoring (e.g., openai/gpt-oss-20b)")
    parser.add_argument("--load_in_8bit", action="store_true",
                       help="If set, attempt to load local model in 8-bit (requires bitsandbytes)")
    parser.add_argument("--api_key", default=None,
                       help="OpenAI API key (if using OpenAI models)")
    parser.add_argument("--sample_size", type=int, default=1000,
                       help="Number of responses to evaluate (0 for all)")
    parser.add_argument("--force_cpu", action="store_true",
                        help="Force loading local model on CPU to avoid GPU OOM (slower)")
    parser.add_argument("--progress_interval", type=int, default=50,
                        help="Print progress and ETA every N items")
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp32","bf16","fp16"],
                        help="Computation dtype for local model (CPU bf16 recommended for gpt-oss)")
    parser.add_argument("--random_seed", type=int, default=42,
                       help="Random seed for sampling")
    args = parser.parse_args()
    
    random.seed(args.random_seed)
    if not args.api_key:
        args.api_key = os.getenv("OPENAI_API_KEY")
    
    evaluator = LLMSubjectiveEvaluator(
        scorer_model=args.scorer_model,
        api_key=args.api_key,
        local_model=args.local_model,
        load_in_8bit=args.load_in_8bit,
        force_cpu=args.force_cpu,
        progress_interval=args.progress_interval,
        dtype=args.dtype,
    )

    print(f"Loading data from {args.input_file}...")
    data = load_ecr_results(args.input_file)
    print(f"Loaded {len(data)} responses")
    
    results = evaluator.evaluate_dataset(data, args.sample_size)
    save_evaluation_results(results, args.output_file)
    print(f"Results saved to {args.output_file}")
    print_evaluation_summary(results)
    
    print("\nExample Evaluations:")
    print("-" * 80)
    for i, result in enumerate(results['results'][:3]):
        print(f"\nExample {i+1}:")
        print(f"Response: {result['response'][:200]}...")
        s = result['scores']
        print(f"Scores: Emo Int={s['emotional_intensity']:.1f}, "
              f"Emo Pers={s['emotional_persuasiveness']:.1f}, "
              f"Log Pers={s['logic_persuasiveness']:.1f}, "
              f"Info={s['informativeness']:.1f}, "
              f"Life={s['lifelikeness']:.1f}")


if __name__ == "__main__":
    main() 
