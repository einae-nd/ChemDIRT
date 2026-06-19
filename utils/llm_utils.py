from transformers import AutoModelForCausalLM, AutoModelForCausalLM, AutoTokenizer, GenerationConfig, BitsAndBytesConfig
import torch
import re

import wolframalpha
import anthropic
from openai import OpenAI

from utils.data_utils import *

anthropic_client = anthropic.Anthropic(
    api_key=os.environ["ANTRHOPIC_API_KEY"],
)

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

system_prompt = "You are a chemistry expert. You will be given a chemistry-related question, and you should provide an accurate answer based on your knowledge of chemistry."

hf_key = os.environ["HUGGINGFACE_API_KEY"] # add your own

client_wolfram = wolframalpha.Client(os.environ["WOLFRAMALPHA_KEY"])
BASE_URL = "http://api.wolframalpha.com/v2/query"

def get_task_prompt(task_name):
    if task_name in ["carbon_count", "ring_count", "esol", "lipo", "freesolv", "o2", "mol_weight", "coeff", "unit_calc", "yield"]:
        return " \nGiven the provided question, return ONLY the relevant answer. ONLY return a number, no units."
    elif task_name in ["functional_group", "bond_type", "name_equiv", "bace", "bbbp", "clintox", "hiv"]:
        return " \nGiven the provided question, return ONLY the relevant answer. ONLY a yes/no answer."
    elif task_name in ["s2i", "p2i"]:
        return " \nGiven the provided question, return ONLY the IUPAC name."
    elif task_name in ["i2s", "f2s", "p2s", "molecule_design"]:
        return " \nGiven the provided question, return ONLY the SMILES string."
    elif task_name in ["s2f"]:
        return " \nGiven the provided question, return ONLY the molecular formula."
    elif task_name in ['molecule_caption']:
        return " \nGiven the provided question, return ONLY the relevant design or caption."
    elif task_name in ["chemical_sim", "property_comp"]:
        return " \nGiven the provided question, return ONLY the relevant answer. ONLY provide 1 or 2. 1 for the first option, 2 for the second option."
    elif task_name in ["retro_smiles", "forward_smiles"]:
        return " \nGiven the provided question, return ONLY the relevant answer. Return ONLY the SMILES string"
    elif task_name in ["retro_formula", "forward_formula"]:
        return " \nGiven the provided question, return ONLY the relevant answer. Return ONLY the molecular formula."
    elif task_name in ["tfq"]:
        return " \nGiven the proivded question, return ONLY one of the following answers: true, false, or maybe."
    elif task_name in ["rag_qa", "open_reponse"]:
        return ""
    else:
        return ""


def query_llm(prompts, task_prompt, model_name='gpt-4o', image_paths=None, tokenizer=None, model=None, config=None, max_tokens=256):
    if image_paths is None:
        image_paths = [None] * len(prompts)
    if model_name in ['gpt-4o-mini', 'gpt-4.1-nano', 'gpt-5.4-nano']:
        #prompts = [prompt + task_prompt for prompt in prompts]
        #for prompt in prompts:
        responses = []
        for prompt, image_path in zip(prompts, image_paths):
            prompt = prompt + task_prompt
            try:
                if image_path is None:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                else:
                    image_b64 = encode_image(image_path)

                    messages = [
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{image_b64}"
                                    }
                                }
                            ]
                        }
                    ]

                reply = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                )
                #response = reply.choices[0].message.content
                responses.append(reply.choices[0].message.content)
                #return response
            except Exception as e:
                print(f"Error querying GPT: {e}")
                responses.append(None)
                #return None  
        return responses
    elif model_name in ["claude"]:
        #prompts = [prompt + task_prompt for prompt in prompts]
        #for prompt in prompts:
        responses = []
        for prompt, image_path in zip(prompts, image_paths):
            prompt = prompt + task_prompt
            try:
                content = []
                if image_path is not None:
                    image_b64 = encode_image(image_path)
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_b64
                        }
                    })
                content.append({
                    "type": "text",
                    "text": prompt
                })
                
                reply = anthropic_client.messages.create(
                    model="claude-sonnet-4-5-20250929",
                    max_tokens=256,
                    temperature=1,
                    system=system_prompt,
                    messages=[
                        {
                            "role": "user",
                            "content": content
                        }
                    ]
                )
                #response = 
                responses.append(reply.content[0].text)
            except Exception as e:
                print(f"Error querying Claude: {e}")
                responses.append(None)   
            """try:
                reply = client.chat.completions.create(
                    model='gpt-4.1-nano',
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": "Please extract ONLY the answer from the following response: " + response}
                    ],
                    temperature=0.0,
                )
                response = reply.choices[0].message.content
                #return response
                responses.append(response)   
            except Exception as e:
                print(f"Error querying GPT: {e}")
                #return None
                responses.append(None) """
        return responses
    elif model_name in ['llama3', 'llama2', 'gemma3', 'gemma2', 'mistral', 'qwen3', 'qwen2', 'falcon', 'chemr', 'chemllm', 'chemdfm', 'deepseek-r1', 'ether0', 'general-reasoner', 'galactica']:
        prompts = [f"{system_prompt}\n{prompt}" + task_prompt for prompt in prompts]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
            
        model.eval()
        
        with torch.inference_mode():
            if model_name == "galactica":
                outputs = model.generate(inputs.input_ids)
            else:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    use_cache=False,
                    #use_cache=True,
                )

            answers = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            responses = []
            for answer in answers:
                try:
                    reply = client.chat.completions.create(
                        model='gpt-4.1-nano',
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": "Please extract ONLY the answer from the following response: " + answer}
                        ],
                    )
                    response = reply.choices[0].message.content
                    responses.append(response)
                except Exception as e:
                    print(f"Error querying GPT: {e}")
                    responses.append(None)
            return responses
        

def get_llm(name):
    
    generation_config = None
    cache_dir = ""
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    if name == 'chemllm':
        tokenizer = AutoTokenizer.from_pretrained("AI4Chem/ChemLLM-7B-Chat-1_5-DPO", cache_dir=cache_dir, trust_remote_code=True, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained("AI4Chem/ChemLLM-7B-Chat-1_5-DPO", cache_dir=cache_dir, trust_remote_code=True, quantization_config=bnb_config, device_map="auto")
    elif name == 'chemdfm':
        tokenizer = AutoTokenizer.from_pretrained("OpenDFM/ChemDFM-v1.5-8B", cache_dir=cache_dir, trust_remote_code=True, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained("OpenDFM/ChemDFM-v1.5-8B", cache_dir=cache_dir, trust_remote_code=True, quantization_config=bnb_config, device_map="auto")
        generation_config = GenerationConfig(
            do_sample=True,
            top_k=20,
            max_new_tokens=256,
            repetition_penalty=1.05,
            eos_token_id=tokenizer.eos_token_id
        )
    elif name == 'general-reasoner':
        tokenizer = AutoTokenizer.from_pretrained("TIGER-Lab/General-Reasoner-Qwen3-4B", cache_dir=cache_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained("TIGER-Lab/General-Reasoner-Qwen3-4B", cache_dir=cache_dir, trust_remote_code=True, quantization_config=bnb_config, device_map="auto")
    elif name == 'ether0':
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        tokenizer = AutoTokenizer.from_pretrained("futurehouse/ether0", cache_dir=cache_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained("futurehouse/ether0", cache_dir=cache_dir, trust_remote_code=True, quantization_config=bnb_config, device_map="auto")
    elif name == 'deepseek-r1':
        tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", cache_dir=cache_dir, trust_remote_code=True, token=hf_key)
        model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", cache_dir=cache_dir, trust_remote_code=True, quantization_config=bnb_config, device_map="auto", token=hf_key)
    elif name == 'llama3':
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", cache_dir=cache_dir, trust_remote_code=True, token=hf_key)
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", cache_dir=cache_dir, trust_remote_code=True, quantization_config=bnb_config, device_map="auto", token=hf_key)
    elif name == 'llama2':
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir=cache_dir, trust_remote_code=True, token=hf_key)
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir=cache_dir, trust_remote_code=True, quantization_config=bnb_config, device_map="auto", token=hf_key)
    elif name == 'gemma3':
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it", cache_dir=cache_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained("google/gemma-3-4b-it", cache_dir=cache_dir, trust_remote_code=True, quantization_config=bnb_config, device_map="auto")
    elif name == 'gemma2':
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b", cache_dir=cache_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained("google/gemma-2-9b", cache_dir=cache_dir, trust_remote_code=True, quantization_config=bnb_config, device_map="auto")
    elif name == 'mistral':
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", cache_dir=cache_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", cache_dir=cache_dir, trust_remote_code=True, quantization_config=bnb_config, device_map="auto")
    elif name == 'qwen3':
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", cache_dir=cache_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", cache_dir=cache_dir, trust_remote_code=True, quantization_config=bnb_config, device_map="auto")
    elif name == 'qwen2':
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct", cache_dir=cache_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-7B-Instruct", cache_dir=cache_dir, trust_remote_code=True, quantization_config=bnb_config, device_map="auto")
    elif name == 'falcon':
        tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct", cache_dir=cache_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b-instruct", cache_dir=cache_dir, trust_remote_code=True, quantization_config=bnb_config, device_map="auto")
    elif name == 'galactica':
        tokenizer = AutoTokenizer.from_pretrained("facebook/galactica-6.7b", cache_dir=cache_dir, trust_remote_code=True, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained("facebook/galactica-6.7b", cache_dir=cache_dir, trust_remote_code=True, quantization_config=bnb_config, device_map="auto")
        tokenizer.add_special_tokens({'eos_token': '[EOS]'})
        model.resize_token_embeddings(len(tokenizer))
    else:
        model = None
        tokenizer = None

    if tokenizer is not None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, generation_config

def extract_answer_letter(response: str) -> str:
    """
    Extracts a single letter A-D from the model's response.
    """
    match = re.search(r"\b([A-D])\b", response.strip().upper())
    return match.group(1) if match else None

def format_mc_prompt(example: dict) -> str:
    """
    Formats a multiple-choice question for prompting a language model.

    Args:
        example (dict): A dictionary with keys 'question', 'A', 'B', 'C', 'D', and optionally 'answer'.

    Returns:
        str: A formatted string suitable for use as a prompt.
    """
    question = example.get("question", "").strip()
    choices = []
    for option in ["A", "B", "C", "D"]:
        if option in example:
            choices.append(f"{option}: {example[option].strip()}")
    
    formatted = f"{question}\n\nChoices:\n" + "\n".join(choices)
    formatted += (
        "\n\nWhich option best satisfies the condition? "
        "Respond with ONLY the letter A, B, C, or D. Do not include any explanation."
    )
    return formatted

def parse_species(mixture_str):
    """Extracts tuples of (quantity, molecule) from a string like '{1}Na+.{2}H2O'."""
    matches = re.findall(r"\{(\d+)\}([^\.]+)", mixture_str)
    return [(int(qty), mol) for qty, mol in matches]

def format_side(parsed_species):
    """
    Format the parsed species into a human-readable chemical side.
    If coeff = 1, omit the number.
    """
    return ' + '.join([f"{coeff if coeff > 1 else ''}{formula}" for coeff, formula in parsed_species])

def build_equation(reactants_str, products_str):
    reactants = format_side(parse_species(reactants_str))
    products = format_side(parse_species(products_str))
    return f"{reactants} → {products}"

def dict_to_side(d):
    """Convert OrderedDict of {formula: coeff} to text side of equation"""
    parts = []
    for formula, coeff in d.items():
        if coeff == 1:
            parts.append(f"{formula}")
        else:
            parts.append(f"{coeff} {formula}")
    return " + ".join(parts)

def equation_to_text(r_bal, p_bal):
    """Turn r_bal and p_bal into a balanced equation string"""
    return f"{dict_to_side(r_bal)} → {dict_to_side(p_bal)}"