import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM,Qwen2VLForConditionalGeneration
import pandas as pd
import os
import json
from tqdm import tqdm
from qwen_vl_utils import process_vision_info
import time
from collections import defaultdict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print (device)

from transformers import AutoTokenizer, AutoProcessor

# tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b")
# model = AutoModelForCausalLM.from_pretrained("google/gemma-7b",device_map='auto')

pth_m = '/Out_weights/Qwen2-VL-7B-Instruct'

model_id = "Qwen/Qwen2-VL-7B-Instruct"
# tokenizer = AutoTokenizer.from_pretrained(pth_m, trust_remote_code=True)
# model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, device_map="cuda:2", trust_remote_code=True, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16).eval()

model = Qwen2VLForConditionalGeneration.from_pretrained(pth_m, device_map="cuda", trust_remote_code=True, torch_dtype=torch.bfloat16).eval()
# model = "NULL"

from concurrent.futures import ThreadPoolExecutor, as_completed

class MultimodalDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_url = self.dataframe.iloc[idx]['Image']
        question = self.dataframe.iloc[idx]['Tr_Question']
        answer = self.dataframe.iloc[idx]['Tr_Answer']
        question_type = self.dataframe.iloc[idx]['Question_Type']
        question_number = self.dataframe.iloc[idx]['Question_Number']
        ground_truth_option = self.dataframe.iloc[idx]['Ground_Truth_Option']
        options_list = self.dataframe.iloc[idx]['Options_List']
        task = self.dataframe.iloc[idx]['Task']
        question_id = self.dataframe.iloc[idx]['Question_ID']
        cls_description = self.dataframe.iloc[idx]['Cls_Description']
        options = self.dataframe.iloc[idx]['Options']

        try:
            ima_path = image_url
        except Exception as e:
            print(f"Error in loading image: {e}")
            print(f"Image URL: {image_url}")
        
        return {
            'image_path': ima_path,
            'question': question,
            'answer': answer,
            'question_type': question_type,
            'question_number': question_number,
            'ground_truth_option': ground_truth_option,
            'options_list': options_list,
            'task': task,
            'question_id': question_id,
            'cls_description': cls_description,
            'options': options
        }


# Load data from XLSX file
def load_data(file_path):
    df = pd.read_excel(file_path)
    return df

# Define the transforms for the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize all images to 224x224
    transforms.ToTensor(),  # Convert to Tensor, values between 0 and 1
])

def collate_fn(batch):
    images = [item['image_path'] for item in batch]
    questions = [item['question'] for item in batch]
    answers = [item['answer'] for item in batch]
    question_types = [item['question_type'] for item in batch]
    question_numbers = [item['question_number'] for item in batch]
    ground_truth_options = [item['ground_truth_option'] for item in batch]
    options_lists = [item['options_list'] for item in batch]
    tasks = [item['task'] for item in batch]
    question_ids = [item['question_id'] for item in batch]
    cls_descriptions = [item['cls_description'] for item in batch]
    options = [item['options'] for item in batch]

    return {
        'images': images,
        'questions': questions,
        'answers': answers,
        'question_type': question_types,
        'question_number': question_numbers,
        'ground_truth_option': ground_truth_options,
        'options_list': options_lists,
        'task': tasks,
        'question_id': question_ids,
        'cls_description': cls_descriptions,
        'options': options
    }

def evaluate(model, dataloader, processor, device):
    model.eval()
    results = []

    with torch.no_grad():
        results_dict = defaultdict(lambda: {
            "predicted_answers": [],
            "ground_truth": None,
            "questions": [],
            # "choices": None,
            "name_images": [],
            "ground_truth_option": None,
            "options_list": None,
            "task": None,
            "question_id": None,
            "cls_description": None,
            "options": None
        })

        for batch in dataloader:
            for img_path, question, answer, question_type, question_number, ground_truth_option, options_list, task, question_id, cls_description, options in zip(
                batch['images'], batch['questions'], batch['answers'], batch['question_type'], batch['question_number'], batch['ground_truth_option'], batch['options_list'], batch['task'], batch['question_id'], batch['cls_description'], batch['options']
            ):
                try:
                    if question_type == "Multiple Choice Questions":
                        choices = "Options: " + options
                        prompt = f"For the given the Multiple Choice Question Answer below, analyze the question and answer strictly from one of the options below. Strictly answer the choice only. No additional text. Provide only the letter (A., B., C., D. or E.) corresponding to the correct answer for the multiple-choice question given. {cls_description}\n{question}\n{choices}"
                    else:
                        prompt = question

                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": f"file://{img_path}"},
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ]
                    
                    print (img_path)

                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    image_inputs, video_inputs = process_vision_info(messages)
                    inputs = processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    ).to(device)

                    generated_ids = model.generate(**inputs, max_new_tokens=128)
                    predicted_answer = processor.batch_decode(
                        [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )

                    key = question_number
                    results_dict[key]["predicted_answers"].extend(predicted_answer)
                    results_dict[key]["questions"].append(question)
                    results_dict[key]["ground_truth"] = answer
                    # results_dict[key]["choices"] = choices
                    results_dict[key]["name_images"].append(img_path)
                    results_dict[key]["ground_truth_option"] = ground_truth_option
                    results_dict[key]["options_list"] = options_list
                    results_dict[key]["task"] = task
                    results_dict[key]["question_id"] = question_id
                    results_dict[key]["cls_description"] = cls_description
                    results_dict[key]["options"] = options
                    
                    # print(results_dict)
                    # print(results_dict.values())

                except Exception as e:
                    print(f"Error in prediction: {e}")
                    print(f"Question: {question}")
            
        results.extend(results_dict.values())
    
    return results


def evaluate_folder(folder_path):
    qa_file_path = None
    for filename in ["qa.json"]:
        potential_path = os.path.join(folder_path, "Single", filename)
        if os.path.exists(potential_path):
            qa_file_path = potential_path
            break
    
    if qa_file_path is None:
        print(f"No matching qa file found in {folder_path}. Skipping.")
        return
    
    with open(qa_file_path, 'r') as file:
        data = json.load(file)
    
    mainp = folder_path
    data_rows = []
    
    for i, question in enumerate(data):  # Directly iterate over JSON list
        image_p = os.path.join(mainp, question.get("image_path", ""))
        ground_truth = question.get("ground_truth", "")
        ground_truth_option = question.get("ground_truth_option", "")
        options_list = question.get("options_list", [])
        task = question.get("task", "")
        question_id = question.get("question_id", "")
        cls_description = question.get("cls_description", "")
        options_str = question.get("options", "")
        
        for prompt in question.get("prompts", []):
            data_rows.append({
                "Question_Number": i,
                "Category": task,
                "Image": image_p,
                "Question_Type": "Multiple Choice Questions",
                "Tr_Question": prompt,
                "Tr_Answer": ground_truth,
                "Ground_Truth_Option": ground_truth_option,
                "Options_List": options_list,
                "Options": options_str,
                "Task": task,
                "Question_ID": question_id,
                "Cls_Description": cls_description
            })
    
    df = pd.DataFrame(data_rows)
    dataset = MultimodalDataset(df, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    processor = AutoProcessor.from_pretrained(pth_m)
    # processor = "NULL"
    scores = evaluate(model, dataloader, processor, device)
    # xx
    result_folder = os.path.join(folder_path, "Results-qwen2-countcls")
    os.makedirs(result_folder, exist_ok=True)
    
    result_file = os.path.join(result_folder, f"evaluation_results_{os.path.basename(folder_path)}.json")
    result_filet = os.path.join(result_folder, f"evaluation_results_{os.path.basename(folder_path)}.txt")
    
    try:
        with open(result_file, "w") as f:
            json.dump(scores, f, indent=4, default=str)
    except Exception as e:
        print(f"Error in saving results: {e}")
        with open(result_filet, "w") as f:
            f.write(str(scores))
    
    print(f"Results saved successfully for folder {folder_path}.")
    

# Main function to iterate over folders
def main(base_folder_path):
    # for folder in os.listdir(base_folder_path):
    # folder_path = os.path.join(base_folder_path, folder)
    folder_path = base_folder_path
    print(folder_path)
    if os.path.isdir(folder_path):
        evaluate_folder(folder_path)

if __name__ == "__main__":
    main("/datasets/GEOBench-VLM")
