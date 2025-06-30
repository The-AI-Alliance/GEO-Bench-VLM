import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from transformers import AutoTokenizer
import pandas as pd
import os
from transformers import AutoModel, AutoTokenizer
from collections import defaultdict
import json
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print (device)

pth_m = "/Out_weights/InternVL2-8B"
model_id = "OpenGVLab/InternVL2-8B" # change
model = AutoModel.from_pretrained(
    pth_m,
    torch_dtype=torch.bfloat16,
    load_in_8bit=False,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True,
    device_map='cuda').eval()

tokenizer = AutoTokenizer.from_pretrained(pth_m, trust_remote_code=True, use_fast=False)

generation_config = dict(max_new_tokens=1024, do_sample=True)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size)
    image = transform(image).unsqueeze(0)
    return image

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
            # ima_path = image_url
            image = load_image(image_url).to(torch.bfloat16).cuda()
        except Exception as e:
            print(f"Error in loading image: {e}")
            print(f"Image URL: {image_url}")
        
        return {
            'image': image,
            'image_path': image_url,
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
    image = [item['image'] for item in batch]
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
        'image': image,
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
    

def evaluate(model, dataloader, device):
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
            for image, img_path, question, answer, question_type, question_number, ground_truth_option, options_list, task, question_id, cls_description, options in zip(
                batch['image'], batch['images'], batch['questions'], batch['answers'], batch['question_type'], batch['question_number'], batch['ground_truth_option'], batch['options_list'], batch['task'], batch['question_id'], batch['cls_description'], batch['options']
            ):
                try:
                    if question_type == "Multiple Choice Questions":
                        choices = "Options: " + options
                        prompt = f"For the given the Multiple Choice Question Answer below, analyze the question and answer strictly from one of the options below. Strictly answer the choice only. No additional text. Provide only the letter (A., B., C., D. or E.) corresponding to the correct answer for the multiple-choice question given. {cls_description}\n{question}\n{choices}"
                    else:
                        prompt = question
                        
                    # print (image.size)
                        
                    pixel_values = image#.unsqueeze(0)  # Add batch dimension
                    generation_config = dict(max_new_tokens=1024, do_sample=True)
                    response = model.chat(tokenizer, pixel_values, prompt, generation_config)
                    
                    # print (response)
                    
                    if "ASSISTANT:" in response:
                        response = response.split("ASSISTANT:")[-1].strip()
                    print (img_path)

                    # Extract only the first valid letter (A, B, C, D, or E)
                    valid_choices = {"A", "B", "C", "D", "E"}
                    predicted_answer = response[0] if response and response[0] in valid_choices else None
                    
                    # print (predicted_answer)

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

    # Create DataFrame from data_rows and prepare the dataloader
    df = pd.DataFrame(data_rows)
    dataset = MultimodalDataset(df, transform=None)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # Evaluate the model and save results
    scores = evaluate(model, dataloader, device)
    result_folder = os.path.join(folder_path, "Results-llavaInternVL-countcls")
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
