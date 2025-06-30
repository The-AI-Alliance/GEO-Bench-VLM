import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import json
from tqdm import tqdm
from collections import defaultdict


# # Set environment variables
# os.environ['TORCH_USE_CUDA_DSA'] = '1'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# print("TORCH_USE_CUDA_DSA:", os.getenv('TORCH_USE_CUDA_DSA'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


from transformers import AutoProcessor, LlavaForConditionalGeneration


model_id = "llava-hf/llava-1.5-7b-hf"
pth_m = "/Out_weights/llava-1.5-7b-hf"
model = LlavaForConditionalGeneration.from_pretrained(
    pth_m, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True,
    device_map=device
)

processor = AutoProcessor.from_pretrained(pth_m)


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
            image = Image.open(image_url).convert('RGB')
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
            for image, img_path, question, answer, question_type, question_number, ground_truth_option, options_list, task, question_id, cls_description, options in zip(
                batch['image'], batch['images'], batch['questions'], batch['answers'], batch['question_type'], batch['question_number'], batch['ground_truth_option'], batch['options_list'], batch['task'], batch['question_id'], batch['cls_description'], batch['options']
            ):
                try:
                    if question_type == "Multiple Choice Questions":
                        choices = "Options: " + options
                        prompt = f"For the given the Multiple Choice Question Answer below, analyze the question and answer strictly from one of the options below. Strictly answer the choice only. No additional text. Provide only the letter (A., B., C., D. or E.) corresponding to the correct answer for the multiple-choice question given. {cls_description}\n{question}\n{choices}"
                    else:
                        prompt = question
                    
                    conversation = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": f"{prompt}"},
                                {"type": "image"},
                            ],
                        },
                    ]
                    
                    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
                    inputs = processor(images=image, text=prompt, return_tensors='pt').to(device, torch.float16)
                    output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
                    response = processor.decode(output[0], skip_special_tokens=True)
                    
                    # print (response)
                    # print("NNN")
                    
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
    # print (df)
    dataset = MultimodalDataset(df, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    scores = evaluate(model, dataloader, processor, device)
    
    result_folder = os.path.join(folder_path, "Results-llava1pt5-countcls")
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
