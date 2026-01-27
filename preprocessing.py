import os
import zipfile
import json
import pandas as pd

def unzip_images(images_path, images_dir):
    if not os.path.exists(images_path):
        raise FileNotFoundError(f"NO SUCH ZIP FILE: {images_path}")
    with zipfile.ZipFile(images_path, "r") as zf:
        zf.extractall(images_dir)

def build_prompt(question, choice_A, choice_B, choice_C,choice_D):
        return(
            "<image>\n"
            f"Question: {str(question).strip()}\n"
            f"{str(choice_A).strip()}\n"
            f"{str(choice_B).strip()}\n"
            f"{str(choice_C).strip()}\n"
            f"{str(choice_D).strip()}\n"
            "Only answer the letter A, B, C or D\n"
        )

#process the data to prepare for the training in llava
def process_llava_data(data, type_of_data):
    # Column mapping
    colname_fig = "Figure_path"
    colname_que = "Question"
    colname_ans = "Answer"
    colname_choiceA = "Choice A"
    colname_choiceB = "Choice B"
    colname_choiceC = "Choice C"
    colname_choiceD = "Choice D"
    colname_ans_label = "Answer_label"

    json_data_record = []
    for i, row in data.iterrows():
        figure = str(row[colname_fig]).strip()

        que = row[colname_que]
        choice_a = row[colname_choiceA]
        choice_b = row[colname_choiceB]
        choice_c = row[colname_choiceC]
        choice_d = row[colname_choiceD]

        ans_label = str(row[colname_ans_label]).strip().upper()

        json_data = {
            "id": f"pmcvqa_{type_of_data}_{figure}_{i}",
            "image": f"images/images/{figure}",
            "conversations": [
                {
                    "from": "human",
                    "value": build_prompt(que, choice_a, choice_b, choice_c, choice_d)
                },
                {
                    "from": "gpt",
                    "value": ans_label
                }
            ]
        }
        json_data_record.append(json_data)
    return json_data_record

#save the files
def save_json_data(path, json_data_record):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as json_file:
        json.dump(json_data_record, json_file, indent=2)

def preprocessing():
    rawdata_dir = "./raw" #need to be change
    # Create the folders for training data
    trainingdata_dir = "./training_dataset"
    images_dir = os.path.join(trainingdata_dir,"images")
    if not os.path.exists(trainingdata_dir):
        os.makedirs(trainingdata_dir)
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)


    trainingdata_path = os.path.join(rawdata_dir,"train.csv")
    testdata_path = os.path.join(rawdata_dir,"test.csv")
    images_path = os.path.join(rawdata_dir,"images.zip")
        
    # this is to check whether there has some unziped images in the folder
    if len(os.listdir(images_dir)) == 0:
        unzip_images(images_path, images_dir)
    else:
        print("Images_dir is not empty")



    train_data_frame = pd.read_csv(trainingdata_path)
    test_data_frame = pd.read_csv(testdata_path)

    train_data_record = process_llava_data(train_data_frame, "train")
    train_json_path = os.path.join(trainingdata_dir, "train", "train_dataset.json")
    save_json_data(train_json_path, train_data_record)

    test_data_record = process_llava_data(test_data_frame, "test")
    test_json_path = os.path.join(trainingdata_dir, "test", "test_dataset.json")
    save_json_data(test_json_path, test_data_record)

    print("\nDone.")
    print("Train JSON:", os.path.join(trainingdata_dir, "train", "train_dataset.json"))
    print("Test JSON:", os.path.join(trainingdata_dir, "test", "test_dataset.json"))
    print("Images folder:", images_dir)

    print("\nLLaVA training paths (typical):")
    print("  --data_path ./training_dataset/train/train_dataset.json")
    print("  --image_folder ./training_dataset")

    # check first sample's image exists
    with open(train_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_rel = data[0]["image"]
    img_abs = os.path.join(trainingdata_dir, img_rel)

    print("First image rel:", img_rel)
    print("First image abs:", img_abs)
    print("Exists:", os.path.exists(img_abs))

    # show first prompt + label
    print("\nPrompt preview:\n", data[0]["conversations"][0]["value"][:400])
    print("\nAnswer:", data[0]["conversations"][1]["value"])

# run everything
def main():
    preprocessing()
