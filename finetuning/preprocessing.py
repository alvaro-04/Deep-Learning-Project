import os
import zipfile
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import datasets
from datasets import load_dataset, Image 
#from datasets import load_from_disk

raw_dir = "./raw"                 
training_dir = "./training_dataset"    
training_dataset = os.path.join(training_dir, "train.parquet")
testing_dataset  = os.path.join(training_dir, "test.parquet")


# build prompt 
# idea from: https://dev.to/nagasuresh_dondapati_d5df/15-prompting-techniques-every-developer-should-know-for-code-generation-1go2
def build_prompt(row):
    q = str(row["Question"]).strip()
    a = str(row["Choice A"]).strip()
    b = str(row["Choice B"]).strip()
    c = str(row["Choice C"]).strip()
    d = str(row["Choice D"]).strip()
    return (
        # "<image>\n"
        f"Question: {q}\n"
        f"Options: {a}\n{b}\n{c}\n{d}\n"
        # "Only with the letter A, B, C or D\n" (instruction added in finetuning chat template)
    )

# read images into bytes
def read_images(zf, fig):
    fig = str(fig).strip()
    if fig.startswith("./"):
        fig = fig[2:]
    return zf.read("images/"+fig)

# write the data into parquet file : image, question and answer
# idea from: https://www.quora.com/How-do-I-convert-CSV-to-parquet-using-Python-and-without-using-Spark
def write_parquet(csv_path, zf, parquet_path):
    if os.path.exists(parquet_path):
        os.remove(parquet_path)

    writer = None

    for chunk in pd.read_csv(csv_path, chunksize = 1000):
        rows = []
        for i in range(len(chunk)):
            row = chunk.iloc[i]
            fig = str(row["Figure_path"]).strip()
            img_bytes = read_images(zf, fig)

            rows.append({
                "image": {"bytes": img_bytes},
                "question": build_prompt(row),
                "answer": str(row["Answer_label"]).strip().upper(),
            })

        table = pa.Table.from_pylist(rows)

        if writer is None:
            writer = pq.ParquetWriter(parquet_path, table.schema, compression="snappy")

        writer.write_table(table)

    if writer:
        writer.close()

def main():
    os.environ["HF_DATASETS_CACHE"] = os.path.join(os.environ.get("TMPDIR", "./"), "hf_cache")
    os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)
    train_csv = os.path.join(raw_dir, "train.csv")
    test_csv  = os.path.join(raw_dir, "test.csv")
    zip_path  = os.path.join(raw_dir, "images.zip")

    with zipfile.ZipFile(zip_path, "r") as zf:
        write_parquet(train_csv, zf, training_dataset)
        write_parquet(test_csv, zf, testing_dataset)

# idea from: https://huggingface.co/docs/datasets/en/about_mapstyle_vs_iterable
    ds_train = load_dataset("parquet", data_files=training_dataset, split="train")
    ds_train = ds_train.cast_column("image", Image(decode=True))
    # ds_train.save_to_disk(os.path.join(training_dir, "image_train"))

    ds_test = load_dataset("parquet", data_files=testing_dataset, split="train")
    ds_test = ds_test.cast_column("image", Image(decode=True))
    # ds_test.save_to_disk(os.path.join(training_dir, "image_test"))
    print(f"Preprocessing finished. Parquet files in {training_dir}")

if __name__ == "__main__":
    main()
