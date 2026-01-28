import os
import zipfile
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


raw_dir = "./raw"                 
training_dir = "./training_dataset"    
training_dataset = os.path.join(training_dir, "train.parquet")
testing_dataset  = os.path.join(training_dir, "test.parquet")

BATCH_SIZE = 128
COMPRESSION = "zstd"

# build prompt
def build_prompt(row) -> str:
    q = str(row["Question"]).strip()
    a = str(row["Choice A"]).strip()
    b = str(row["Choice B"]).strip()
    c = str(row["Choice C"]).strip()
    d = str(row["Choice D"]).strip()
    return (
        "<image>\n"
        f"Question: {q}\n"
        f"{a}\n{b}\n{c}\n{d}\n"
        "Only answer the letter A, B, C or D\n"
    )

# read images into bytes
def read_images(zf, fig: str) -> bytes:

    fig = str(fig).strip().lstrip("./")
    for image in (fig, f"images/{fig}", f"images/images/{fig}"):
        try:
            return zf.read(image)
        except KeyError:
            pass

    base = os.path.basename(fig)
    for name in zf.namelist():
        if os.path.basename(name) == base:
            return zf.read(name)

    raise KeyError(f"Not found in zip: {fig}")

# write the data into parquet file : image, question and answer
def write_parquet(df: pd.DataFrame, zf: zipfile.ZipFile, path: str):
    if os.path.exists(path):
        os.remove(path)

    writer = None
    buf = []

    def flush():
        nonlocal writer, buf
        table = pa.Table.from_pylist(buf)  
        if writer is None:
            writer = pq.ParquetWriter(
                path,
                table.schema,
                compression=COMPRESSION,
                use_dictionary=True,
                write_statistics=True,
            )
        writer.write_table(table)
        buf = []

    for _, row in df.iterrows():
        fig = str(row["Figure_path"]).strip()
        img_bytes = read_images(zf, fig)

        buf.append({
            "image": img_bytes,
            "question": build_prompt(row),
            "answer": str(row["Answer_label"]).strip().upper(),  
        })

        if len(buf) >= BATCH_SIZE:
            flush()

    if buf:
        flush()

    if writer is not None:
        writer.close()

def main():
    train_csv = os.path.join(raw_dir, "train.csv")
    test_csv  = os.path.join(raw_dir, "test.csv")
    zip_path  = os.path.join(raw_dir, "images.zip")

    train_df = pd.read_csv(train_csv)
    test_df  = pd.read_csv(test_csv)

    with zipfile.ZipFile(zip_path, "r") as zf:

        write_parquet(train_df, zf, os.path.join(training_dir, "train.parquet"))
        write_parquet(test_df,  zf, os.path.join(training_dir, "test.parquet"))

    print("Done:", os.path.join(training_dir, "train.parquet"), os.path.join(training_dir, "test.parquet"))

if __name__ == "__main__":
    main()
