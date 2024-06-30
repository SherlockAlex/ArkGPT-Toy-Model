import pyarrow as pa
with open("./json-train.arrow",'rb') as f:
    r = pa.ipc.RecordBatchStreamReader(f)
    df = r.read_pandas()
    df = df['completion']

print(df)