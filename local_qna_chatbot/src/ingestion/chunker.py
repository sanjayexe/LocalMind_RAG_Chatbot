def chunk_text(text, size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), size):
        chunk = " ".join(words[i:i+size])
        chunks.append(chunk)
    return chunks

def process_records(records):
    dataset = []
    chunk_id = 1
    for r in records:
        chunks = chunk_text(r["text"])
        for c in chunks:
            dataset.append({"id": chunk_id, "text": c})
            chunk_id += 1
    return dataset
