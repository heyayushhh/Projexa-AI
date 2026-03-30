def run_inference(classifier, chunks, timestamps, batch_size=16):
    results = []

    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        batch_timestamps = timestamps[i:i + batch_size]

        predictions = classifier(batch_chunks)

        for pred, (start, end) in zip(predictions, batch_timestamps):
            results.append({
                "start": start,
                "end": end,
                "label": pred[0]["label"],
                "confidence": float(pred[0]["score"])
            })

    return results