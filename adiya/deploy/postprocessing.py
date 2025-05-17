def reconstruct_text(predictions, boundaries):
    results = []
    for text, bbox in zip(predictions, boundaries):
        results.append({
            "text": text,
            "bbox": bbox
        })

    return {
        "results": results
    }