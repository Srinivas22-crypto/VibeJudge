from concurrent.futures import ProcessPoolExecutor, as_completed
from models.transcriber import FastTranscriber


def transcribe_single_chunk(chunk_path: str, model_size="small"):
    transcriber = FastTranscriber(model_size=model_size, device="cpu", compute_type="int8")
    result = transcriber.transcribe_chunk(chunk_path)
    return {
        "chunk_path": chunk_path,
        "result": result
    }


def transcribe_chunks_parallel(chunk_paths: list, model_size="small", max_workers=2):
    """
    Transcribe chunks in parallel.
    """
    results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(transcribe_single_chunk, chunk_path, model_size)
            for chunk_path in chunk_paths
        ]

        for future in as_completed(futures):
            results.append(future.result())

    # Sort results based on chunk file name to preserve order
    results = sorted(results, key=lambda x: x["chunk_path"])
    return results
