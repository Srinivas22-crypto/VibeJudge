# utils/full_transcription_pipeline.py

from utils.audio_preprocessing import convert_to_wav_16k_mono
from utils.audio_chunking import split_audio_into_chunks
from utils.parallel_transcription import transcribe_chunks_parallel
from utils.transcript_merger import merge_chunk_transcripts, save_merged_transcript


def run_optimized_transcription_pipeline(input_audio_path: str):
    # Step 1: Preprocess audio
    processed_audio_path = convert_to_wav_16k_mono(input_audio_path)

    # Step 2: Chunk audio
    chunk_paths = split_audio_into_chunks(processed_audio_path, chunk_length_ms=5 * 60 * 1000)

    # Step 3: Parallel transcription
    chunk_results = transcribe_chunks_parallel(chunk_paths, model_size="small", max_workers=2)

    # Step 4: Merge transcripts
    merged_transcript = merge_chunk_transcripts(chunk_results, chunk_duration_sec=300)

    # Step 5: Save final transcript
    save_merged_transcript(merged_transcript, "data/transcripts/final_transcript.json")

    return merged_transcript
