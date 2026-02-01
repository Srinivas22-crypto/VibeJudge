from models.transcriber import get_transcriber
import os

# Initialize transcriber
transcriber = get_transcriber()

# Test with a sample file (you'll need to provide this)
# For testing, you can:
# 1. Record a short voice memo (30 seconds)
# 2. Download a podcast clip from YouTube
# 3. Use text-to-speech to generate test audio

test_audio = "data/uploads/test_sample.mp3"  # Place your test file here

if os.path.exists(test_audio):
    print("Testing transcription...")
    
    # Preprocess
    processed_audio = transcriber.preprocess_audio(test_audio)
    
    # Transcribe
    result = transcriber.transcribe(processed_audio)
    
    # Print results
    print("\n" + "="*50)
    print("TRANSCRIPT:")
    print("="*50)
    print(result['text'])
    print("\n" + "="*50)
    
    # Print first 3 segments with timestamps
    print("TIMESTAMPED SEGMENTS:")
    for seg in result['segments'][:3]:
        timestamp = transcriber.format_timestamp(seg['start'])
        print(f"[{timestamp}] {seg['text']}")
    
    # Save
    output_path = "data/transcripts/test_transcript.json"
    transcriber.save_transcript(result, output_path)
    
    # Summary
    summary = transcriber.get_transcript_summary(result)
    print("\n" + "="*50)
    print("SUMMARY:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

else:
    print(f"Please place a test audio file at: {test_audio}")