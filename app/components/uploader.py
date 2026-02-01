import streamlit as st
from pathlib import Path
import uuid
import subprocess
from typing import Optional, Tuple
from config.settings import (
    UPLOAD_DIR, MAX_FILE_SIZE_MB, 
    MAX_DURATION_SECONDS, ALLOWED_AUDIO_FORMATS
)

def validate_audio_file(uploaded_file) -> Tuple[bool, Optional[str]]:
    """
    Validate uploaded audio file
    
    Args:
        uploaded_file: Streamlit UploadedFile object
    
    Returns:
        (is_valid, error_message)
    """
    # Check file exists
    if uploaded_file is None:
        return False, "No file uploaded"
    
    # Check file extension
    file_ext = uploaded_file.name.split('.')[-1].lower()
    if file_ext not in ALLOWED_AUDIO_FORMATS:
        return False, f"Invalid format. Allowed: {', '.join(ALLOWED_AUDIO_FORMATS)}"
    
    # Check file size
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        return False, f"File too large ({file_size_mb:.1f}MB). Max: {MAX_FILE_SIZE_MB}MB"
    
    return True, None


def get_audio_duration(file_bytes, file_ext: str) -> Optional[float]:
    """
    Get duration of audio file using ffprobe
    
    Args:
        file_bytes: Audio file bytes
        file_ext: File extension
    
    Returns:
        Duration in seconds or None if error
    """
    try:
        # Save temporarily
        temp_path = UPLOAD_DIR / f"temp_{uuid.uuid4()}.{file_ext}"
        with open(temp_path, 'wb') as f:
            f.write(file_bytes)
        
        # Get duration using ffprobe directly
        cmd = [
            "ffprobe", 
            "-v", "error", 
            "-show_entries", "format=duration", 
            "-of", "default=noprint_wrappers=1:nokey=1", 
            str(temp_path)
        ]
        
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        duration = float(result.stdout.strip())
        
        # Clean up
        temp_path.unlink()
        
        return duration
    
    except subprocess.CalledProcessError:
        # If ffprobe fails, we might just proceed without duration or log warning
        # st.warning("Could not determine duration via ffprobe.")
        if temp_path.exists():
            temp_path.unlink()
        return None
    except Exception as e:
        st.warning(f"Could not determine audio duration: {e}")
        if 'temp_path' in locals() and temp_path.exists():
            temp_path.unlink()
        return None


def save_uploaded_file(uploaded_file) -> Tuple[str, str, float]:
    """
    Save uploaded file to disk
    
    Args:
        uploaded_file: Streamlit UploadedFile object
    
    Returns:
        (podcast_id, file_path, duration)
    """
    # Generate unique ID
    podcast_id = str(uuid.uuid4())
    
    # Get file extension
    file_ext = uploaded_file.name.split('.')[-1].lower()
    
    # Create filename
    filename = f"podcast_{podcast_id}.{file_ext}"
    file_path = UPLOAD_DIR / filename
    
    # Read file bytes
    file_bytes = uploaded_file.read()
    
    # Get duration before saving
    duration = get_audio_duration(file_bytes, file_ext)
    
    # Validate duration
    if duration and duration > MAX_DURATION_SECONDS:
        raise ValueError(
            f"Audio too long ({duration/60:.1f} min). "
            f"Max: {MAX_DURATION_SECONDS/60:.0f} min"
        )
    
    # Save file
    with open(file_path, 'wb') as f:
        f.write(file_bytes)
    
    return podcast_id, str(file_path), duration


def render_upload_section():
    """
    Render the file upload section
    
    Returns:
        (uploaded_file, podcast_id, file_path, duration) if file uploaded, else None
    """
    st.header("📤 Upload Podcast")
    
    # Upload widget
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=ALLOWED_AUDIO_FORMATS,
        help=f"Max size: {MAX_FILE_SIZE_MB}MB | Max duration: {MAX_DURATION_SECONDS/60:.0f} min"
    )
    
    if uploaded_file:
        # Display file info
        file_size_mb = uploaded_file.size / (1024 * 1024)
        st.info(f"📁 **{uploaded_file.name}** ({file_size_mb:.2f} MB)")
        
        # Validate
        is_valid, error_msg = validate_audio_file(uploaded_file)
        
        if not is_valid:
            st.error(f"❌ {error_msg}")
            return None
        
        # Save file button
        if st.button("✅ Process This Podcast", type="primary"):
            with st.spinner("Saving file..."):
                try:
                    podcast_id, file_path, duration = save_uploaded_file(uploaded_file)
                    
                    st.success("✓ File uploaded successfully!")
                    
                    # Display details
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Podcast ID", podcast_id[:8] + "...")
                    with col2:
                        st.metric("File Size", f"{file_size_mb:.2f} MB")
                    with col3:
                        if duration:
                            st.metric("Duration", f"{duration/60:.1f} min")
                    
                    return uploaded_file, podcast_id, file_path, duration
                
                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    return None
    
    return None