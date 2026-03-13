import spacy
import logging
from spacy.util import is_package

logger = logging.getLogger(__name__)

def load_spacy_model(model_name: str = "en_core_web_sm", **kwargs):
    """
    Safely load a spaCy model, downloading it if it's not installed.
    
    Args:
        model_name: Name of the spaCy model to load.
        **kwargs: Additional arguments to pass to spacy.load.
        
    Returns:
        The loaded spaCy model.
    """
    try:
        # Check if the model is installed
        if not is_package(model_name):
            logger.info(f"spaCy model '{model_name}' not found. Downloading...")
            spacy.cli.download(model_name)
            logger.info(f"✓ spaCy model '{model_name}' downloaded successfully.")
        
        # Load the model
        nlp = spacy.load(model_name, **kwargs)
        logger.info(f"✓ spaCy model '{model_name}' loaded successfully.")
        return nlp
        
    except Exception as e:
        logger.error(f"Failed to load or download spaCy model '{model_name}': {e}")
        # If it still fails, try one more time as a fallback
        try:
            import subprocess
            import sys
            logger.info(f"Attempting fallback download for '{model_name}'...")
            subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])
            nlp = spacy.load(model_name, **kwargs)
            return nlp
        except Exception as fallback_err:
            logger.error(f"Fallback download also failed: {fallback_err}")
            raise
