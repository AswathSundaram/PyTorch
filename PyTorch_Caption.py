import warnings,logging
warnings.simplefilter('ignore')
logging.disable(logging.WARNING)

from transformers import pipeline

caption=pipeline('image-to-text')

