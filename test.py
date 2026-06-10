from PIL import Image
from surya.inference import SuryaInferenceManager
from surya.recognition import RecognitionPredictor

manager = SuryaInferenceManager()
rec = RecognitionPredictor(manager)
print("Surya loaded OK")