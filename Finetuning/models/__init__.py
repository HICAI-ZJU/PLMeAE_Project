import os
from fairseq.models import import_models

models_dir = os.path.dirname(__file__)
import_models(models_dir, "ma-tuning.models")