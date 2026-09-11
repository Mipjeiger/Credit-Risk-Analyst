import os
import json
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import joblib
from dotenv import load_dotenv

from langchain_core.documents import Document

"""Implementing LLM RAG loader exploration (on notebook) to production"""