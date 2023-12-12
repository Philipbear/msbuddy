import argparse
import json

import joblib
import numpy as np
from brainpy import isotopic_variants
from imblearn.over_sampling import SMOTE
from scipy.stats import norm
from tqdm import tqdm

from msbuddy.base import read_formula, MetaFeature, Spectrum, Formula, CandidateFormula
from msbuddy.main import Msbuddy, MsbuddyConfig, _gen_subformula
from msbuddy.load import init_db
from msbuddy.ml import gen_ml_b_feature_single, pred_formula_feasibility
from msbuddy.cand import _calc_ms1_iso_sim
from msbuddy.utils import form_arr_to_str

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart



