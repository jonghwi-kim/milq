import os 
from .clef_extractors import *


if not os.environ.get("CLEF_HOME"):
    os.environ["CLEF_HOME"] = os.path.join(os.path.expanduser("~"), "data", "clef", "E0008")

if not os.path.exists(os.environ["CLEF_HOME"]):
    raise RuntimeError("Environment variable CLEF_HOME is not a valid path.")

CLEF_BASE_DIR = os.environ["CLEF_HOME"]

CLEF_LOWRES_DIR = os.path.join(CLEF_BASE_DIR, "ictir19_simulate_low_resource", "long_paper")

DEFAULT_OUTPUT_DIR = "./clef_preprocessed"

if not CLEF_BASE_DIR:
    raise FileNotFoundError(f"Download CLEF and set CLEF_BASE_DIR in {__file__}")

#
# CLEF paths
#
PATH_BASE_QUERIES = os.path.join(CLEF_BASE_DIR, "Topics")
PATH_BASE_DOCUMENTS = os.path.join(CLEF_BASE_DIR, "DocumentData", "DataCollections")
PATH_BASE_EVAL = os.path.join(CLEF_BASE_DIR, "RelAssess")


# Prepare dutch CLEF data paths
nl_algemeen_dagblad = (os.path.join(PATH_BASE_DOCUMENTS, "Dutch_data", "algemeen_dagblad"), extract_dutch)
nl_nrc_handelsblad = (os.path.join(PATH_BASE_DOCUMENTS, "Dutch_data", "nrc_handelsblad"), extract_dutch)
dutch = {
    "2001": [nl_algemeen_dagblad, nl_nrc_handelsblad],
    "2002": [nl_algemeen_dagblad, nl_nrc_handelsblad],
    "2003": [nl_algemeen_dagblad, nl_nrc_handelsblad]
}


# Prepare italian CLEF data paths
it_lastampa = (os.path.join(PATH_BASE_DOCUMENTS, "Italian_data", "la_stampa"), extract_italian_lastampa)
it_sda94 = (os.path.join(PATH_BASE_DOCUMENTS, "Italian_data", "sda_italian"), extract_italian_sda9495)
it_sda95 = (os.path.join(PATH_BASE_DOCUMENTS, "Italian_data", "agz95"), extract_italian_sda9495)
italian = {
    "2001": [it_lastampa, it_sda94],
    "2002": [it_lastampa, it_sda94],
    "2003": [it_lastampa, it_sda94, it_sda95]
}


# Prepare finnish CLEF data paths
aamu9495 = os.path.join(PATH_BASE_DOCUMENTS, "Finnish_data", "aamu")
fi_ammulethi9495 = (aamu9495, extract_finish_aamuleth9495)
finnish = {
    "2001": [],
    "2002": [fi_ammulethi9495],
    "2003": [fi_ammulethi9495]
}

# Prepare english CLEF data paths
gh95 = (os.path.join(PATH_BASE_DOCUMENTS, "English_data", "GlasgowHerald95", "GH95"), extract_english_gh)
latimes = (os.path.join(PATH_BASE_DOCUMENTS, "English_data", "LATimes94"), extract_english_latimes)
english = {
    "2000": [latimes],
    "2001": [latimes],
    "2002": [latimes],
    "2003": [latimes]
}


# Prepare german CLEF data paths
der_spiegel = (os.path.join(PATH_BASE_DOCUMENTS, "German_data", "der_spiegel"), extract_german_derspiegel)
fr_rundschau = (os.path.join(PATH_BASE_DOCUMENTS, "German_data", "fr_rundschau"), extract_german_frrundschau)
de_sda94 = (os.path.join(PATH_BASE_DOCUMENTS, "German_data", "sda_german"), extract_german_sda)
de_sda95 = (os.path.join(PATH_BASE_DOCUMENTS, "German_data", "sda95"), extract_german_sda)
german = {
    "2002": [der_spiegel, fr_rundschau, de_sda94, de_sda95],
    "2003": [der_spiegel, fr_rundschau, de_sda94, de_sda95]
}


# Prepare russian CLEF data paths
xml = (os.path.join(PATH_BASE_DOCUMENTS, "Russian_data", "xml"), extract_russian)
russian = {"2003": [xml]}
all_paths = {"nl": dutch, "it": italian, "fi": finnish, "en": english, "de": german, "ru": russian}

# Utility function
ALL_LANGUAGES = [("de", "german"), ("en", "english"), ("ru", "russian"), ("fi", "finnish"), ("it", "italian"),
             ("fr", "french"), ("tr", "turkish"), ("es", "spanish"), ("nl", "dutch"), ("sv", "sv"), 
             ("sw", "swedish"), ("so", "somali")]
short2pair = {elem[0]: elem for elem in ALL_LANGUAGES}
long2pair = {elem[1]: elem for elem in ALL_LANGUAGES}

def get_lang2pair(language):
    return long2pair[language] if len(language) != 2 else short2pair[language]