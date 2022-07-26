import os
import glob

from mtist import mtist_utils as mu
from mtist import master_dataset_generation as mdg
from mtist import assemble_mtist as am
from mtist import infer_mtist as im

from matplotlib import pyplot as plt

###############################
# ADJUSTING GLOBAL PARAMETERS #
###############################

# We will be using all of the defaults (new defaults set in MTIST)

# mu.GLOBALS.MASTER_DATASET_DIR = 'master_datasets'
# mu.GLOBALS.MTIST_DATASET_DIR = path_to("mtist2_3sp_assembled")
# mu.GLOBALS.GT_DIR = path_to("mtist2_3sp_gts")

# Master dataset specifics
# mdg.MASTER_DATASET_DEFAULTS.random_seeds = mdg.MASTER_DATASET_DEFAULTS.expanded_random_seeds

# MTIST dataset specifics
# am.ASSEMBLE_MTIST_DEFAULTS.SAMPLING_FREQ_PARAMS = [3, 5, 8, 10, 15]
# am.ASSEMBLE_MTIST_DEFAULTS.N_TIMESERIES_PARAMS = [5, 10, 25, 50]

# Ground truth specifics
mu.GLOBALS.GT_NAMES = [
    path.split("/")[-1].split(".csv")[0].replace("aij", "gt")
    for path in glob.glob("ground_truths/interaction_coefficients/*.csv")
]

###############################
# GENERATE 3-SPECIES MTIST  #
###############################

mdg.generate_mtist_master_datasets()
plt.close("all")

am.assemble_mtist()

################################
# INFER 3-SPECIES MTIST 2 WAYS #
################################

# inference_names = [
#     "default",
#     "ols_with_p",
# ]
# prefixes = [f"{name}_" for name in inference_names]

# inference_fxn_handles = [
#     im.infer_from_did,
#     lambda v: im.infer_from_did_ols_with_p(v, save=True),
# ]


# for inference_type, prefix, handle in zip(inference_names, prefixes, inference_fxn_handles):
#     print(inference_type)
#     im.INFERENCE_DEFAULTS.INFERENCE_PREFIX = prefix
#     im.INFERENCE_DEFAULTS.INFERENCE_FUNCTION = handle
#     _ = im.infer_and_score_all(save_inference=True, save_scores=True)