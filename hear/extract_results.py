import json
import os
import glob
from tqdm import tqdm
import numpy as np
import math
import pprint


BASE_DIR = "/rds/general/user/jla21/ephemeral/hear/embeddings"
OUTPATH = "/rds/general/user/jla21/home/SSL_audio/hear/results.json"
TASKS = dict(
	environmental=[
		"beehive_states_fold0-v2-full",
		"beehive_states_fold1-v2-full",
		"esc50-v2.0.0-full",
		"fsd50k-v1.0-full",
		"gunshot_triangulation-v1.0-full",
	],
	speech=[
		"speech_commands-v0.0.2-5h",
		"speech_commands-v0.0.2-full",
		"tfds_crema_d-1.0.0-full",
		"vocal_imitation-v1.1.3-full",
		"vox_lingua_top10-hear2021-full",
		"libricount-v1.0.0-hear2021-full",	
	],
	music=[
		"beijing_opera-v1.0-hear2021-full",
		"mridangam_stroke-v1.5-full",
		"mridangam_tonic-v1.5-full",
		"nsynth_pitch-v2.2.3-50h",
		"nsynth_pitch-v2.2.3-5h",
		"tfds_gtzan-1.0.0-full",
		"tfds_gtzan_music_speech-1.0.0-full",
	],
	other=[
		"dcase2016_task2-hear2021-full",
		"maestro-v3.0.0-5h",
	],
)


def extract_task_score(model_dir, task):

	sampling_dir = os.listdir(model_dir)[0]
	results_json = os.path.join(*[model_dir, sampling_dir, task, "test.predicted-scores.json"])

	try:
		with open(results_json, "r") as jsonfile:
			results = json.load(jsonfile)
	except FileNotFoundError:
		return

	if "test" in results:
		score = results["test"]["test_score"]
	elif "aggregated_scores" in results:
		score = results["aggregated_scores"]["test_score_mean"]

	return score


def extract_model_scores(model_dir):

	scores = {}
	for task_type, tasks in TASKS.items():
		for task in tasks:
			scores.setdefault(task_type, {})
			score = extract_task_score(model_dir, task)
			if score is not None:
				scores[task_type][task] = score
		avg = np.mean(list(scores[task_type].values()))
		if math.isfinite(avg):
			scores[task_type]["AVERAGE"] = avg
	return scores


def extract_all():

	model_dirs = glob.glob(BASE_DIR + "/*/")
	all_scores = {}
	for model_dir in tqdm(model_dirs):
		model_name = model_dir.strip("/").split("/")[-1]
		all_scores[model_name] = extract_model_scores(model_dir)

	with open(OUTPATH, "w") as jsonfile:
		json.dump(all_scores, jsonfile, indent=4)



if __name__ == "__main__":
	extract_all()
