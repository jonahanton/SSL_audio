import json 
import os 
import fire
import glob 
from tqdm import tqdm
import pprint 


BASE_DIR = "${EPHEMERAL}/hear/embeddings"
TASKS = [
	"beehive_states_fold0-v2-full",
	"beehive_states_fold1-v2-full",
	"beijing_opera-v1.0-hear2021-full",
	"dcase2016_task2-hear2021-full",
	"esc50-v2.0.0-full",
	"fsd50k-v1.0-full",
	"gunshot_triangulation-v1.0-full",
	"libricount-v1.0.0-hear2021-full",
	"maestro-v3.0.0-5h",
	"mridangam_stroke-v1.5-full",
	"mridangam_tonic-v1.5-full",
	"nsynth_pitch-v2.2.3-50h",
	"nsynth_pitch-v2.2.3-5h",
	"speech_commands-v0.0.2-5h",
	"speech_commands-v0.0.2-full",
	"tfds_crema_d-1.0.0-full",
	"tfds_gtzan-1.0.0-full",
	"tfds_gtzan_music_speech-1.0.0-full",
	"vocal_imitation-v1.1.3-full",
	"vox_lingua_top10-hear2021-full",
]


def extract_task_score(model_dir, task):

	results_json = os.path.join(*[model_dir, "hear.sample", task, "test.predicted-scores.json"])
	with open(results_json, "w") as jsonfile:
		results = json.load(jsonfile)

	if "test" in results:
		score = results["test"]["test_score"]
	elif "aggregated_scores" in results:
		score = results["aggregated_scores"]["test_score_mean"]
	
	return score


def extract_model_scores(model_dir):

	scores = {}
	for task in TASKS:
		scores[task] = extract_task_score(model_dir, task)
	return scores


def extract_all():
	
	model_dirs = glob.glob(BASE_DIR + "/*/")
	all_scores = {}
	for model_dir in tqdm(model_dirs):
		all_scores[model_dir.split("/")[-1]] = extract_model_scores(model_dir)

	pprint.pprint(all_scores)
		 

if __name__ == "__main__":
	extract_all()
 