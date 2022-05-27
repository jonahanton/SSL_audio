import os 
import csv
from tqdm import tqdm
import time

import torchaudio

from joblib import Parallel, delayed


data_dir = "/vol/bitbucket/jla21/proj/data/audioset"


def find_downloads():
	downloaded_files = {}
	# find out which files have been downloaded for audioset
	for subset_name in ["balanced_train_segments", "unbalanced_train_segments"]:
		
		subset_path = os.path.join(data_dir, f"{subset_name}.csv")
		subset_dir = os.path.join(data_dir, subset_name)
		
		downloaded_files[subset_name] = []
		
		# get the files which have been downloaded
		downloaded_YTIDs = set([os.path.splitext(fname)[0] for fname in os.listdir(subset_dir)])
		
		# get all files from the subset csv files
		with open(subset_path, 'r') as f:
			subset_data = csv.reader(f)
			
			for row_idx, row in enumerate(subset_data):
				# skip commented lines
				if row[0][0] == "#":
					continue
				
				YTID = row[0]
				labels = [s.strip().replace("\"", "") for s in row[3:]]
				labels = "#".join(labels)
				# check if YTID has been downloaded
				if YTID in downloaded_YTIDs:
					downloaded_files[subset_name].append([YTID, labels, subset_name])

		
		# write a new csv containing only the YouTube audio segments that have been downloaded
		downloaded_subset_path = os.path.join(data_dir, f"{subset_name}-downloaded.csv")
		with open(downloaded_subset_path, 'w') as f:
			writer = csv.writer(f)
			writer.writerows(downloaded_files[subset_name])


def check_downloads():

	downloaded_incorrectly = {}
	for subset_name in ["unbalanced_train_segments"]:
		downloaded_incorrectly[subset_name] = []
		subset_dir = os.path.join(data_dir, subset_name)

		# Parallel(n_jobs=-1, require='sharedmem')(delayed(check_audio_length)
		# 			(downloaded_incorrectly, subset_dir, subset_name, fname) 
		# 			for fname in os.listdir(subset_dir))
		dir_files = os.listdir(subset_dir)
		start = time.time()
		for i in tqdm(range(len(dir_files))):
			fname = dir_files[i]
			check_audio_length(downloaded_incorrectly, subset_dir, subset_name, fname)
		end = time.time()
		print(f'Total time: {start - end}')


		downloaded_incorrectly_subset_path = os.path.join(data_dir, f"{subset_name}-incorrectly_downloaded.csv")
		with open(downloaded_incorrectly_subset_path, 'w') as f:
			writer = csv.writer(f)
			writer.writerows(downloaded_incorrectly[subset_name])


def check_audio_length(downloaded_incorrectly, subset_dir, subset_name, fname):
	audio_fpath = os.path.join(subset_dir, fname)
	YTID = os.path.splitext(fname)[0]
	wav, sr = torchaudio.load(audio_fpath)
	if len(wav[0]) / sr < 9:
		# print(YTID)
		downloaded_incorrectly[subset_name].append([YTID])


if __name__ == "__main__":
	check_downloads()