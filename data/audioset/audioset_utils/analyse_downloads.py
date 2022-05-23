import os 
import csv
from tqdm import tqdm

data_dir = "/vol/bitbucket/jla21/proj/data/audioset"

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
				downloaded_files[subset_name].append([YTID, labels])

	
	# write a new csv containing only the YouTube audio segments that have been downloaded
	downloaded_subset_path = os.path.join(data_dir, f"{subset_name}-downloaded.csv")
	with open(downloaded_subset_path, 'w') as f:
		writer = csv.writer(f)
		writer.writerows(downloaded_files[subset_name])
