import argparse
import csv 
import os 

from joblib import Parallel, delayed


def get_audio(args, row):

    # ignore first few lines (all start with #)
    if "#" in row[0]:
        return

    # skip over files already downloaded 
    if os.path.exists(args.outpath + "/" + str(row[0]) + ".wav"):
        return

    os.system(("ffmpeg -ss " + str(row[1]) + " -t 10 -i $(youtube-dl -f 'bestaudio' -g https://www.youtube.com/watch?v=" +
                str(row[0]) + ") -ar " + str(args.sample_rate) + " -- \"" + args.outpath + "/" + str(row[0]) + ".wav\""))
    


def download(args):

    with open(args.csv_dataset) as csv_file:
        reader = csv.reader(csv_file, skipinitialspace=True)

        Parallel(n_jobs=-1)(delayed(get_audio)(args, row) for row in reader)

            

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-fs', "--sample_rate", type=int, default=16000,
                        help="Sample rate of audio to download. Default 16kHz")
    parser.add_argument('--csv_dataset', type=str, default='./unbalanced_train_segments.csv',
                        help='Path to CSV file containing AudioSet in YouTube-id/timestamp form')
    parser.add_argument('--outpath', type=str, default='./unbalanced_train_segments/')

    args = parser.parse_args()


    if not os.path.isdir(args.outpath):
        os.makedirs(args.outpath)

    download(args)