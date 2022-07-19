"""
Plot & save graphs from training from the log file.
"""
import matplotlib.pyplot as plt 
import seaborn as sns
import os 
import fire 
import json
import pprint


def plot(*checkpoint_paths):
    sns.set()
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
    plt.xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax2.set_ylabel('kNN mAP')

    for checkpoint_path in checkpoint_paths:

        name = checkpoint_path.split('/')[-1]

        log_file = os.path.join(checkpoint_path, 'log.txt')
        if not os.path.exists(log_file):
            print(f'No log file found in path {log_file}!')
            continue 
        
        loss = []
        kNN_mAP = []
        with open(log_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                if 'train_loss' in data:
                    loss.append((data.get('train_loss'), data.get('epoch')))
                if 'train_knn_mAP' in data:
                    kNN_mAP.append((data.get('train_knn_mAP'), data.get('epoch')))


        ax1.plot([x[1] for x in loss], [x[0] for x in loss], marker='.', label=name)
        ax2.plot([x[1] for x in kNN_mAP], [x[0] for x in kNN_mAP], marker='.', label=name)

    ax1.legend(bbox_to_anchor=(1.02, 0.15), loc='lower left')
    ax2.legend(bbox_to_anchor=(1.02, 0.15), loc='lower left')

    if len(checkpoint_paths) == 1:
        outpath = os.path.join(checkpoint_paths[0], 'plots')
        if not os.path.exists(outpath):
            os.makedirs(outpath)
    else:
        outpath = './'

    plt.savefig(os.path.join(outpath, 'training_curves.png'), bbox_inches='tight')


if __name__ == "__main__":
    fire.Fire(plot)