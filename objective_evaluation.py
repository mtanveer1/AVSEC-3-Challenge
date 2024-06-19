'''
Adapted from original code by Clarity Challenge
https://github.com/claritychallenge/clarity
'''

from queue import Empty
import hydra
from omegaconf import DictConfig
import os
from tqdm import tqdm
import csv
import json
from soundfile import SoundFile
from pesq import pesq
from pystoi import stoi
from concurrent.futures import ProcessPoolExecutor

def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def run_pesq(target, enhanced, sr, mode):
    """Compute PESQ from: https://github.com/ludlows/python-pesq/blob/master/README.md
        Args:
            target (string): Name of file to read
            enhanced (string): Name of file to read
            sr (int): sample rate of files
            mode (string): 'wb' = wide-band (16KHz); 'nb' narrow-band (8KHz)
        Returns:
            PESQ metric (float)
                """
    return pesq(sr, target, enhanced, mode)

def run_stoi(target, enhanced, sr):
    """Compute STOI from: https://github.com/mpariente/pystoi
           Args:
               target (string): Name of file to read
               enhanced (string): Name of file to read
               sr (int): sample rate of files
           Returns:
               STOI metric (float)
                   """
    return stoi(target, enhanced, sr)

def read_audio(filename):
    """Read a wavefile and return as numpy array of floats.
            Args:
                filename (string): Name of file to read
            Returns:
                ndarray: audio signal
            """
    try:
        wave_file = SoundFile(filename)
    except:
        # Ensure incorrect error (24 bit) is not generated
        raise Exception(f"Unable to read {filename}.")
    return wave_file.read()

def run_metrics(scene, enhanced, target, cfg):

    # Retrieve the scene name
    scene_name = scene["scene"]

    enh_file = os.path.join(enhanced, f"{scene_name}{cfg['enhanced_suffix']}.wav")
    #print(enh_file)
    tgt_file = os.path.join(target, f"{scene_name}{cfg['target_suffix']}.wav")
    scene_metrics_file = os.path.join(cfg["metrics_results"], f"{scene_name}.csv")
    #print(scene_metrics_file)

    # Skip processing with files dont exist or metrics have already been computed
    if ( not os.path.isfile(enh_file) ) or ( not os.path.isfile(tgt_file) ) or ( os.path.isfile(scene_metrics_file)) :
        return

    # Read enhanced signal
    enh = read_audio(enh_file)
    # Read clean/target signal
    clean = read_audio(tgt_file)

    # Check that both files are the same length, otherwise computing the metrics results in an error
    if len(clean) != len(enh):
        raise Exception(
            f"Wav files {enh_file} and {tgt_file} should have the same length"
        )

    # Compute metrics
    m_stoi = run_stoi(clean, enh, cfg["objective_metrics"]["fs"])
    m_pesq = run_pesq(clean, enh, cfg["objective_metrics"]["fs"], cfg["objective_metrics"]["mode"])

    # Store scene metrics in a tmp file
    with open(scene_metrics_file, "w") as csv_f:
        csv_writer = csv.writer(csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow([scene_name, m_stoi, m_pesq])

@hydra.main(config_path=".", config_name="config", version_base="1.2")
def compute_metrics(cfg: DictConfig) -> None:
    # paths to data
    enhanced = os.path.join(cfg["enhanced"])
    target = os.path.join(cfg["target"])
    # json file with info about scenes
    scenes_eval = json.load(open(cfg["scenes_names"]))
    # csv file to store metrics
    create_dir(cfg["metrics_results"])
    metrics_file = os.path.join(cfg["metrics_results"], "metrics.csv")
    #print(metrics_file)
    csv_lines = [["scene", "stoi", "pesq"]]
    futures = []
    ncores = 20
    with ProcessPoolExecutor(max_workers=ncores) as executor:
        for scene in scenes_eval:
            futures.append(executor.submit(run_metrics, scene, enhanced, target, cfg))
        proc_list = [future.result() for future in tqdm(futures)]

    # Store results in one file
    with open(metrics_file, "w", newline='') as csv_f:
        csv_writer = csv.writer(csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        try:
            for scene in tqdm(scenes_eval):
                scene_name = scene["scene"]
                scene_metrics_file = os.path.join(cfg["metrics_results"], f"{scene_name}.csv")
                with open(scene_metrics_file, newline='') as csv_f:
                    scene_metrics = csv.reader(csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    for row in scene_metrics:
                        if row:
                            csv_writer.writerow(row)
                # remove tmp file
                #os.system(f"del {scene_metrics_file}")
                absolute_path = os.path.abspath(scene_metrics_file)
                raw_filename = fr"{absolute_path}"
                os.system(f"del {raw_filename}")
        except Exception as e:
            print("An error occurred:", e)    

if __name__ == "__main__":

    compute_metrics()
