import argparse
from forest.oak.activity_metrics import process_activity_metrics
import logging
'''
parser = argparse.ArgumentParser()
parser.add_argument("study_folder", help="path to the folder with raw data", type=str)
parser.add_argument("output_folder", help="path to output folder", type=str)
parser.add_argument("beiwe_id", help="Beiwe user id", type=str)
parser.add_argument("time_start", help="start of time window to analyze data", type=str)
parser.add_argument("end_start", help="end of time window to analyze data", type=str)
parser.add_argument("tz", help="string representation of timezone", type=str, default="UTC")
args = parser.parse_args()
'''

CUSTOM_FORMAT = '%(created)f %(levelname)-8s %(message)s'
logging.basicConfig(level = logging.DEBUG,  # Ensure that all messages are printed.
                    force = True)

study_folder = "/Users/jadrake/Local_dev/beiwe_msk_analysis/temp_data/"
output_folder = "/Users/jadrake/Local_dev/beiwe_msk_analysis/results/"
tz_str = "America/Chicago"
beiwe_id = "g5xnzgjd"
time_start = None
time_stop = None

process_activity_metrics(study_folder=study_folder, output_folder=output_folder, beiwe_id=beiwe_id, tz_str=tz_str, convert_to_g_unit=True)