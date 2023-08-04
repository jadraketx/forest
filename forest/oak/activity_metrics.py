import logging
import os
import pandas as pd
import numpy as np
from forest.constants import Frequency
from dateutil import tz
from datetime import datetime, timedelta
from time import perf_counter
import sys

logger = logging.getLogger(__name__)

TIMESTAMP_COL = 'timestamp'
X_COL = "x"
Y_COL = "y"
Z_COL = "z"
G_UNIT = 9.80665 #m/s^2
EXERT_CUTOFF = 0.15 #g^2
MINUTES_IN_DAY = 1440
WEAR_CUTOFF = 0.004 #g
M5 = 5
M15 = 15
M30 = 30
M60 = 60
M120 = 120

def load_raw_data(d_datetime, dates_shifted, source_folder, file_list, tz_str="UTC"):

    logger.info("Loading raw data")
    t1 = perf_counter()
    # find file indices for this d_ind
    file_ind = [i for i, x in enumerate(dates_shifted)
                if x == d_datetime]
    # initiate dataframe
    data = pd.DataFrame()

    # check if there is at least one file for a given day
    if len(file_ind) <= 0:
        logger.info(f"No data found for date: {d_datetime}")
        return data

    # load data for a given day
    df_list = []
    for f in file_ind:
        # read data
        file_path = os.path.join(source_folder, file_list[f])
        df = pd.read_csv(file_path, header=0, usecols=[TIMESTAMP_COL, X_COL, Y_COL, Z_COL], dtype={TIMESTAMP_COL: 'int', X_COL: 'float', Y_COL: 'float', Z_COL: 'float'})
        df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], unit='ms', utc=True).dt.tz_convert(tz_str)
        df = df.set_index(TIMESTAMP_COL)
        df_list.append(df)

    data = pd.concat(df_list)
    t2 = perf_counter()
    logger.info("Finished loading data:{0:4.2f}s".format(t2-t1))
    return(data)

def calculate_day_coverage(epochs):
    s = epochs.size()
    coverage = sum(s != 0) / MINUTES_IN_DAY
    return(coverage)

def compile_epoch_metadata(epochs):
    s = epochs.size()
    meta = {
        'min_epoch': str(s.index[0]),
        'max_epoch': str(s.index[-1]),
        'num_epochs': len(s),
        'day_coverage': sum(s != 0) / MINUTES_IN_DAY,
        'frac_non_empty_epochs': sum(s != 0) / len(s),
        'num_empty_epochs': sum(s == 0),
        'frac_empty_epochs': sum(s == 0) / len(s),
        'avg_count_per_epoch': int(s.mean()),
        'std_count_per_epoch': int(s.std()),
        'max_count_across_epochs': int(s.max()),
        'min_count_across_epochs': int(s.min()),
        'num_raw_data': int(s.sum())
    }
    return(meta)

def find_non_wear_segments(epochs, sd_non_wear_threshold, non_wear_window):

    #see https://doi.org/10.1080/02640414.2019.1703301
    #method 1: calc variance of acc_x, acc_y, acc_z for each epoch
    #if variance is < 0.004g for 30 consecutive epochs, label all epochs as non-wear

    #method 2: same thing as above but with variance of the vector magnitude
    logger.info("Finding non-wear segments")
    t1 = perf_counter()

    l = len(epochs)
    candidates_1 = [0] * l
    candidates_2 = [0] * l
    time = [0] * l

    i = 0
    for t, row in epochs.iterrows():
        vars = row[['var_x','var_y','var_z']]
        if sum(pd.isna(vars)) > 0:
            candidates_1[i] = pd.NA
            candidates_2[i] = pd.NA
        else:
            stds = vars ** (1/2)
            if sum(stds < WEAR_CUTOFF) == 3:
                candidates_1[i] = 1
            std_vm = row['var_vm'] ** (1/2)
            if std_vm < WEAR_CUTOFF:
                candidates_2[i] = 1

        time[i] = t
        i = i + 1

    candidates_1 = pd.DataFrame({"candidates_1":candidates_1}, index=time)
    candidates_2 = pd.DataFrame({"candidates_2": candidates_2}, index=time)

    non_wear_1 = pd.DataFrame({"non_wear_1": ([0] * len(candidates_1))}, index=time)
    non_wear_2 = pd.DataFrame({"non_wear_2": ([0] * len(candidates_2))}, index=time)

    stop = l - non_wear_window
    for k in range(0, stop):
        window_1 = candidates_1['candidates_1'][k:(k+non_wear_window)]
        window_2 = candidates_2['candidates_2'][k:(k+non_wear_window)]

        temp_sum_1 = window_1.sum()
        temp_sum_2 = window_2.sum()

        num_na_1 = pd.isna(window_1).sum()
        num_na_2 = pd.isna(window_2).sum()

        # if temp_sum == non_wear_window, then 30 consecutive minutes identified, set all to non_wear
        if temp_sum_1 == (non_wear_window - num_na_1):
            non_wear_1['non_wear_1'][window_1.index] = 1
        if temp_sum_2 == (non_wear_window - num_na_2):
            non_wear_2['non_wear_2'][window_2.index] = 1


    epochs.insert(len(epochs.columns), "non_wear_1", non_wear_1)
    epochs.insert(len(epochs.columns), "non_wear_2", non_wear_2)

    t2 = perf_counter()
    logger.info("Finished finding non-wear segments: {0:4.2f}s".format(t2 - t1))
    return(epochs)

    #
    # l = len(epochs)
    # time = [0] * l
    # candidate = [0] * l
    # i = 0
    #
    # #determine which 1min epochs meet criteria - these become candidate non-wear times, encoded as 1's
    # #combine the 'votes' for each method
    # for t, df in epochs:
    #     std = df.std()
    #     vm = (df[X_COL] ** 2 + df[Y_COL]**2 + df[Z_COL]**2)**(1/2)
    #     std_vm = vm.std()
    #
    #     temp_std = 0
    #     temp_vm = 0
    #     res = 0
    #     if sum(pd.isna(std)) > 0:
    #         res = pd.NA
    #     else:
    #         if sum(std < WEAR_CUTOFF) == 3:
    #             temp_std = 1
    #         if std_vm < WEAR_CUTOFF:
    #             temp_vm = 1
    #         if (temp_std == 1) & (temp_vm == 1):
    #             res = 1
    #
    #     time[i] = t
    #     candidate[i] = res
    #     i = i + 1
    #
    # epoch_bool = pd.DataFrame({"time":time, "candidate": candidate})
    # epoch_bool = epoch_bool.set_index("time")
    #
    # epoch_non_wear = pd.DataFrame({"time":time, "non_wear":([0]*len(epoch_bool))})
    # epoch_non_wear = epoch_non_wear.set_index("time")
    # #identify 30 consecutive minutes of candidate non-wear epochs (i.e. 30 consecutive 1's)
    # #NAs are possible. If all non-NA's are 1's in a 30 min span with some NAs, label as non-wear
    # stop = len(epoch_bool) - non_wear_window
    # for k in range(0, stop):
    #     window = epoch_bool['candidate'][k:(k+non_wear_window)]
    #     temp_sum = window.sum()
    #     num_na = pd.isna(window).sum()
    #
    #     #if temp_sum == non_wear_window, then 30 consecutive minutes identified, set all to non_wear
    #     if temp_sum == (non_wear_window-num_na):
    #         epoch_non_wear['non_wear'][window.index] = 1


def calculate_epoch_statistics(epochs):

    logger.info("Calculating epoch statistcs")
    t1 = perf_counter()
    res = [0] * len(epochs)
    i = 0
    for t, df in epochs:
        vm = (df[X_COL] ** 2 + df[Y_COL]**2 + df[Z_COL]**2)**(1/2)
        df.insert(len(df.columns), 'vm', vm)
        vars = df.var()
        means = df.mean()
        mins = df.min()
        maxs = df.max()
        n = len(df)

        enmo = means['vm'] - 1
        res[i] = {
            "time":t,
            "nsamples":n,
            "min_x":mins['x'],
            "min_y":mins['y'],
            "min_z":mins['z'],
            "min_vm":mins['vm'],
            "max_x":maxs['x'],
            "max_y":maxs['y'],
            "max_z":maxs['z'],
            "max_vm":maxs['vm'],
            "var_x":vars['x'],
            "var_y":vars['y'],
            "var_z":vars['z'],
            "var_vm":vars['vm'],
            "mean_x":means['x'],
            "mean_y":means['y'],
            "mean_z":means['z'],
            "mean_vm":means['vm']
        }
        i = i + 1
    res = pd.DataFrame(res)
    res = res.set_index("time")
    t2 = perf_counter()
    logger.info("Finished calculating epoch statistics data:{0:4.2f}s".format(t2 - t1))
    return(res)


def calculate_daily_metrics(df):

    logger.info("Calculating physical activity metrics")
    t1 = perf_counter()
    #original exertional activity
    temp = df[df['nsamples'] > 1]
    sum_var = temp['var_x'] + temp['var_y'] + temp['var_z']
    is_exert = sum_var > EXERT_CUTOFF
    prop = sum(is_exert) / len(is_exert)
    daily_exert_orig = prop * 1440

    #new exertional activity based on wear time
    df_wear = temp[temp['non_wear_2'] == 0]
    sum_var2 = df_wear['var_x'] + df_wear['var_y'] + df_wear['var_z']
    is_exert2 = sum_var2 > EXERT_CUTOFF
    prop2 = sum(is_exert2) / len(is_exert2)
    daily_exert_wear = prop2 * 1440

    #average enmo
    enmo = df_wear['mean_vm'] - 1
    enmo = enmo.sort_values(ascending=False)
    avg_enmo = np.mean(enmo)

    #MX metrics
    l = len(enmo)

    m5 = -1
    m15 = -1
    m30 = -1
    m60 = -1
    m120 = -1

    if l >= M5:
        m5 = enmo[M5 - 1]
    if l >= M15:
        m15 = enmo[M15 - 1]
    if l >= M30:
        m30 = enmo[M30 - 1]
    if l >= M60:
        m60 = enmo[M60 - 1]
    if l >= M120:
        m120 = enmo[M120 - 1]

    #wear/non-wear ratio
    frac_wear = sum(df['non_wear_2'] == 0) / len(df)
    frac_non_wear = sum(df['non_wear_2'] == 1) / len(df)
    day = df.index[0].strftime('%Y-%m-%d')
    res = {
        "day":day,
        "daily_exert_orig":daily_exert_orig,
        "daily_exert_wear":daily_exert_wear,
        "avg_enmo":avg_enmo,
        "m5":m5,
        "m15":m15,
        "m30":m30,
        "m60":m60,
        "m120":m120,
        "frac_wear":frac_wear,
        "frac_non_wear":frac_non_wear
    }

    t2 = perf_counter()
    logger.info("Finished calculating metrics:{0:4.2f}s".format(t2 - t1))
    return(res)




def process_activity_metrics(study_folder: str, output_folder: str, beiwe_id: str, tz_str: str = None,
                             frequency: Frequency = Frequency.DAILY, time_start: str = None,
                             time_end: str = None, convert_to_g_unit=False, epoch_size="1min",
                             coverage_threshold=0.6, sd_non_wear_threshold=0.004, non_wear_window=30) -> None:

    t1 = perf_counter()

    user = beiwe_id
    logger.info(f"Processing data for beiwe_id: {user}")

    # determine timezone shift
    fmt = '%Y-%m-%d %H_%M_%S'
    from_zone = tz.gettz('UTC')
    if tz_str is None:
        tz_str = 'UTC'
    to_zone = tz.gettz(tz_str)

    # create folders to store results
    if frequency == Frequency.HOURLY_AND_DAILY:
        os.makedirs(os.path.join(output_folder, "daily"), exist_ok=True)
        os.makedirs(os.path.join(output_folder, "hourly"), exist_ok=True)
    else:
        os.makedirs(
            os.path.join(output_folder, frequency.name.lower()), exist_ok=True
        )

    source_folder = os.path.join(study_folder, user, "accelerometer")
    file_list = os.listdir(source_folder)
    file_list.sort()
    if len(file_list) == 0:
        logger.info(f"No files found for {user}")
        sys.exit(0)
    # transform all files in folder to datelike format
    if "+00_00.csv" in file_list[0]:
        file_dates = [file.replace("+00_00.csv", "") for file in file_list]
    else:
        file_dates = [file.replace(".csv", "") for file in file_list]

    file_dates.sort()

    # process dates
    dates = [datetime.strptime(file, fmt) for file in file_dates]
    dates = [date.replace(tzinfo=from_zone).astimezone(to_zone)
             for date in dates]

    # trim dataset according to time_start and time_end
    if time_start is not None and time_end is not None:
        time_min = datetime.strptime(time_start, fmt)
        time_min = time_min.replace(tzinfo=from_zone).astimezone(to_zone)
        time_max = datetime.strptime(time_end, fmt)
        time_max = time_max.replace(tzinfo=from_zone).astimezone(to_zone)
        dates = [date for date in dates if time_min <= date <= time_max]

    dates_shifted = [date - timedelta(hours=date.hour) for date in dates]

    # create time vector with days for analysis
    if time_start is None:
        date_start = dates_shifted[0]
        date_start = date_start - timedelta(hours=date_start.hour)
    else:
        date_start = datetime.strptime(time_start, fmt)
        date_start = date_start - timedelta(hours=date_start.hour)

    if time_end is None:
        date_end = dates_shifted[-1]
        date_end = date_end - timedelta(hours=date_end.hour)
    else:
        date_end = datetime.strptime(time_end, fmt)
        date_end = date_end - timedelta(hours=date_end.hour)

    days = pd.date_range(date_start, date_end, freq='D')

    if (
            frequency == Frequency.HOURLY_AND_DAILY
            or frequency == Frequency.HOURLY
    ):
        freq = 'H'
    else:
        freq = str(frequency.value) + 'H'
    days_hourly = pd.date_range(date_start, date_end + timedelta(days=1),
                                freq=freq)[:-1]

    meta_days = {}
    all_epochs = []
    metrics = []
    for d_ind, d_datetime in enumerate(days):
        logger.info(f"Processing day {str(d_ind)}/{str(len(days))}: {d_datetime}")

        #split data into 1 min chunks, or epochs
        data = load_raw_data(d_datetime, dates_shifted, source_folder, file_list, tz_str)
        if len(data) == 0:
            continue
        if convert_to_g_unit == True:
            data = data/G_UNIT

        #group data into 1 min epochs
        epochs = data.resample(epoch_size)
        meta = compile_epoch_metadata(epochs)
        print(meta)
        #meta_days[d_datetime] = meta
        coverage = calculate_day_coverage(epochs)
        coverage_threshold=0
        #check if epoch coverage meets daily threshold
        if coverage >= coverage_threshold:
            logger.info("Sufficient amount of data collected")
            #calculate epoch level statistics
            df = calculate_epoch_statistics(epochs)

            #compute wear/non-wear time
            df = find_non_wear_segments(df, sd_non_wear_threshold, non_wear_window)

            #calulate daily physical activity metrics
            #temp_metrics = calculate_daily_metrics(df)

            all_epochs.append(df)
            #metrics.append(temp_metrics)
        else:
            logger.info(f"Insufficient coverage: {d_datetime} will not be processed")

    if len(all_epochs) < 1:
        logger.info("Little to no data available for " + user + ". Terminating.")
        sys.exit(0)
    all_epochs = pd.concat(all_epochs)
    #metrics = pd.DataFrame(metrics)

    logger.info("Writing results to file")
    output_file_epochs = user + "_epochs.csv"
    dest_path_epochs = os.path.join(output_folder, "daily", output_file_epochs)
    all_epochs.to_csv(dest_path_epochs, index=True)

    #output_file_metrics = user + "_daily_pa_metrics.csv"
    #dest_path_metrics = os.path.join(output_folder, "daily", output_file_metrics)
    #metrics.to_csv(dest_path_metrics, index=False)

    t2 = perf_counter()
    logger.info("COMPLETE: {0:4.2f}s".format(t2-t1))
