from tqdm import tqdm
import time

# Simulate a list of tasks
tasks = list(range(12))

# Iterate through the tasks with tqdm to show the progress bar
for task in tqdm(tasks, desc="Processing", ncols=100, bar_format="{l_bar}{bar} | {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"):
    # Simulate a time-consuming task
    time.sleep(0.81)
