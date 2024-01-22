import time
import ray
from tqdm import trange
import random

## num_cpus=1だと0~5が99%サンプルされてたまに6が生成される。num_cpus=16までは早い処理順に実行されるがnum_cpus=17以降はi=0から順に実行される 
@ray.remote
def selfplay(pid):
    sleep_length = random.randint(0, 10)
    time.sleep(sleep_length)
    return f"pid {pid} finished in {sleep_length} sec"

def run_selfplay():
    work_in_progresses = [selfplay.remote(i) for i in range(20)]
    for _ in range(10):
        finished, work_in_progresses = ray.wait(work_in_progresses, num_returns=1)
        print(ray.get(finished)[0])


if __name__ == "__main__":
    ray.init()
    start = time.time()
    run_selfplay()
    print("Elapsed:", time.time() - start)