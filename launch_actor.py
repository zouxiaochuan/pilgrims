import click
import common_utils
import multiprocessing as mp

import actor

@click.command()
@click.option('--config-file', default='config.json')
@click.option('--devices', default='2,3,4,5,6,7')
@click.option('--worker-per-device', default=4)
def run_cmd(config_file, devices, worker_per_device):
    config = common_utils.load_config(config_file)
    
    devices = ['cuda:' + i for i in devices.split(',')]

    procs = []
    for device in devices:
        for iw in range(worker_per_device):
            process = mp.Process(target=actor.run, args=(config, device))
            process.start()
            procs.append(process)
            pass
        pass

    for proc in procs:
        proc.join()
        pass
    pass


if __name__ == '__main__':
    run_cmd()
    pass
