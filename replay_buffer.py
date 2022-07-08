import Pyro5.api as api
from Pyro5.nameserver import NameServer
import Pyro5.server as server
import click
import pyro_utils
import random
from threading import Thread
import time
import pickle


NAME_KEY = 'rb'


@api.expose
class ReplayBuffer():
    def __init__(self, max_size):
        self.buffer = []
        self.max_size = max_size

    @api.oneway
    def put(self, data):
        self.buffer.append(data)

        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)
            pass
        pass

    def sample(self, num=1):
        num = min(num, len(self.buffer))
        return random.sample(self.buffer, num)
        # return self.buffer[:num]
    pass


def report_thread(rb_obj):
    while True:
        time.sleep(5)
        print(f'buffer size: {len(rb_obj.buffer)}')
    pass


def start_report_thread(rb_obj: ReplayBuffer):
    thread = Thread(target=report_thread, args=(rb_obj,))
    thread.start()
    pass


@click.command()
@click.option('--master-ip', default='localhost')
@click.option('--master-port', default=32171)
@click.option('--max-size', default=1024 * 1024)
def run(master_ip, master_port, max_size):
    ns: NameServer = api.locate_ns(host=master_ip, port=master_port)
    id = pyro_utils.get_next_id(ns, NAME_KEY)
    name = f'{NAME_KEY}_{id}'
    daemon = server.Daemon()
    rb_obj = ReplayBuffer(max_size)
    start_report_thread(rb_obj)
    uri = daemon.register(rb_obj)
    ns.register(name, uri)
    try:
        daemon.requestLoop()
    finally:
        ns.remove(name)
        pass
    pass


if __name__ == '__main__':
    run()
    pass
