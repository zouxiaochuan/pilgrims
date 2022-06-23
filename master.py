from Pyro5 import nameserver
import click
import pyro_utils


pyro_utils.init()


@click.command()
@click.option('--host', default='0.0.0.0')
@click.option('--port', default=32171)
def run(host, port):
    daemon = nameserver.NameServerDaemon(host=host, port=port)
    daemon.requestLoop()    
    pass


if __name__ == '__main__':
    run()
