from Pyro5.nameserver import NameServer
import Pyro5.api as api
import Pyro5


def get_next_id(ns: NameServer, prefix) -> int:
    ids = ns.list(prefix)

    return len(ids)


def get_all_objects(ns: NameServer, prefix: str) -> list:
    proxies = []
    for name, uri in ns.list(prefix).items():
        print(name + ': ' + uri)
        obj = api.Proxy(uri)
        proxies.append(obj)
        pass

    return proxies


def init():
    Pyro5.config.SERIALIZER = 'marshal'
