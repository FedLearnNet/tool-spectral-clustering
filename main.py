from pyfedappwrap.engine.runtime import FedDBEngine

from app import SpectralClusteringAPP

engine = FedDBEngine()
engine.register(SpectralClusteringAPP())

if __name__ == '__main__':
    engine.start()
    engine.wait_until_stop()
