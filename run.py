import sys, os
from classes import *
from multiprocessing import Pool, cpu_count
from ConfigParser import SafeConfigParser

def generate(config, seed, samples):
    tracer = generator(config, seed)
    tracer.generate(samples)

def flag_parser():
    config  = SafeConfigParser()
    config.read('config/%s.ini' % sys.argv[1])
    ntracks = int(sys.argv[2])
    return config, ntracks

if __name__ == '__main__':
    config, ntracks = flag_parser()
    cores = cpu_count()

    savedir = config.get('I/O', 'save')
    if savedir not in os.listdir('data/'):
        os.mkdir('data/%s/' % savedir)

    p = Pool()
    for n in range(cores):
        p.apply_async(generate, (config, n, ntracks/cores))
    p.close()
    p.join()
