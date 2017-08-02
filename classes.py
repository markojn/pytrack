import copy, time
import numpy as np
from fields import *
from ConfigParser import SafeConfigParser

kb = 1.38E-23
I  = np.identity(3, dtype=float)

def gaussian(x_, b, n):
    '''defines a flat gaussian function
    with width b and power n'''
    exponent = -np.power(x_/b, n)
    output   = np.exp(exponent)
    return output/np.sum(output)

def thermal(v_, T, m):
    '''defines a maxwell-boltzmann
    distribution at temperature T'''
    exponent = -m * np.square(v_)/(2 * kb * T)
    output   = np.multiply(np.square(v_), np.exp(exponent))
    return output/np.sum(output)

class generator(object):
    '''object defining a particle track generator'''

    def __init__(self, config, seed):
        '''initialise the particle type and
        initial source distributions'''

        self.seed = seed
        np.random.seed(self.seed)
        ptype = config.get('setup', 'type').lower()

        if ptype in ['pbar', 'antiproton']:
            self.m =  1.67E-27
            self.q = -1.60E-19

        elif ptype in ['e-', 'electron']:
            self.m =  9.11E-31
            self.q = -1.60E-19

        elif ptype in ['e+', 'positron']:
            self.m =  9.11E-31
            self.q = +1.60E-19

        self.dt = config.getfloat('setup', 'step')

        self.xsource = []
        for xi in ['x', 'y', 'z']:
            self.xsource.append(config.getfloat('source', xi))
        self.xsource = np.array(self.xsource, dtype=float)

        b = 1E-3 * config.getfloat('source', 'b')
        n = config.getfloat('source', 'n')

        self.radii = np.linspace(0, 4. * b, int(1E+4))
        self.rpdf  = gaussian(self.radii, b, n)

        T = config.getfloat('source', 'T')
        vmean = np.sqrt(2. * kb * T / self.m)

        self.vtrans = np.linspace(0, 4. * vmean, int(1E+4))
        self.vpdf   = thermal(self.vtrans, T, self.m)

        E = config.getfloat('source', 'E')
        self.vaxis  = np.sqrt(2. * E * abs(self.q) / self.m)

        self.interpolator = field()
        self.sample_source()

        self.savedir = config.get('I/O', 'save')

    def sample_source(self):
        '''sample an initial state from the
        source plasma distributions'''

        self.x_ = copy.copy(self.xsource)
        self.v_ = np.zeros(3, dtype=float)

        theta    = np.random.uniform(0, 2. * np.pi)
        unit     = np.array([np.cos(theta), np.sin(theta), 0], dtype=float)
        self.x_ += np.random.choice(self.radii, p=self.rpdf) * unit

        phi      = np.random.uniform(0, 2. * np.pi)
        unit     = np.array([np.cos(phi), np.sin(phi), 0], dtype=float)
        speed    = np.random.choice(self.vtrans, p=self.vpdf)
        self.v_ += speed * unit

        axis     = np.array([0, 0, 1], dtype=float)
        unit     = np.cross(axis, unit)
        field    = np.linalg.norm(self.interpolator.solve(self.x_))
        self.x_ += (self.m * speed)/(self.q * field) * unit

        self.v_ += self.vaxis * axis

    def hat_map(self):
        '''evaluate the leapfrog algorithm magnetic map'''
        Bx, By, Bz = self.interpolator.solve(self.x_)

        hat_map = np.array([[ 0, -Bz, By],
                            [ Bz, 0, -Bx],
                            [-By, Bx, 0 ]], dtype=float)

        return (self.q * self.dt * hat_map)/(2. * self.m)

    def push(self, fname):
        '''push a single particle and save to file'''
        max_steps = int(1E+6)

        x_store = np.zeros((max_steps, 3), dtype=float)
        v_store = np.zeros((max_steps, 3), dtype=float)

        x_store[0] = self.x_
        v_store[0] = self.v_

        for n in range(1, max_steps):
            omega = self.hat_map()
            self.v_  = np.dot(np.linalg.inv(I + omega), np.dot(I - omega, self.v_))
            self.x_ += self.dt * self.v_

            x_store[n] = self.x_
            v_store[n] = self.v_

            if self.x_[2] > -4.00:
                break

        np.save('data/%s/%04d.npy' % (self.savedir, fname), (x_store[:n], v_store[:n]))

    def generate(self, ntracks):
        '''sequentially generates particle tracks'''
        for n in range(ntracks):
            fname = (self.seed * ntracks) + n
            self.push(fname)
            self.sample_source()
