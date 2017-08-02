import os, sys
import numpy as np
import copy

force = False

class mesh(object):
    '''class defining a field map on a regular grid'''
    def __init__(self, fname):
        if '%s.npy' % fname in os.listdir('maps/cache'):
            self.z_, self.r_, self.dz, self.dr, self.Fz, self.Fr = np.load('maps/cache/%s.npy' % fname)

        elif '%s.table' % fname in os.listdir('maps/'):
            header = 6
            with open('maps/%s.table' % fname) as file:
                parser = file.readlines()[header:]
            lines = len(parser)
            for n in range(lines):
                parser[n] = parser[n].split()
            parser = np.transpose(np.array(parser, dtype=float))

            zmin, zmax = np.amin(parser[0]), np.amax(parser[0])
            rmin, rmax = np.amin(parser[1]), np.amax(parser[1])

            nz = len(np.where(parser[1] == rmin)[0])
            nr = len(np.where(parser[0] == zmin)[0])

            self.z_ = np.linspace(zmin, zmax, nz)
            self.r_ = np.linspace(rmin, rmax, nr)

            self.dz = self.z_[1] - self.z_[0]
            self.dr = self.r_[1] - self.r_[0]

            Fz, Fr = [], []
            for j in range(nr):
                start = (j+0) * nz
                stop  = (j+1) * nz
                Fz.append(parser[2][start:stop])
                Fr.append(parser[3][start:stop])

            self.Fz = np.transpose(np.array(Fz, dtype=float))
            self.Fr = np.transpose(np.array(Fr, dtype=float))
            np.save('maps/cache/%s.npy' % fname, (self.z_, self.r_, self.dz, self.dr, self.Fz, self.Fr))

        else:
            print 'Failed to load all field maps!\nExiting...'
            exit()

class field(object):
    '''interpolator class that retains bicubic coefficients'''
    def __init__(self):
        self.i = np.nan
        self.j = np.nan

        self.mesh = mesh('ALPHA-II')

        self.z_coeffs = np.nan
        self.r_coeffs = np.nan

        self.calls = 0

    def coefficients(self, i, j):
        self.calls += 1

        fz_matrix = np.zeros((4,4), dtype=float)
        fr_matrix = np.zeros((4,4), dtype=float)

        for l in range(2):
            for m in range(2):
                fz_matrix[l,m] = self.mesh.Fz[i+l,j+m]
                fr_matrix[l,m] = self.mesh.Fr[i+l,j+m]

                fz_matrix[l+2,m] = (-self.mesh.Fz[abs(i+l-1), j+m] + self.mesh.Fz[i+l+1, j+m])/2.
                fr_matrix[l+2,m] = (-self.mesh.Fr[abs(i+l-1), j+m] + self.mesh.Fr[i+l+1, j+m])/2.

                fz_matrix[l,m+2] = (-self.mesh.Fz[i+l, abs(j+m-1)] + self.mesh.Fz[i+l, j+m+1])/2.
                fr_matrix[l,m+2] = (-self.mesh.Fr[i+l, abs(j+m-1)] + self.mesh.Fr[i+l, j+m+1])/2.

                fz_matrix[l+2,m+2] = (self.mesh.Fz[abs(i+l-1), abs(j+m-1)] - self.mesh.Fz[i+l+1, abs(j+m-1)] - self.mesh.Fz[abs(i+l-1), j+m+1] + self.mesh.Fz[i+l+1, j+m+1])/4.
                fr_matrix[l+2,m+2] = (self.mesh.Fr[abs(i+l-1), abs(j+m-1)] - self.mesh.Fr[i+l+1, abs(j+m-1)] - self.mesh.Fr[abs(i+l-1), j+m+1] + self.mesh.Fr[i+l+1, j+m+1])/4.

        M = np.array([[1, 0,-3, 2],
                      [0, 0, 3,-2],
                      [0, 1,-2, 1],
                      [0, 0,-1, 1]], dtype=float)

        self.z_coeffs = np.dot(np.transpose(M), np.dot(fz_matrix, M))
        self.r_coeffs = np.dot(np.transpose(M), np.dot(fr_matrix, M))

    def solve(self, x_):
        x, y, z = 1E+2 * x_
        r = np.sqrt(x**2 + y**2)
        if r != float(0):
            unit = np.array([x, y, 0], dtype=float)/r
        else:
            unit = np.zeros(3, dtype=float)

        i = np.where(self.mesh.z_ <= z)[0][-1]
        j = np.where(self.mesh.r_ <= r)[0][-1]

        if (i, j) != (self.i, self.j) or force:
            self.coefficients(i, j)
            self.i = i
            self.j = j

        z = (z - self.mesh.z_[i])/self.mesh.dz
        r = (r - self.mesh.r_[j])/self.mesh.dr
        z_vector = np.array([1, z, z**2, z**3], dtype=float)
        r_vector = np.array([1, r, r**2, r**3], dtype=float)

        Bz = 1E-4 * np.dot(z_vector, np.dot(self.z_coeffs, r_vector))
        Br = 1E-4 * np.dot(z_vector, np.dot(self.r_coeffs, r_vector))

        return (Br * unit) + (Bz * np.array([0,0,1], dtype=float))
