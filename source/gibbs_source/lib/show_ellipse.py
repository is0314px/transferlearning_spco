import numpy as np
import math as m
import matplotlib.pylab as plt
import matplotlib.pyplot as pp
from matplotlib import patches
import scipy.stats as ss
from scipy.stats import invwishart
from numpy.random import *
from scipy.stats import chi2
from matplotlib.patches import Ellipse

class ConfidenceEllipse:
    def __init__(self, Mu, Sigma, p=0.95):
        self.p = p

        self.means = Mu
        self.cov = Sigma

        lambdas, vecs = np.linalg.eigh(self.cov)
        order = lambdas.argsort()[::-1]
        lambdas, vecs = lambdas[order], vecs[:,order]

        c = np.sqrt(chi2.ppf(self.p, 2))
        self.w, self.h = 2 * c * np.sqrt(lambdas)
        self.theta = np.degrees(np.arctan(
            ((lambdas[0] - lambdas[1])/self.cov[0,1])))
        
    def get_params(self):
        return self.means, self.w, self.h, self.theta

    def get_patch(self, line_color="black", face_color="none", alpha=0):
        el = Ellipse(xy=self.means,
                     width=self.w, height=self.h,
                     angle=self.theta, color=line_color, alpha=alpha)
        el.set_facecolor(face_color)
        return el

def main(mu,sigma,K,ax):
    new_mu = np.delete(mu,[2,3])
    new_sigma = np.delete(np.delete(sigma,[2,3],0),[2,3],1)

    el = ConfidenceEllipse(new_mu, new_sigma, p=0.95)
    ax.add_artist(el.get_patch(line_color="black",face_color="none",alpha=0.5))

        
        
