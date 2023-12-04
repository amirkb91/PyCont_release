import os
import json


class Prob:
    def __init__(self):
        self.cont_params = None
        self.doffunction = None
        self.icfunction = None
        self.zerofunction = None
        self.zerofunction_firstpoint = None
        self.partitionfunction = None

    def read_contparams(self, cont_paramfile):
        self.cont_params = json.load(open(cont_paramfile))

    def add_doffunction(self, fxn):
        self.doffunction = fxn

    def add_icfunction(self, fxn):
        self.icfunction = fxn

    def add_zerofunction(self, fxn, fxn2=None):
        self.zerofunction = fxn
        if not fxn2:
            self.zerofunction_firstpoint = fxn
        else:
            self.zerofunction_firstpoint = fxn2

    def add_zerofunction_firstpoint(self, fxn):
        self.zerofunction_firstpoint = fxn

    def add_partitionfunction(self, fxn):
        self.partitionfunction = fxn
