#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib
import typing
import json
import os

def auto_property(attr_storage_name):
    '''
    Automatically decorate the attribute with @property
    '''
    def get_attr(instance):
        return instance.__dict__[attr_storage_name]

    def set_attr(instance, value):
        instance.__dict__[attr_storage_name] = value
    
    return property(get_attr, set_attr)
       
class FrozenClass():
    '''
    For child classes. It prevent to add new attributes to the setting from outside
    and prevents giving mistaken arguments to the setting.
    '''
    __is_frozen = False
    def __setattr__(self, key, value):
        if self.__is_frozen and not hasattr(self, key):
            raise AttributeError(f"'{key}' is not among the attributes of targeted object. It is an instance of a frozen class.")

        super().__setattr__(key, value)

    def set_attributes_with_keys(self, kwargs):
        for attr_name, attr_value in  kwargs.items():
            self.__setattr__(attr_name, attr_value)

    def __call__(self, kwargs):
        self.set_attributes_with_keys(self, kwargs)
        
    def _frozen(self):
        self.__is_frozen = True


        
class SHAPRConfig(FrozenClass):
    """
    Manages the configuration of the SHAPR
    """
    __config_param_names = dict(
        path = "__path",
        result_path = "__result_path",
        pretrained_weights_path = "__pretrained_weights_path",
        random_seed = "__random_seed",
        batch_size="__batch_size",
        epochs_SHAPR="epochs_SHAPR",
        epochs_cSHAPR="epochs_cSHAPR",
        topo_lambda="__topo_lambda",  # strength of topological regularisation
        topo_interp="__topo_interp",  # size of downsampled input data
        topo_feat_d='__topo_feat_d',  # dim. of topological features to use
        topo_feat_s='__topo_feat_s',  # superlevel features 
        topo_loss_q='__topo_loss_q',  # exponent for loss calculations
        topo_loss_r='__topo_loss_r',  # additional regularisation
    )

    __config_param_default = dict(
        path = "",
        result_path = "",
        pretrained_weights_path = "",
        random_seed = 0,
        batch_size = 6,
        epochs_SHAPR = 30,
        epochs_cSHAPR = 30,
        topo_lambda=0.0,
        topo_interp=0,
        topo_feat_d=2,
        topo_feat_s=False,
        topo_loss_q=2,
        topo_loss_r=False,
    )

    def __init__(self, params=None):
        for param, storing_param in self.__config_param_names.items():
            self.__setattr__(param, auto_property(storing_param))
        self._frozen()

        # setting initial values.
        self.set_attributes_with_keys(self.__config_param_default)

        if params is None:
            params_fname = os.path.join(
                os.path.dirname(__file__), 'params.json'
            )

            if os.path.isfile(params_fname):
                self.read_json(params_fname)
        else:
            self.read_json(params)

    def __str__(self):
        s = "------ settings parameters ------\n"
        for _k, _ in self.__config_param_names.items():
            s += f"{_k}: {self.__getattribute__(_k)}\n"
        return s

    def read_json(self, file_path):
        with open(file_path) as f:
            _initial_values = json.load(f)
        _initial_values = {_k: _v for (_k, _v) in _initial_values.items() if not _k.startswith("_comment")}
        self.set_attributes_with_keys(_initial_values)
