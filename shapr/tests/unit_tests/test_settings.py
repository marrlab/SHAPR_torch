from shapeae import settings
import pytest
import json
from pathlib import Path

def test_settings_default_values():
    with open(Path(__file__).parents[2].joinpath("default_params.json"), 'r') as f:
        default_params = json.load(f)
    
    for (_k, _v) in default_params.items():
        assert getattr(settings, _k) == _v

def test_setting_values_with_object_attributes():
    _params = dict(
        path= "path1",
        result_path = "path2",
        pretrained_weights_path = "path3",
        random_seed = 42,
        batch_size = 10,
        epochs_ShapeAE = 100,
        epochs_cShapeAE = 1000,
    )
    settings.path = _params['path']
    settings.result_path = _params['result_path']
    settings.pretrained_weights_path = _params['pretrained_weights_path']
    settings.random_seed = _params['random_seed']
    settings.batch_size = _params['batch_size']
    settings.epochs_ShapeAE = _params['epochs_SHAPR']
    settings.epochs_cShapeAE = _params['epochs_cSHAPR']

    assert settings.path == _params['path']
    assert settings.result_path == _params['result_path']
    assert settings.pretrained_weights_path == _params['pretrained_weights_path']
    assert settings.random_seed == _params['random_seed']
    assert settings.batch_size == _params['batch_size']
    assert settings.epochs_ShapeAE == _params['epochs_SHAPR']
    assert settings.epochs_cShapeAE == _params['epochs_cSHAPR']

def test_setting_read_json():
    settings.read_json(Path(__file__).parents[1].joinpath("test_data/json", "settings_testing_params.json"))


    assert settings.path == "path1_json"
    assert settings.result_path == "path2_json"
    assert settings.pretrained_weights_path == "path3_json"
    assert settings.random_seed == 21
    assert settings.batch_size == 5
    assert settings.epochs_ShapeAE == 50
    assert settings.epochs_cShapeAE == 500


