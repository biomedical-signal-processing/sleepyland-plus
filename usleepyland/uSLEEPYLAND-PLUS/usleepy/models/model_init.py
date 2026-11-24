"""
Functions for initializing models from build hyperparameters and loading of
parameters.
"""

import logging
import os
import tensorflow as tf
from usleepy import models
from usleepy.models.utils import get_best_model, get_last_model

logger = logging.getLogger(__name__)


def init_model(build_hparams, clear_previous=False):
    """
    From a set of hyperparameters 'build_hparams' (dict) initializes the
    model specified under build_hparams['model_class_name'].

    Typically, this function is not called directly, but used by the
    higher-level 'initialize_model' function.

    Args:
        build_hparams:  A dictionary of model build hyperparameters
        clear_previous: Clear previous tf sessions

    Returns:
        A tf.keras Model instance
    """
    if clear_previous:
        tf.keras.backend.clear_session()
    # Build new model of the specified type
    cls_name = build_hparams["model_class_name"]
    logger.info(f"Creating new model of type '{cls_name}'")
    return models.__dict__[cls_name](**build_hparams)


def load_from_file(model, file_path, by_name=True):
    """
    Load parameters from file 'file_path' into model 'model'.

    Args:
        model:      A tf.keras Model instance
        file_path:  A path to a parameter file (h5 format typically)
        by_name:    Load parameters by layer names instead of order (default).
    """
    model.load_weights(file_path, by_name=by_name)
    logger.info(f"Loading parameters from: {file_path}")


def init_and_load_model(hparams, weights_file, clear_previous=False, by_name=True):
    """
    Initializes a model according to hparams. Then sets its parameters from
    the parameters in h5 file 'weights_file'.

    Args:
        hparams:        A YAMLHparams object of hyperparameters
        weights_file:   A path to a h5 parameter file to load
        clear_previous: Clear previous keras session before initializing new model graph.
        by_name:        Load parameters by layer names instead of order (default).

    Returns:
        A tf.keras Model instance
    """
    model = init_model(build_hparams=hparams["build"], clear_previous=clear_previous)
    load_from_file(model, weights_file, by_name=by_name)
    return model


def init_and_load_best_model(hparams, model_dir, clear_previous=False, by_name=True):
    """
    Initializes a model according to hparams. Then finds the best model in
    model_dir and loads it (see usleepy.utils.get_best_model).

    Args:
        hparams:        A YAMLHparams object of hyperparameters
        model_dir:      A path to the directory that stores model param files
        clear_previous: Clear previous keras session before initializing new model graph.
        by_name:        Load parameters by layer names instead of order (default).

    Returns:
        A tf.keras Model instance
        The file name of the parameter file that was loaded
    """
    model = init_model(hparams["build"], clear_previous=clear_previous)
    model_path = get_best_model(model_dir)
    load_from_file(model, model_path, by_name=by_name)
    model_file_name = os.path.split(model_path)[-1]
    return model, model_file_name