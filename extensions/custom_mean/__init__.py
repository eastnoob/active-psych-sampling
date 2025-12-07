#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Custom Mean Functions for AEPsych

This module contains custom mean functions that can be used with AEPsych models.

Available mean functions:
- CustomBaseGPPriorMean: Fixed BaseGP mean (no learnable parameters)
- CustomMeanWithOffsetPrior: BaseGP mean + learnable offset (1 parameter)
"""

from .custom_basegp_prior_mean import CustomBaseGPPriorMean, CustomMeanWithOffsetPrior

__all__ = ["CustomBaseGPPriorMean", "CustomMeanWithOffsetPrior"]
