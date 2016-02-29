#!/usr/bin/env python

# Machine Learning 445
# HW 2: Neural Networks
# Katie Abrahams, abrahake@pdx.edu
# 1/28/16


class letter:
    """Letter entity class
    contains alphabet letter value and dataset attributes"""
    # value = None # alphabet letter
    # attributes = []  # 16 numerical attributes for a letter from the data set
    # target = 0.0

    def __init__(self, input):
        """
        :param input: list containing letter and attributes
        normalize attributes
        """
        self.value = input[:1]
        # map(func, iterable) applies func to every item of iterable and return a list of the results
        self.attributes = map(float, input[1:])
        # bias for input is always 1, hard code it here
        # to make matrix multiplication simpler
        self.bias_input = [1.0]
        # list of bias inputs + attributes for filling matrices
        self.bias_input_plus_attributes = self.bias_input+self.attributes
        # neuron target
        self.target = 0.0