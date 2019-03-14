"""Config File"""

"""Dataset Path"""
PATH = "./data/goblet.txt"

"""Training Parameters"""
LEARNING_RATE = 0.1
EPSILON = 1e-7
BATCH_SIZE = 25
EPOCH = 10

"""Model Parameters"""
M = 100
SIG = 0.01

"""Synthesize Text"""
SHORT_TEXT_LENGTH = 200
SYN_STEP = 10000
SYN_BOUND = 100002
PASSAGE_LENGTH = 1000

"""Save Path"""
SAVE_PATH = './results/'