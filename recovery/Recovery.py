import numpy as np

class Recovery():
    def __init__(self):
        self.env = None
        self.env_name = ''
        self.model = None
        self.latent = None

    def setEnvironment(self, env):
        self.env = env

    def run_model(self, time_series, original_decision):
        return original_decision

