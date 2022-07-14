




class BYOLATrainer:

    def __init__(self, cfg, wandb_run, logger):
        self.cfg = cfg
        self.wandb_run = wandb_run 
        self.logger = logger 
        