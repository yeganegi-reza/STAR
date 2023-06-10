import torch.optim as optim


class Optimizer:
    def __init__(self, params, args):        
        self.optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.w_dc)
        self.set_scheduler(args)
        
    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()
        
    def set_scheduler(self, args):
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                   step_size=args.lr_dc_step,
                                                   gamma=args.lr_dc)
    def step_scheduler(self):
        self.scheduler.step()
