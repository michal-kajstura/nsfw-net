from nsfw.training.experiment import create_loaders, Experiment

path = '/home/michal/data/zpi/images/'

train_loader, val_loader = create_loaders(path)

experiment = Experiment()
experiment.run(train_loader, val_loader)

