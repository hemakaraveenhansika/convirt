from train import SimCLR
import yaml
from dataloader.dataset_wrapper import DataSetWrapper

def main():
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    
    dataset = DataSetWrapper(config['batch_size'], **config['dataset'])

    simclr = SimCLR(dataset, config)

    simclr.train()

    # simclr.test()


if __name__ == "__main__":
    main()
