import torch

class DefaultConfig(object):
    """
    Default configuration class for PyGraphsage model.
    """

    model = 'PyGraphsage'  # Default model name
    use_gpu = False  # Default GPU usage flag
    device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'  # Default device (GPU if available, else CPU)
    load_model_path = None  # Default path to load a pre-trained model
    network = 'cora'  # Default network dataset
    batch_size = 8  # Default batch size
    num_workers = 3  # Default number of data loading workers
    max_epoch = 10  # Default maximum number of training epochs
    lr = 0.005  # Default learning rate
    lr_decay = 0.9  # Default learning rate decay factor
    weight_decay = 1e-5  # Default weight decay for optimization
    train_rate = 0.8  # Default training set rate
    val_rate = 0.1  # Default validation set rate
    dropout = 0.5  # Default dropout rate

    def _parse(self, kwargs):
        """
        Update config parameters based on the dictionary kwargs.
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribute %s" % k)
            setattr(self, k, v)

        # Update device based on GPU availability and user's preference
        self.device = torch.device('cuda') if self.use_gpu and torch.cuda.is_available() else torch.device('cpu')

        # Print user configuration
        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, getattr(self, k))

# Instantiate DefaultConfig object with default values
opt = DefaultConfig()
