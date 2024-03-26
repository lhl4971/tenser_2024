from tqdm import trange
import torch

class ModelTraining:
    def __init__(self, train_loader, train_size, val_loader=None, val_size=0):
        self.train_loader = train_loader
        self.train_size = train_size
        self.val_loader = val_loader
        self.val_size = val_size
        self.train_loss_list = []
        self.val_loss_list = []


    def set_network(self, net):
        self.net = net
    
    def set_criterion(self, criterion):
        self.criterion = criterion

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def train(self, epoch_iter, val=False, print_loss=False):
        for epoch in trange(epoch_iter):
            # Training model
            self.net.train()
            train_running_loss = 0.0

            for i, data in enumerate(self.train_loader, 0):
                # Get input data
                inputs, labels = data
                # Gradient zero
                self.optimizer.zero_grad()
                # Forward + Backward + Optimize
                outputs = self.net(inputs)
                train_loss = self.criterion(outputs, labels)
                train_loss.backward()
                self.optimizer.step()
                train_running_loss += train_loss.item()

            # Calculate the loss and accuracy on train_set
            avg_train_loss = train_running_loss / self.train_size * self.train_loader.batch_size
            self.train_loss_list.append(avg_train_loss)

            if val:
                # Test on the testing set
                self.net.eval()
                val_running_loss = 0
                # Disable the Gradient
                with torch.no_grad():
                    for data in self.val_loader:
                        # Get input data
                        inputs, labels = data
                        # Forward
                        outputs = self.net(inputs)
                        val_loss = self.criterion(outputs, labels)
                        val_running_loss += val_loss.item() 
                # Calculate the loss and accuracy on test_set
                avg_val_loss = val_running_loss / self.val_size * self.val_loader.batch_size
                self.val_loss_list.append(avg_val_loss)

                if print_loss:
                    # Print statistics
                    print(f'{epoch + 1}\t{avg_train_loss:.4f}\t\t{avg_val_loss:.4f}')
    
    def get_train_loss_list(self):
        return self.train_loss_list
    
    def get_val_loss_list(self):
        return self.val_loss_list
    
    def get_model(self):
        return self.net

