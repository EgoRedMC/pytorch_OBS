import torch as t


class Net(t.nn.Module):
    def __init__(self, dim, device, n_layers, n_count, act):
        super(Net, self).__init__()
        self.dim = dim
        self.device = device
        self.k = n_count
        self.l = n_layers
        self.act = act
        self.l1 = t.nn.Linear(dim, n_count).to(device)
        if n_layers >= 2:
            self.l2 = t.nn.Linear(n_count, n_count).to(device)
            if n_layers >= 3:
                self.l3 = t.nn.Linear(n_count, n_count).to(device)
                if n_layers >= 4:
                    self.l4 = t.nn.Linear(n_count, n_count).to(device)
        self.out = t.nn.Linear(n_count, 1).to(device)
        self.out_act = t.nn.Sigmoid().to(device)
        if act == "sigm":
            self.act1 = t.nn.Sigmoid().to(device)
            self.act2 = t.nn.Sigmoid().to(device)
            self.act3 = t.nn.Sigmoid().to(device)
            self.act4 = t.nn.Sigmoid().to(device)
            return
        if act == "relu":
            self.act1 = t.nn.ReLU().to(device)
            self.act2 = t.nn.ReLU().to(device)
            self.act3 = t.nn.ReLU().to(device)
            self.act4 = t.nn.ReLU().to(device)
            return
        if act == "lrelu":
            self.act1 = t.nn.LeakyReLU().to(device)
            self.act2 = t.nn.LeakyReLU().to(device)
            self.act3 = t.nn.LeakyReLU().to(device)
            self.act4 = t.nn.LeakyReLU().to(device)
            return
        if act == "thr":
            self.act1 = t.nn.Threshold(threshold=0.5, value=1).to(device)
            self.act2 = t.nn.Threshold(threshold=0.5, value=1).to(device)
            self.act3 = t.nn.Threshold(threshold=0.5, value=1).to(device)
            self.act4 = t.nn.Threshold(threshold=0.5, value=1).to(device)
            return
        print("activation function is not set")

    def forward(self, x):
        x = self.l1(x)
        x = self.act1(x)
        if self.l >= 2:
            x = self.l2(x)
            x = self.act2(x)
            if self.l >= 3:
                x = self.l3(x)
                x = self.act3(x)
                if self.l >= 4:
                    x = self.l4(x)
                    x = self.act4(x)
        x = self.out(x)
        x = self.out_act(x)
        return x

    def forward_round(self, x):
        x = self.forward(x)
        return t.round(x)

    def print_data(self):
        for name, p in self.named_parameters():
            print(name)
            print(p)

    def print_info(self):
        print("Число внутренних слоев: " + str(self.l))
        print("Число нейронов в каждом слое: " + str(self.k))
        print("Функция активации: " + self.act)


def copy_net(n):
    # функция используется для прореживания до обучения
    # (а поскольку прореживание осуществляется после обучения, то прореживается копия сети, которая потом и обучается)
    new_n = Net(n.dim, n.device, n.l, n.k, n.act)
    new_n.l1.load_state_dict(n.l1.state_dict())
    if n.l >= 2:
        new_n.l2.load_state_dict(n.l2.state_dict())
        if n.l >= 3:
            new_n.l3.load_state_dict(n.l3.state_dict())
    new_n.out.load_state_dict(n.out.state_dict())
    return new_n
