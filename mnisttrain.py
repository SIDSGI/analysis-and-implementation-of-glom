import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

import matplotlib.pyplot as plt

batch_size_train = 60
batch_size_test = 60
learning_rate = 0.01
log_interval = 10

num_vectors = 4
len_vectors = 10
img_height = 28
img_width = 28
win_size = 3
epsilon = .7
epochs = 6000
steps = 30


train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./data/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./data/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)

def data_to_state(example_data,batch_size):
    temp_example_data = torch.reshape(example_data,(batch_size,img_height,img_width))
    temp_inp = [temp_example_data for i in range(10)]
    temp_inp_data = torch.stack((temp_inp),dim = 3)
    return temp_inp_data.to(device)


def targets_to_state(example_targets,batch_size):
    temp_out_state = torch.nn.functional.one_hot(example_targets,num_classes=10).repeat(1,img_height*img_width)
    temp_out_state = temp_out_state.view((batch_size,img_height,img_width,10))
    return temp_out_state.float().to(device)


def init_state(batch_size,img_height,img_width,num_vectors,len_vectors):
    state = torch.rand((batch_size,img_height,img_width,num_vectors,len_vectors))*.1
    return state.to(device)


class model(nn.Module):
    def __init__(self, num_inp, num_out):
        super(model, self).__init__()
        self.Q1 = nn.Linear(num_inp, num_out)
        self.K1 = nn.Linear(num_out, num_out)
        self.V1 = nn.Linear(num_out, num_out)

        self.m = nn.Dropout(p=0.1)

        self.act = nn.LeakyReLU()
        self.act1 = nn.Tanh()
        self.act3 = nn.GELU()

    def forward(self, x):
        Q = self.act1(self.Q1(self.m(x)))
        K = self.act1(self.K1(self.m(Q))) + Q
        V = self.act1(self.V1(self.m(K))) + Q + K

        return V * .01


def get_layer_attention(center_matrix, roll_matrix, after_tri):
    # dot product vectors to find similarity of adjacent vectors
    after_mul = torch.matmul(center_matrix, roll_matrix.permute((0, 1, 2, 4, 3)))
    after_diag = torch.diagonal(after_mul, offset=0, dim1=3, dim2=4)

    # multiply vectors by lambda matrix to find full attention numbers
    after_eps = torch.matmul(after_diag, after_tri)

    # stack full attention numbers so that each vector gets its proper attention number
    after_sim = torch.stack([after_eps for i in range(len_vectors)], dim=3).permute((0, 1, 2, 4, 3))

    # multiply each vector by the attention numbers to complete the attention step
    full_vec_dis = center_matrix * after_sim.detach()

    return full_vec_dis


def compute_all(bottom_up_model_list, top_down_model_list, layer_att_model_list, state, len_vectors, num_vectors,
                batch_size):
    # shift state to in 9 directions along the x and y plane
    roll1 = torch.roll(state, shifts=(-1, -1), dims=(1, 2)).to(device)
    roll2 = torch.roll(state, shifts=(-1, 0), dims=(1, 2)).to(device)
    roll3 = torch.roll(state, shifts=(-1, 1), dims=(1, 2)).to(device)
    roll4 = torch.roll(state, shifts=(0, -1), dims=(1, 2)).to(device)
    roll5 = torch.roll(state, shifts=(0, 0), dims=(1, 2)).to(device)
    roll6 = torch.roll(state, shifts=(0, 1), dims=(1, 2)).to(device)
    roll7 = torch.roll(state, shifts=(1, -1), dims=(1, 2)).to(device)
    roll8 = torch.roll(state, shifts=(1, 0), dims=(1, 2)).to(device)
    roll9 = torch.roll(state, shifts=(1, 1), dims=(1, 2)).to(device)
    roll_list = [roll1, roll2, roll3, roll4, roll5, roll6, roll7, roll8, roll9]

    eps_matrix = epsilon ** torch.arange(start=1, end=num_vectors + 1)
    try_roll = [torch.roll(eps_matrix, shifts=(i), dims=(0)) for i in range(eps_matrix.shape[0])]
    try_roll = torch.stack(try_roll)
    after_tri = torch.triu(try_roll, diagonal=0).T.to(device)

    att_list = []
    for roll in roll_list:
        att_list.append(get_layer_attention(roll, roll5, after_tri))

    # concatenate vectors so that att_list contains the state and every adjacent vector on the same vector level
    att_list = torch.cat(roll_list, dim=4)


    delta = [torch.zeros((batch_size * img_height * img_width, len_vectors)).to(device) for i in range(num_vectors)]
    for i in range(num_vectors):
        if (i < num_vectors - 2):
            top_down_temp = top_down_model_list[i](torch.reshape(att_list[:, :, :, i + 2, :], (-1, len_vectors * 9)))
            delta[i + 1] = delta[i + 1] + top_down_temp
        if (i < num_vectors - 1):
            bottom_up_temp = bottom_up_model_list[i](torch.reshape(att_list[:, :, :, i, :], (-1, len_vectors * 9)))
            att_layer_temp = layer_att_model_list[i](torch.reshape(att_list[:, :, :, i + 1, :], (-1, len_vectors * 9)))
            delta[i + 1] = delta[i + 1] + bottom_up_temp + att_layer_temp

    # format delta so that delta and state can be added together
    delta = torch.stack(delta, dim=1)  # .permute(0,2,1)#.permute(2,0,1)
    delta = torch.reshape(delta, (batch_size, img_height, img_width, num_vectors, len_vectors))
    return delta


bottom_up_model_list = [model(9*len_vectors,len_vectors).cuda() for i in range(num_vectors-1)]
top_down_model_list= [model(9*len_vectors,len_vectors).cuda() for i in range(num_vectors-2)]
layer_att_model_list = [model(9*len_vectors,len_vectors).cuda() for i in range(num_vectors-1)]

# create parameter list of every model to feed into optimizer
param_list = []
for i in range(num_vectors):
    if (i < num_vectors - 2):
        param_list = param_list + list(top_down_model_list[i].parameters())
    if (i < num_vectors - 1):
        param_list = param_list + list(bottom_up_model_list[i].parameters())
        param_list = param_list + list(layer_att_model_list[i].parameters())

optimizer = torch.optim.Adam(param_list, lr=learning_rate)
mse = nn.MSELoss()

examples = enumerate(train_loader)
batch_idx, (example_data, example_targets) = next(examples)
train_loss = []
try:
    for epoch in range(epochs):
        optimizer.zero_grad()

        # in case StopIteration error is raised
        try:
            batch_idx, (example_data, example_targets) = next(examples)
        except StopIteration:
            examples = enumerate(train_loader)
            batch_idx, (example_data, example_targets) = next(examples)

        # initialize state
        state = init_state(batch_size_train, img_height, img_width, num_vectors, len_vectors)

        # put current batches into state
        state[:, :, :, 0, :] = data_to_state(example_data, batch_size_train)
        state1 = torch.clone(state)
        state2 = torch.clone(state)
        state3 = torch.clone(state)
        for step in range(steps):

            delta = compute_all(bottom_up_model_list, top_down_model_list, layer_att_model_list, state, len_vectors,
                                num_vectors, batch_size_train)

            state = state + delta + .0001 * torch.rand((state.shape)).to(device)

            # add first state to state in the middle of the steps (allows for RESNET type gradient backprop)
            if (step % int(steps / 2) == 0):
                state = state + state1 * .1
                state1 = torch.clone(state)

            if (step % int(steps / 4) == 0):
                state = state + state2 * .1
                state2 = torch.clone(state)

            if (step % int(steps / 8) == 0):
                state = state + state3 * .1
                state3 = torch.clone(state)

        state = state + state1 * .1 + state2 * .1 + state3 * .1
        # get loss
        pred_out = state[:, :, :, -1]
        targ_out = targets_to_state(example_targets, batch_size_train)
        loss = mse(pred_out, targ_out)
        loss.backward()
        optimizer.step()
        train_loss.append(loss)
        print("Epoch: {}/{}  Loss: {}".format(epoch, epochs, loss))


except:
    print(train_loss)
    fig = plt.figure()
    plt.plot(range(len(train_loss)),train_loss, color='blue')
    plt.xlabel('number of training epochs')
    plt.ylabel('loss')
    plt.savefig("train.png")

print(train_loss)
fig = plt.figure()
plt.plot(range(len(train_loss)),train_loss, color='blue')
plt.xlabel('number of training epochs')
plt.ylabel('loss')
plt.savefig("train.png")
