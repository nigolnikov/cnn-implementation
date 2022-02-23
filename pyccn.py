from abc import ABC, abstractmethod
import torch
from utils import hp, acc_loss
torch.manual_seed(42)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Layer(ABC):
    @abstractmethod
    def forward_pass(self, _input):
        pass

    @abstractmethod
    def backward_pass(self, d_output):
        pass

    @abstractmethod
    def step(self, eta, mu):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def validation(self):
        pass

    @abstractmethod
    def to(self, _device):
        pass


class Conv2D(Layer):
    def __init__(self, input_channels, output_channels,
                 kernel_height, kernel_width, padding):
        # He init
        self.weights =\
            torch.randn(1, output_channels, input_channels,
                        kernel_height, kernel_width) *\
            torch.sqrt(torch.Tensor([1.0 / (input_channels * output_channels *
                                     kernel_height * kernel_width)]))
        self.bias = torch.zeros(1, output_channels, 1, 1, 1)
        self.padding = padding

        self.train = True
        self.input = None
        self.output_shape = None

        self.device = None

        self.weights_grad = torch.zeros_like(self.weights)
        self.prev_weights_grad = torch.zeros_like(self.weights)
        self.bias_grad = torch.zeros_like(self.bias)
        self.prev_bias_grad = torch.zeros_like(self.bias)

    def to(self, _device):
        self.device = _device
        self.weights = self.weights.to(_device)
        self.bias = self.bias.to(_device)
        self.weights_grad = self.weights_grad.to(_device)
        self.bias_grad = self.bias_grad.to(_device)
        self.prev_weights_grad = self.prev_weights_grad.to(_device)
        self.prev_bias_grad = self.prev_bias_grad.to(_device)

    def pad(self, tensor, pad, zeros=False, rnb=False):
        _input = tensor.clone().detach().to(self.device)
        if pad == 0:
            return _input
        input_shape = tuple(_input.size())

        if rnb & zeros:
            output = torch.zeros(input_shape[:3] + (input_shape[3] + pad,) +
                                 (input_shape[4] + pad,)).to(self.device)
            output[:, :, :, :-pad, :-pad] += _input  # center
            return output

        output = torch.zeros(
            input_shape[:3] + (input_shape[3] + 2 * pad,) +
            (input_shape[4] + 2 * pad,), dtype=_input.dtype
        ).to(self.device)
        output[:, :, :, pad:-pad, pad:-pad] += _input                 # center
        if zeros:
            return output
        output[:, :, :, :pad, pad:-pad] += _input[:, :, :, :1, :]     # top
        output[:, :, :, -pad:, pad:-pad] += _input[:, :, :, -1:, :]   # bottom
        output[:, :, :, pad:-pad, :pad] += _input[:, :, :, :, :1]     # left
        output[:, :, :, pad:-pad, -pad:] += _input[:, :, :, :, -1:]   # right
        output[:, :, :, :pad, :pad] += _input[:, :, :, :1, :1]        # top l
        output[:, :, :, :pad, -pad:] += _input[:, :, :, :1, -1:]      # top r
        output[:, :, :, -pad:, :pad] += _input[:, :, :, -1:, :1]      # bot. l
        output[:, :, :, -pad:, -pad:] += _input[:, :, :, -1:, -1:]    # bot. r
        return output

    def forward_pass(self, _input):
        if self.train:
            self.input = _input.clone().detach().to(self.device)

        input_shape = tuple(_input.size())
        filter_shape = tuple(self.weights.size())
        batch_size = input_shape[0]
        input_height, input_width = input_shape[3:5]
        output_channels = filter_shape[1]
        filter_height, filter_width = filter_shape[3:5]

        output_height = input_height - filter_height + 2 * self.padding + 1
        output_width = input_width - filter_width + 2 * self.padding + 1

        self.output_shape = (batch_size, output_channels,  1,
                             output_height, output_width)

        _input = self.pad(_input, self.padding)

        output = torch.zeros(batch_size, output_channels,  1,
                             output_height, output_width).to(self.device)

        for i in range(output_height):
            for j in range(output_width):
                output[:, :, :, i, j] =\
                    (_input[:, :, :, i: i + filter_height,
                     j: j + filter_width] *
                     self.weights).sum(dim=2, keepdim=True).sum(dim=(3, 4))
        output += self.bias
        return output.permute(0, 2, 1, 3, 4)

    def backward_pass(self, d_output):
        d_output_shape = tuple(d_output.size())
        d_output_shape = self.output_shape
        d_output = d_output.view(torch.Size(d_output_shape))
        filter_shape = tuple(self.weights.size())
        d_input_shape = tuple(self.input.size())

        batch_size = d_output_shape[0]

        d_output_height, d_output_width = d_output_shape[3:5]
        filter_height, filter_width = filter_shape[3:5]
        d_input_channels, d_input_height, d_input_width = d_input_shape[2 : 5]

        backprop_padding = int((d_input_height -
                            d_output_height +
                            filter_height - 1) / 2)
        backprop_padding = filter_height - self.padding -1

        grad_padding = int((filter_height -
                        d_input_height +
                        d_output_height - 1) / 2)

        self.prev_weights_grad =\
            self.weights_grad.clone().detach().to(self.device)
        self.prev_bias_grad = self.bias_grad.clone().detach().to(self.device)

        self.weights_grad =\
            torch.zeros(tuple(self.weights.size())).to(self.device)
        self.bias_grad = torch.zeros(tuple(self.bias.size())).to(self.device)


        # Backpropagate error
        backprop_resized_d_output = self.pad(d_output, backprop_padding)
        d_input = torch.zeros(batch_size, d_input_channels, 1,
                              d_input_height, d_input_width).to(self.device)

        # Flip kernels and reverse channels
        backprop_weights =\
            torch.rot90(self.weights, 2, (3, 4)).permute(0, 2, 1, 3, 4)

        backprop_resized_d_output =\
            backprop_resized_d_output.permute(0, 2, 1, 3, 4)

        for i in range(d_input_height):
            for j in range(d_input_width):
                d_input[:, :, :, i, j] =\
                    (backprop_resized_d_output[:, :, :, i:i + filter_height,
                     j:j + filter_width] *
                     backprop_weights).sum(dim=(3, 4)).sum(2, keepdim=True)
        d_input = d_input.permute(0, 2, 1, 3, 4)

        del backprop_resized_d_output


        # Compute gradients
        # Reverse channels
        grad_d_output = d_output

        grad_resized_d_input = self.pad(self.input, grad_padding)

        for i in range(filter_height):
            for j in range(filter_width):
                self.weights_grad[:, :, :, i:i + 1, j:j + 1] =\
                    (grad_resized_d_input[:, :, :, i:i + d_output_height,
                     j:j + d_output_width] *
                     grad_d_output).sum(dim=(3, 4), keepdim=True).sum(dim=0, keepdim=True)
        self.bias_grad = grad_d_output.sum(dim=(0, 2, 3, 4), keepdim=True)

        return d_input

    def step(self, eta, mu):
        self.weights -= eta * self.weights_grad
        self.bias -= eta * self.bias_grad

    def train(self):
        self.train = True

        self.weights_grad = torch.zeros_like(self.weights)
        self.prev_weights_grad = torch.zeros_like(self.weights)
        self.bias_grad = torch.zeros_like(self.bias)
        self.prev_bias_grad = torch.zeros_like(self.bias)

    def validation(self):
        self.train = False
        self.input = None

        self.weights_grad = None
        self.prev_weights_grad = None
        self.bias_grad = None
        self.prev_bias_grad = None


class ReLU(Layer):
    def __init__(self):
        self.train = True
        self.ones = None

    def to(self, _device):
        pass

    def forward_pass(self, _input):
        if self.train:
            self.ones = (_input >= 0.0).int()

        _input[_input < 0.0] = 0.0
        return _input

    def backward_pass(self, d_input):
        return d_input * self.ones.float()

    def step(self, eta, mu):
        pass

    def train(self):
        self.train = True

    def validation(self):
        self.train = False
        self.ones = None


class MaxPool2D(Layer):
    def __init__(self):
        self.device = None
        self.max_map = None
        self.train = True
        self.output_shape = None

    def to(self, _device):
        self.device = _device

    def forward_pass(self, _input):
        input_shape = tuple(_input.size())
        self.max_map = torch.zeros(input_shape).int().to(self.device)
        original_input = _input.clone().detach().to(self.device)
        # Zero padding in odd case
        if input_shape[3] % 2:
            _input = Conv2D.pad(_input, pad=1, zeros=True, rnb=True)
            input_shape = tuple(_input.size())
        output = torch.zeros(input_shape[:3] +
                             (int(input_shape[3] / 2),) +
                             (int(input_shape[4] / 2),)).to(self.device)
        output_shape = tuple(output.size())
        for i in range(output_shape[3]):
            for j in range(output_shape[4]):
                output[:, :, :, i:i + 1, j:j + 1], will_see =\
                    _input[:, :, :, 2 * i:2 * (i + 1), 2 * j:2 * (j + 1)].contiguous().view(
                        input_shape[:3] + (1, 4)
                    ).max(dim=4, keepdim=True)
                self.max_map[:, :, :, 2 * i:2 * (i + 1), 2 * j:2 * (j + 1)] =\
                    original_input[
                        :, :, :, 2 * i:2 * (i + 1), 2 * j:2 * (j + 1)
                    ] == output[:, :, :, i:i + 1, j:j + 1]
        self.output_shape = output.size()
        return output

    def backward_pass(self, d_output):
        d_output = d_output.view(self.output_shape)
        d_input_shape = tuple(self.max_map.size())
        d_input = torch.zeros(d_input_shape).to(self.device)
        d_output_shape = tuple(d_output.size())

        self.max_map = self.max_map.float()

        for i in range(d_output_shape[3]):
            for j in range(d_output_shape[4]):
                d_input[:, :, :, 2 * i:2 * (i + 1), 2 * j:2 * (j + 1)] =\
                    self.max_map[
                        :, :, :, 2 * i:2 * (i + 1), 2 * j:2 * (j + 1)
                    ] * d_output[:, :, :, i:i + 1, j:j + 1]
        return d_input

    def step(self, eta, mu):
        pass

    def train(self):
        self.train = True

    def validation(self):
        self.train = False
        self.max_map = None


class Dense(Layer):
    def __init__(self, input_layer, next_layer):
        self.weights = torch.randn(input_layer, next_layer) *\
                       (1.0 / input_layer) ** 0.5
        self.bias = torch.zeros(1, next_layer)

        self.weights_grad = torch.zeros_like(self.weights)
        self.bias_grad = torch.zeros_like(self.bias)

        self.device = None
        self.train = True
        self.input = None

    def to(self, _device):
        self.device = _device
        self.weights.to(self.device)
        self.bias.to(self.device)
        self.weights_grad.to(self.device)
        self.bias_grad.to(self.device)

    def forward_pass(self, _input):
        self.input = _input.view(tuple(_input.size())[0], -1)
        self.weights = self.weights.to(self.device)
        self.bias = self.bias.to(self.device)
        return self.input.mm(self.weights) + self.bias

    def backward_pass(self, d_output):
        d_output = d_output.view(tuple(d_output.size())[0], -1)
        d_input = d_output.mm(self.weights.transpose(0, 1))
        self.weights_grad = self.input.transpose(0, 1).mm(d_output)
        self.bias_grad = d_output.sum(dim=0, keepdim=True)
        return d_input

    def step(self, eta, mu):
        self.weights -= eta * self.weights_grad
        self.bias -= eta * self.bias_grad

    def train(self):
        self.train = True
        self.weights_grad = torch.zeros_like(self.weights).to(self.device)
        self.bias_grad = torch.zeros_like(self.bias).to(self.device)

    def validation(self):
        self.train = False
        self.weights_grad = None
        self.bias_grad = None
        self.input = None


class CrossEntropyLoss:
    def __init__(self):
        self.none = None

    def __call__(self, scores, ground_truth):
        # Softmax loss
        exp_scores = torch.exp(scores)
        probabilities = exp_scores / exp_scores.sum(dim=1, keepdim=True)

        n = len(ground_truth)

        correct_log_probabilities =\
            -torch.log(probabilities[range(n), ground_truth.long()])

        data_loss = correct_log_probabilities.sum() / n

        reg_loss = 0.0

        loss = data_loss + reg_loss

        d_scores = torch.zeros_like(probabilities)

        d_scores = probabilities.clone().detach()
        d_scores[range(n), ground_truth] -= 1

        d_scores /= n
        return d_scores, loss


class Net(object):
    def __init__(self, *layers):
        self.layers = layers
        self.train = True
        self.loss = None
        self.device = None

    def to(self, _device):
        self.device = _device
        for layer in self.layers:
            layer.to(self.device)

    def __call__(self, x, gt=None):
        if self.train:
            output = self.forward_pass(x)
            loss = criterion(output, gt)
            self.backward_pass(loss)
            return output, loss
        else:
            return self.forward_pass(x)

    def forward_pass(self, _input):
        x = _input.clone().detach()
        for layer in self.layers:
            x = layer.forward_pass(x)
        return x

    def backward_pass(self, d_output):
        d_x = d_output.clone().detach()
        for layer in self.layers[::-1]:
            d_x = layer.backward_pass(d_x)
        return d_x

    def train(self):
        self.train = True
        for layer in self.layers:
            layer.train()

    def eval(self):
        self.train = False
        for layer in self.layers:
            layer.validation()

    def step(self, eta, mu):
        for layer in self.layers:
            layer.step(eta, mu)

    def save(self):
        pass

    def load(self):
        pass


batch_size = 128

DATA_PATH = "data/hpdset"
data_dict, train_sampler, val_sampler = hp.load(DATA_PATH, _32=True)
train_loader = torch.utils.data.DataLoader(
    data_dict['train_full'], batch_size=batch_size,
    sampler=val_sampler, num_workers=1
)
validation_loader = torch.utils.data.DataLoader(
    data_dict['train_full'], batch_size=batch_size,
    sampler=val_sampler, num_workers=1
)

model = Net(
    Conv2D(1, 4, 3, 3, 1),
    ReLU(),
    MaxPool2D(),
    Conv2D(4, 8, 3, 3, 1),
    ReLU(),
    MaxPool2D(),
    Conv2D(8, 16, 3, 3, 1),
    ReLU(),
    MaxPool2D(),
    Dense(16 * 4 * 4, 128),
    ReLU(),
    Dense(128, 33)
)
model.to(device)

criterion = CrossEntropyLoss()
lr = 1e-1
epochs = 1
print_every = 20
loss_history = []
train_accuracy_history = []
validation_accuracy_history = []


for epoch in range(epochs):
    running_loss = 0.0
    running_total = 0
    running_correct = 0
    for i, (images, labels) in enumerate(train_loader, 0):
        images = images.to(device)
        labels = labels.to(device)
        images = images.view(-1, 1, 1, 32, 32)

        x = model.forward_pass(images)
        d_x, loss = criterion(x, labels)
        _ = model.backward_pass(d_x)
        model.step(lr, 1e-3)

        _, predicted = torch.max(x, dim=1)
        running_correct += (torch.sum(predicted == labels)).item()
        running_total += len(x)
        running_loss += loss

        if (i + 1) % print_every == 0:
            train_accuracy = running_correct / running_total
            running_correct = 0
            running_total = 0
            loss = running_loss / print_every
            running_loss = 0.0
            loss_history.append(loss)
            train_accuracy_history.append(train_accuracy)

            val_running_correct = 0
            val_running_total = 0
            for v_i, (v_images, v_labels) in enumerate(validation_loader, 0):
                v_images = v_images.to(device)
                v_labels = v_labels.to(device)
                v_images = v_images.view(-1, 1, 1, 32, 32)

                x = model.forward_pass(v_images)
                _, predicted = torch.max(x, dim=1)
                val_running_correct += (torch.sum(predicted == v_labels)).item()
                val_running_total += len(x)
            val_accuracy = val_running_correct / val_running_total
            validation_accuracy_history.append(val_accuracy)

            print(
                ('[%i/%i][Train loss: %.2f]' +
                 '[Train accuracy: %.2f][Val accuracy: %.2f]') %
                (i + 1, len(validation_loader), loss,
                 train_accuracy, val_accuracy)
            )
    # lr *= 0.999


running_correct = 0
running_total = 0
for i, (images, labels) in enumerate(validation_loader, 0):
    images = images.to(device)
    labels = labels.to(device)
    images = images.view(-1, 1, 1, 32, 32)

    x = model.forward_pass(images)

    _, predicted = torch.max(x, dim=1)
    running_correct = (torch.sum(predicted == labels)).item()
    running_total = len(x)

acc_loss.save(loss_history, train_accuracy_history,
              validation_accuracy_history, 'train_stats.png')

test_accuracy = running_correct / running_total
print('TEST ACCURACY: %.3f' % test_accuracy)
