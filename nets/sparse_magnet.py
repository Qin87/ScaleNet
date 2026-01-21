import numpy as np
import torch, math
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor

from nets.hermitian import hermitian_decomp_sparse, cheb_poly_sparse, QinDirect_hermitian_decomp_sparse


# from torch.nn import MultiheadAttention


def process(mul_L_real, mul_L_imag, weight, X_real, X_imag):
    data = torch.spmm(mul_L_real, X_real)
    real = torch.matmul(data, weight)
    data = -1.0 * torch.spmm(mul_L_imag, X_imag)
    real += torch.matmul(data, weight)

    data = torch.spmm(mul_L_imag, X_real)
    imag = torch.matmul(data, weight)
    data = torch.spmm(mul_L_real, X_imag)
    imag += torch.matmul(data, weight)
    return torch.stack([real, imag])


class ChebConv(nn.Module):
    """
    The MagNet convolution operation.

    :param in_c: int, number of input channels.
    :param out_c: int, number of output channels.
    :param K: int, the order of Chebyshev Polynomial.
    :param L_norm_real, L_norm_imag: normalized laplacian of real and imag
    """

    def __init__(self, in_c, out_c, K, L_norm_real, L_norm_imag, bias=True):
        super(ChebConv, self).__init__()

        L_norm_real, L_norm_imag = L_norm_real, L_norm_imag

        # list of K sparsetensors, each is N by N
        self.mul_L_real = L_norm_real  # [K, N, N]
        self.mul_L_imag = L_norm_imag  # [K, N, N]

        self.weight = nn.Parameter(torch.Tensor(K + 1, in_c, out_c))  # [K+1, 1, in_c, out_c]

        stdv = 1. / math.sqrt(self.weight.size(-1))
        self.weight.data.uniform_(-stdv, stdv)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, out_c))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter("bias", None)

    def forward(self, data):
        """
        :param inputs: the input data, real [B, N, C], img [B, N, C]
        :param L_norm_real, L_norm_imag: the laplace, [N, N], [N,N]
        """
        X_real, X_imag = data[0], data[1]

        real = 0.0
        imag = 0.0

        future = []
        for i in range(len(self.mul_L_real)):  # [K, B, N, D]
            future.append(torch.jit.fork(process,
                                         self.mul_L_real[i], self.mul_L_imag[i],
                                         self.weight[i], X_real, X_imag))
        result = []
        for i in range(len(self.mul_L_real)):
            result.append(torch.jit.wait(future[i]))
        result = torch.sum(torch.stack(result), dim=0)

        real = result[0]
        imag = result[1]
        return real + self.bias, imag + self.bias

class ChebConv2026_Direct(nn.Module):
    """
    differ from ChebConv is parameter(X_real, X_imag) in __init__ move to forward.
    """
    def __init__(self, in_c, out_c, K, bias=True):
        super(ChebConv2026_Direct, self).__init__()

        self.weight = nn.Parameter(torch.Tensor(K + 1, in_c, out_c))  # [K+1, 1, in_c, out_c]

        stdv = 1. / math.sqrt(self.weight.size(-1))
        self.weight.data.uniform_(-stdv, stdv)      # Qin learn: initializes the weights from a uniform distribution bounded by -stdv and stdv.

        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, out_c))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter("bias", None)

    # def forward(self, X_real, X_imag,  edges, q=0, edge_weight=None):
    def forward(self, data):
        '''
        main body is process: complex muplication between input x and L
        Args:
            data:real and img,

        Returns: new real and img after this ChebConv layer.( get the last layer of X from first layer of X)

        '''
        X_real, X_imag = data[0], data[1]
        edges, q, edge_weight = data[2], data[3], data[4],
        device = X_real.device
        size = X_real.size(0)

        f_node, e_node = edges[0], edges[1]
        laplacian = True
        gcn_appr = False
        L = QinDirect_hermitian_decomp_sparse(f_node, e_node, size, q, norm=True, QinDirect=laplacian, max_eigen=2.0, gcn_appr=gcn_appr, edge_weight=edge_weight)   # should norm
        multi_order_laplacian = cheb_poly_sparse(L, K=2)   # K=2 is temp by me
        L = multi_order_laplacian
        L_img = []
        L_real = []
        for i in range(len(L)):
            L_img.append(sparse_mx_to_torch_sparse_tensor(L[i].imag).to(device))
            L_real.append(sparse_mx_to_torch_sparse_tensor(L[i].real).to(device))
        # list of K sparsetensors, each is N by N
        L_norm_real = L_img     # [K, N, N]
        L_norm_imag = L_real    # [K, N, N]

        real = 0.0
        imag = 0.0

        future = []     # future stores handles to ongoing asynchronous computations
        for i in range(len(L_norm_real)):  # [K, B, N, D]
            future.append(torch.jit.fork(process,
                                         L_norm_real[i], L_norm_imag[i],
                                         self.weight[i], X_real, X_imag))
        result = []
        for i in range(len(L_norm_real)):
            result.append(torch.jit.wait(future[i]))
        result = torch.sum(torch.stack(result), dim=0)

        real = result[0]
        imag = result[1]
        # return real + self.bias, imag + self.bias
        return real + self.bias, imag + self.bias, edges, q, edge_weight

class ChebConv2026(nn.Module):
    """
    differ from ChebConv is parameter(X_real, X_imag) in __init__ move to forward.
    """
    def __init__(self, in_c, out_c, K, bias=True):
        super().__init__()

        self.weight = nn.Parameter(torch.Tensor(K + 1, in_c, out_c))  # [K+1, 1, in_c, out_c]

        stdv = 1. / math.sqrt(self.weight.size(-1))
        self.weight.data.uniform_(-stdv, stdv)      # Qin learn: initializes the weights from a uniform distribution bounded by -stdv and stdv.

        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, out_c))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter("bias", None)

    # def forward(self, X_real, X_imag,  edges, q=0, edge_weight=None):
    def forward(self, data):
        '''
        main body is process: complex muplication between input x and L
        Args:
            data:real and img,

        Returns: new real and img after this ChebConv layer.( get the last layer of X from first layer of X)

        '''
        X_real, X_imag = data[0], data[1]
        edges, q, edge_weight = data[2], data[3], data[4],
        device = X_real.device
        size = X_real.size(0)

        f_node, e_node = edges[0], edges[1]
        laplacian = True
        gcn_appr = False
        L = hermitian_decomp_sparse(f_node, e_node, size, q, norm=True, laplacian=laplacian, max_eigen=2.0, gcn_appr=gcn_appr, edge_weight=edge_weight)

        # try:
        #     L = hermitian_decomp_sparse(f_node, e_node, size, q, norm=True, laplacian=laplacian, max_eigen=2.0, gcn_appr=gcn_appr, edge_weight=edge_weight)
        #     # L = QinDirect_hermitian_decomp_sparse(f_node, e_node, size, q, norm=True, laplacian=laplacian, max_eigen=2.0, gcn_appr=gcn_appr, edge_weight=edge_weight)
        # except AttributeError:
        #     L = hermitian_decomp_sparse(f_node, e_node, size, q, norm=True, laplacian=laplacian, max_eigen=2.0, gcn_appr=gcn_appr, edge_weight=None)
        multi_order_laplacian = cheb_poly_sparse(L, K=2)   # K=2 is temp by me
        L = multi_order_laplacian
        L_img = []
        L_real = []
        for i in range(len(L)):
            L_img.append(sparse_mx_to_torch_sparse_tensor(L[i].imag).to(device))
            L_real.append(sparse_mx_to_torch_sparse_tensor(L[i].real).to(device))
        # list of K sparsetensors, each is N by N
        L_norm_real = L_img     # [K, N, N]
        L_norm_imag = L_real    # [K, N, N]

        real = 0.0
        imag = 0.0

        future = []     # future stores handles to ongoing asynchronous computations
        for i in range(len(L_norm_real)):  # [K, B, N, D]
            future.append(torch.jit.fork(process,
                                         L_norm_real[i], L_norm_imag[i],
                                         self.weight[i], X_real, X_imag))
        result = []
        for i in range(len(L_norm_real)):
            result.append(torch.jit.wait(future[i]))
        result = torch.sum(torch.stack(result), dim=0)

        real = result[0]
        imag = result[1]
        # return real + self.bias, imag + self.bias
        return real + self.bias, imag + self.bias, edges, q, edge_weight

class ChebConv2026_05(nn.Module):
    """
    differ from ChebConv is parameter(X_real, X_imag) in __init__ move to forward.
    """
    def __init__(self, in_c, out_c, K, bias=True):
        super(ChebConv2026_05, self).__init__()

        self.weight = nn.Parameter(torch.Tensor(K + 1, in_c, out_c))  # [K+1, 1, in_c, out_c]

        stdv = 1. / math.sqrt(self.weight.size(-1))
        self.weight.data.uniform_(-stdv, stdv)      # Qin learn: initializes the weights from a uniform distribution bounded by -stdv and stdv.

        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, out_c))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter("bias", None)

    def forward(self, data):
        '''
        main body is process: complex muplication between input x and L
        Args:
            data:real and img,

        Returns: new real and img after this ChebConv layer.( get the last layer of X from first layer of X)

        '''
        X_real, X_imag = data[0], data[1]
        edges, q, edge_weight = data[2], data[3], data[4],
        device = X_real.device
        size = X_real.size(0)

        f_node, e_node = edges[0], edges[1]
        laplacian = True
        gcn_appr = False
        L = hermitian_decomp_sparse(f_node, e_node, size, q, norm=True, laplacian=laplacian, max_eigen=2.0, gcn_appr=gcn_appr, edge_weight=edge_weight)

        # try:
        #     L = hermitian_decomp_sparse(f_node, e_node, size, q, norm=True, laplacian=laplacian, max_eigen=2.0, gcn_appr=gcn_appr, edge_weight=edge_weight)
        #     # L = QinDirect_hermitian_decomp_sparse(f_node, e_node, size, q, norm=True, laplacian=laplacian, max_eigen=2.0, gcn_appr=gcn_appr, edge_weight=edge_weight)
        # except AttributeError:
        #     L = hermitian_decomp_sparse(f_node, e_node, size, q, norm=True, laplacian=laplacian, max_eigen=2.0, gcn_appr=gcn_appr, edge_weight=None)
        multi_order_laplacian = cheb_poly_sparse(L, K=2)   # K=2 is temp by me
        L = multi_order_laplacian
        L_img = []
        L_real = []
        for i in range(len(L)):
            L_img.append(sparse_mx_to_torch_sparse_tensor(L[i].imag).to(device))
            L_real.append(sparse_mx_to_torch_sparse_tensor(L[i].real).to(device))
        # list of K sparsetensors, each is N by N
        L_norm_real = L_img     # [K, N, N]
        L_norm_imag = L_real    # [K, N, N]

        real = 0.0
        imag = 0.0

        future = []     # future stores handles to ongoing asynchronous computations
        for i in range(len(L_norm_real)):  # [K, B, N, D]
            future.append(torch.jit.fork(process,
                                         L_norm_real[i], L_norm_imag[i],
                                         self.weight[i], X_real, X_imag))
        result = []
        for i in range(len(L_norm_real)):
            result.append(torch.jit.wait(future[i]))
        result = torch.sum(torch.stack(result), dim=0)

        real = result[0]
        imag = result[1]
        # return real + self.bias, imag + self.bias
        return real + self.bias, imag + self.bias, edges, q, edge_weight

class ChebConv2026Direct(nn.Module):
    """
    differ from ChebConv is parameter(X_real, X_imag) in __init__ move to forward.
    """
    def __init__(self, in_c, out_c, K, bias=True):
        super(ChebConv2026Direct, self).__init__()

        self.weight = nn.Parameter(torch.Tensor(K + 1, in_c, out_c))  # [K+1, 1, in_c, out_c]

        stdv = 1. / math.sqrt(self.weight.size(-1))
        self.weight.data.uniform_(-stdv, stdv)      # Qin learn: initializes the weights from a uniform distribution bounded by -stdv and stdv.

        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, out_c))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter("bias", None)

    # def forward(self, X_real, X_imag,  edges, q=0, edge_weight=None):
    def forward(self, data):
        '''
        main body is process: complex muplication between input x and L
        Args:
            data:real and img,

        Returns: new real and img after this ChebConv layer.( get the last layer of X from first layer of X)

        '''
        X_real, X_imag = data[0], data[1]
        edges, q, edge_weight = data[2], data[3], data[4],
        device = X_real.device
        size = X_real.size(0)

        f_node, e_node = edges[0], edges[1]
        laplacian = True
        gcn_appr = False
        try:
            # L = hermitian_decomp_sparse(f_node, e_node, size, q, norm=True, laplacian=laplacian, max_eigen=2.0, gcn_appr=gcn_appr, edge_weight=edge_weight)
            L = QinDirect_hermitian_decomp_sparse(f_node, e_node, size, q, norm=True, laplacian=laplacian, max_eigen=2.0, gcn_appr=gcn_appr, edge_weight=edge_weight)
        except AttributeError:
            L = hermitian_decomp_sparse(f_node, e_node, size, q, norm=True, laplacian=laplacian, max_eigen=2.0, gcn_appr=gcn_appr, edge_weight=None)
        multi_order_laplacian = cheb_poly_sparse(L, K=2)   # K=2 is temp by me
        L = multi_order_laplacian
        L_img = []
        L_real = []
        for i in range(len(L)):
            L_img.append(sparse_mx_to_torch_sparse_tensor(L[i].imag).to(device))
            L_real.append(sparse_mx_to_torch_sparse_tensor(L[i].real).to(device))
        # list of K sparsetensors, each is N by N
        L_norm_real = L_img     # [K, N, N]
        L_norm_imag = L_real    # [K, N, N]

        real = 0.0
        imag = 0.0

        future = []     # future stores handles to ongoing asynchronous computations
        for i in range(len(L_norm_real)):  # [K, B, N, D]
            future.append(torch.jit.fork(process,
                                         L_norm_real[i], L_norm_imag[i],
                                         self.weight[i], X_real, X_imag))
        result = []
        for i in range(len(L_norm_real)):
            result.append(torch.jit.wait(future[i]))
        result = torch.sum(torch.stack(result), dim=0)

        real = result[0]
        imag = result[1]
        # return real + self.bias, imag + self.bias
        return real + self.bias, imag + self.bias, edges, q, edge_weight


class ChebConv2026_2bias(nn.Module):
    """
    differ from ChebConv is parameter(X_real, X_imag) in __init__ move to forward.
    """
    def __init__(self, in_c, out_c, K, bias=True):
        super(ChebConv2026_2bias, self).__init__()

        self.weight = nn.Parameter(torch.Tensor(K + 1, in_c, out_c))  # [K+1, 1, in_c, out_c]

        stdv = 1. / math.sqrt(self.weight.size(-1))
        self.weight.data.uniform_(-stdv, stdv)      # Qin learn: initializes the weights from a uniform distribution bounded by -stdv and stdv.

        if bias:
            self.biasreal = nn.Parameter(torch.Tensor(1, out_c))
            self.biasimag = nn.Parameter(torch.Tensor(1, out_c))
            nn.init.zeros_(self.biasreal)
            nn.init.zeros_(self.biasimag)
        else:
            self.register_parameter("bias", None)

    # def forward(self, X_real, X_imag,  edges, q=0, edge_weight=None):
    def forward(self, data):
        '''
        bias of real and imag are different
        main body is process: complex muplication between input x and L
        Args:
            data:real and img,

        Returns: new real and img after this ChebConv layer.( get the last layer of X from first layer of X)

        '''
        X_real, X_imag = data[0], data[1]
        edges, q, edge_weight = data[2], data[3], data[4],
        device = X_real.device
        size = X_real.size(0)

        f_node, e_node = edges[0], edges[1]
        laplacian = True
        gcn_appr = False
        try:
            L = hermitian_decomp_sparse(f_node, e_node, size, q, norm=True, laplacian=laplacian, max_eigen=2.0, gcn_appr=gcn_appr, edge_weight=edge_weight)
        except AttributeError:
            L = hermitian_decomp_sparse(f_node, e_node, size, q, norm=True, laplacian=laplacian, max_eigen=2.0, gcn_appr=gcn_appr, edge_weight=None)
        multi_order_laplacian = cheb_poly_sparse(L, K=2)   # K=2 is temp by me
        L = multi_order_laplacian
        L_img = []
        L_real = []
        for i in range(len(L)):
            L_img.append(sparse_mx_to_torch_sparse_tensor(L[i].imag).to(device))
            L_real.append(sparse_mx_to_torch_sparse_tensor(L[i].real).to(device))
        # list of K sparsetensors, each is N by N
        L_norm_real = L_img     # [K, N, N]
        L_norm_imag = L_real    # [K, N, N]

        real = 0.0
        imag = 0.0

        future = []     # future stores handles to ongoing asynchronous computations
        for i in range(len(L_norm_real)):  # [K, B, N, D]
            future.append(torch.jit.fork(process,
                                         L_norm_real[i], L_norm_imag[i],
                                         self.weight[i], X_real, X_imag))
        result = []
        for i in range(len(L_norm_real)):
            result.append(torch.jit.wait(future[i]))
        result = torch.sum(torch.stack(result), dim=0)

        real = result[0]
        imag = result[1]
        # return real + self.bias, imag + self.bias
        return real + self.biasreal, imag + self.biasimag, edges, q, edge_weight


class complex_relu_layer(nn.Module):
    def __init__(self, ):
        super(complex_relu_layer, self).__init__()

    def complex_relu(self, real, img):
        mask = 1.0 * (real >= 0)
        return mask * real, mask * img

    def forward(self, real, img=None):
        # for torch nn sequential usage
        # in this case, x_real is a tuple of (real, img)
        if img == None:
            img = real[1]
            real = real[0]

        real, img = self.complex_relu(real, img)
        return real, img

# def dropout(input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False) -> Tensor:
#     r"""
#     During training, randomly zeroes some of the elements of the input
#     tensor with probability :attr:`p` using samples from a Bernoulli
#     distribution.
#
#     See :class:`~torch.nn.Dropout` for details.
#
#     Args:
#         p: probability of an element to be zeroed. Default: 0.5
#         training: apply dropout if is ``True``. Default: ``True``
#         inplace: If set to ``True``, will do this operation in-place. Default: ``False``
#     """
#     if has_torch_function_unary(input):
#         return handle_torch_function(dropout, (input,), input, p=p, training=training, inplace=inplace)
#     if p < 0.0 or p > 1.0:
#         raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")
#     return _VF.dropout_(input, p, training) if inplace else _VF.dropout(input, p, training)

class complex_relu_layer2026(nn.Module):
    def __init__(self, ):
        super(complex_relu_layer2026, self).__init__()

    def complex_relu(self, real, img):
        mask = 1.0 * (real >= 0)
        return mask * real, mask * img

    def forward(self, real, img=None):
    # def forward(self, real, img,edges, q, edge_weight):
        # for torch nn sequential usage
        # in this case, x_real is a tuple of (real, img)
        if img == None:
            img = real[1]
            edges, q, edge_weight = real[2],real[3],real[4]
            real = real[0]
        real, img = self.complex_relu(real, img)
        return real, img, edges, q, edge_weight

class complex_relu_layer_SigBen(nn.Module):
    def __init__(self, ):
        super(complex_relu_layer_SigBen, self).__init__()

    def complex_relu(self, real, img):
        mask = 1.0 * (real >= 0)
        return mask * real, mask * img

    # def forward(self, real, img, edges, q, edge_weight):
    def forward(self, real, img, norm_real, norm_imag, edges):
        # for torch nn sequential usage
        # in this case, x_real is a tuple of (real, img)
        if img is None:
            img = real[1]
            real = real[0]
            norm_real, norm_imag, edges = real[2],real[3],real[4],
        real, img = self.complex_relu(real, img)
        return real, img, norm_real, norm_imag, edges


class ChebNet(nn.Module):
    def __init__(self, in_c, L_norm_real, L_norm_imag, num_filter=2, K=2, label_dim=2, activation=False, layer=2, dropout=False):
        """
        :param in_c: int, number of input channels.
        :param hid_c: int, number of hidden channels.
        :param K: for cheb series
        :param L_norm_real, L_norm_imag: normalized laplacian
        """
        super(ChebNet, self).__init__()

        chebs = [ChebConv(in_c=in_c, out_c=num_filter, K=K, L_norm_real=L_norm_real, L_norm_imag=L_norm_imag)]
        # chebs = [ChebConv2026(in_c=in_c, out_c=num_filter, K=K)]
        # self.ib1 = InceptionBlock(num_features, hidden)
        if activation:
            chebs.append(complex_relu_layer())

        for i in range(1, layer):
            chebs.append(ChebConv(in_c=num_filter, out_c=num_filter, K=K, L_norm_real=L_norm_real, L_norm_imag=L_norm_imag))
            if activation:
                chebs.append(complex_relu_layer())

        self.Chebs = torch.nn.Sequential(*chebs)

        last_dim = 2
        self.Conv = nn.Conv1d(num_filter * last_dim, label_dim, kernel_size=1)
        self.dropout = dropout

    def forward(self, real, imag):
        real, imag = self.Chebs((real, imag))
        # x = self.Chebs_Qin(x)
        # out = self.propagate(edge_index, x=x, size=size)
        # x0, x1, x2 = self.ib1(x, edge_index, edge_weight, edge_index2, edge_weight2)
        x = torch.cat((real, imag), dim=-1)

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)

        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv(x)
        x = F.log_softmax(x, dim=1)
        return x
class ChebNet_BenQin(nn.Module):
    # def __init__(self, in_c, L_norm_real, L_norm_imag, num_filter=2, K=2, label_dim=2, activation=False, layer=2, dropout=False):
    def __init__(self, in_c, num_filter=2, K=2, label_dim=2, activation=False, layer=2, dropout=False):
        """
        :param in_c: int, number of input channels.
        :param hid_c: int, number of hidden channels.
        :param K: for cheb series
        :param L_norm_real, L_norm_imag: normalized laplacian
        """
        super(ChebNet_BenQin, self).__init__()
        self.dropout = dropout

        chebs = [ChebConv2026_Direct(in_c=in_c, out_c=num_filter, K=K)]
        if activation:
            chebs.append(complex_relu_layer2026())

        for i in range(1, layer):
            chebs.append(ChebConv2026_Direct(in_c=num_filter, out_c=num_filter, K=K))
            if activation:
                chebs.append(complex_relu_layer2026())
        self.Chebs = torch.nn.Sequential(*chebs)

        last_dim = 2
        self.Conv = nn.Conv1d(num_filter * last_dim, label_dim, kernel_size=1)  # the input to nn.Conv1d is expected to be a 3D tensor with the shape (batch_size, in_channels, input_length)


    def forward(self, real, imag, edges, q, edge_weight):
        '''

        Args:
            real:
            imag:
            edges:
            q:
            edge_weight:

        Returns:

        '''
        real, imag, edges, q, edge_weight = self.Chebs((real, imag,  edges, q, edge_weight))
        # real, imag = self.cheb_Qin((real, imag, edges, q, edge_weight))
        x = torch.cat((real, imag), dim=-1)     # unwind the complex X into real-valued X

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)


        x = x.unsqueeze(0)      # can't simplify, because the input of Conv1d is 3D
        x = x.permute((0, 2, 1))
        x = self.Conv(x)
        x = F.log_softmax(x, dim=1)     # transforms the raw output scores (logits) into log probabilities, which are more numerically stable for computation and training
        x = x.permute(2, 1, 0).squeeze()
        return x
class ChebNet_Ben(nn.Module):
    # def __init__(self, in_c, L_norm_real, L_norm_imag, num_filter=2, K=2, label_dim=2, activation=False, layer=2, dropout=False):
    def __init__(self, in_c, num_filter=2, K=2, label_dim=2, activation=False, layer=2, dropout=False):
        """
        :param in_c: int, number of input channels.
        :param hid_c: int, number of hidden channels.
        :param K: for cheb series
        :param L_norm_real, L_norm_imag: normalized laplacian
        """
        super().__init__()

        chebs = [ChebConv2026(in_c=in_c, out_c=num_filter, K=K)]
        if activation:
            chebs.append(complex_relu_layer2026())

        for i in range(1, layer):
            chebs.append(ChebConv2026(in_c=num_filter, out_c=num_filter, K=K))
            if activation:
                chebs.append(complex_relu_layer2026())

        self.Chebs = torch.nn.Sequential(*chebs)

        last_dim = 2
        self.Conv = nn.Conv1d(num_filter * last_dim, label_dim, kernel_size=1)  # the input to nn.Conv1d is expected to be a 3D tensor with the shape (batch_size, in_channels, input_length)
        self.dropout = dropout

    def forward(self, real, imag, edges, q, edge_weight):
        '''

        Args:
            real:
            imag:
            edges:
            q:
            edge_weight:

        Returns:

        '''

        real, imag, edges, q, edge_weight = self.Chebs((real, imag,  edges, q, edge_weight))
        # real, imag = self.cheb_Qin((real, imag, edges, q, edge_weight))
        x = torch.cat((real, imag), dim=-1)     # unwind the complex X into real-valued X

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)


        x = x.unsqueeze(0)      # can't simplify, because the input of Conv1d is 3D
        x = x.permute((0, 2, 1))
        x = self.Conv(x)
        x = F.log_softmax(x, dim=1)     # transforms the raw output scores (logits) into log probabilities, which are more numerically stable for computation and training
        x = x.permute(2, 1, 0).squeeze()
        return x

class ChebNet_Ben_05(nn.Module):
    # def __init__(self, in_c, L_norm_real, L_norm_imag, num_filter=2, K=2, label_dim=2, activation=False, layer=2, dropout=False):
    def __init__(self, in_c, num_filter=2, K=2, label_dim=2, activation=False, layer=2, dropout=False):
        """
        :param in_c: int, number of input channels.
        :param hid_c: int, number of hidden channels.
        :param K: for cheb series
        :param L_norm_real, L_norm_imag: normalized laplacian
        """
        super(ChebNet_Ben_05, self).__init__()

        chebs = [ChebConv2026_05(in_c=in_c, out_c=num_filter, K=K)]
        if activation:
            chebs.append(complex_relu_layer2026())

        for i in range(1, layer):
            chebs.append(ChebConv2026_05(in_c=num_filter, out_c=num_filter, K=K))
            if activation:
                chebs.append(complex_relu_layer2026())

        self.Chebs = torch.nn.Sequential(*chebs)

        last_dim = 2
        self.Conv = nn.Conv1d(num_filter * last_dim, label_dim, kernel_size=1)  # the input to nn.Conv1d is expected to be a 3D tensor with the shape (batch_size, in_channels, input_length)
        self.dropout = dropout

    def forward(self, real, imag, edges, q, edge_weight):
        '''

        Args:
            real:
            imag:
            edges:
            q:
            edge_weight:

        Returns:

        '''

        real, imag, edges, q, edge_weight = self.Chebs((real, imag,  edges, q, edge_weight))
        # real, imag = self.cheb_Qin((real, imag, edges, q, edge_weight))
        x = torch.cat((real, imag), dim=-1)     # unwind the complex X into real-valued X

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)


        x = x.unsqueeze(0)      # can't simplify, because the input of Conv1d is 3D
        x = x.permute((0, 2, 1))
        x = self.Conv(x)
        x = F.log_softmax(x, dim=1)     # transforms the raw output scores (logits) into log probabilities, which are more numerically stable for computation and training
        x = x.permute(2, 1, 0).squeeze()
        return x

class ChebNet_BenX(nn.Module):
    def __init__(self, in_c, L_norm_real, L_norm_imag, num_filter=2, K=2, label_dim=2, activation=False, layer=2, dropout=False):
        """
        :param in_c: int, number of input channels.
        :param hid_c: int, number of hidden channels.
        :param K: for cheb series
        :param L_norm_real, L_norm_imag: normalized laplacian
        """
        super(ChebNet_BenX, self).__init__()

        chebs = [ChebConv(in_c=in_c, out_c=num_filter, K=K, L_norm_real=L_norm_real, L_norm_imag=L_norm_imag)]
        if activation:
            chebs.append(complex_relu_layer())

        for i in range(1, layer):
            chebs.append(ChebConv(in_c=num_filter, out_c=num_filter, K=K, L_norm_real=L_norm_real, L_norm_imag=L_norm_imag))
            if activation:
                chebs.append(complex_relu_layer())

        self.Chebs = torch.nn.Sequential(*chebs)

        last_dim = 2
        self.Conv = nn.Conv1d(num_filter * last_dim, label_dim, kernel_size=1)
        self.dropout = dropout

    def forward(self, real, imag):
        real, imag = self.Chebs((real, imag))
        x = torch.cat((real, imag), dim=-1)

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)

        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv(x)
        x = F.log_softmax(x, dim=1)
        return x


class ChebNet_Edge(nn.Module):
    def __init__(self, in_c, L_norm_real, L_norm_imag, num_filter=2, K=2, label_dim=2, activation=False, layer=2, dropout=False):
        """
        :param in_c: int, number of input channels.
        :param hid_c: int, number of hidden channels.
        :param K: for cheb series
        :param L_norm_real, L_norm_imag: normalized laplacian
        """
        super(ChebNet_Edge, self).__init__()

        chebs = [ChebConv(in_c=in_c, out_c=num_filter, K=K, L_norm_real=L_norm_real, L_norm_imag=L_norm_imag)]
        if activation and (layer != 1):
            chebs.append(complex_relu_layer())

        for i in range(1, layer):
            chebs.append(ChebConv(in_c=num_filter, out_c=num_filter, K=K, L_norm_real=L_norm_real, L_norm_imag=L_norm_imag))
            if activation:
                chebs.append(complex_relu_layer())
        self.Chebs = torch.nn.Sequential(*chebs)

        last_dim = 2
        self.linear = nn.Linear(num_filter * last_dim * 2, label_dim)
        self.dropout = dropout

    def forward(self, real, imag, index):
        real, imag = self.Chebs((real, imag))
        x = torch.cat((real[index[:, 0]], real[index[:, 1]], imag[index[:, 0]], imag[index[:, 1]]), dim=-1)
        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)

        x = self.linear(x)
        x = F.log_softmax(x, dim=1)
        return x

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)