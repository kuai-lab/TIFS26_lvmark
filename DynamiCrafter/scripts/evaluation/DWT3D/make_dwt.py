import numpy as np
import torch
import torch.nn as nn
import pywt
import math


class DWT_3D(nn.Module):
    def __init__(self, wavename, device='cuda:0'):
        super(DWT_3D, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.rec_lo
        self.band_high = wavelet.rec_hi
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)
        self.device = device  # 원하는 디바이스 설정

    def get_matrix(self):
        L1 = np.max((self.input_height, self.input_width))
        L = math.floor(L1 / 2)
        matrix_h = np.zeros((L, L1 + self.band_length - 2))
        matrix_g = np.zeros((L1 - L, L1 + self.band_length - 2))
        end = None if self.band_length_half == 1 else (- self.band_length_half + 1)

        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index + j] = self.band_low[j]
            index += 2
        matrix_h_0 = matrix_h[0:(math.floor(self.input_height / 2)),
                              0:(self.input_height + self.band_length - 2)]
        matrix_h_1 = matrix_h[0:(math.floor(self.input_width / 2)),
                              0:(self.input_width + self.band_length - 2)]
        matrix_h_2 = matrix_h[0:(math.floor(self.input_depth / 2)),
                              0:(self.input_depth + self.band_length - 2)]

        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index + j] = self.band_high[j]
            index += 2
        matrix_g_0 = matrix_g[0:(self.input_height - math.floor(self.input_height / 2)),
                              0:(self.input_height + self.band_length - 2)]
        matrix_g_1 = matrix_g[0:(self.input_width - math.floor(self.input_width / 2)),
                              0:(self.input_width + self.band_length - 2)]
        matrix_g_2 = matrix_g[0:(self.input_depth - math.floor(self.input_depth / 2)),
                              0:(self.input_depth + self.band_length - 2)]

        matrix_h_0 = matrix_h_0[:, (self.band_length_half - 1):end]
        matrix_h_1 = matrix_h_1[:, (self.band_length_half - 1):end]
        matrix_h_1 = np.transpose(matrix_h_1)
        matrix_h_2 = matrix_h_2[:, (self.band_length_half - 1):end]

        matrix_g_0 = matrix_g_0[:, (self.band_length_half - 1):end]
        matrix_g_1 = matrix_g_1[:, (self.band_length_half - 1):end]
        matrix_g_1 = np.transpose(matrix_g_1)
        matrix_g_2 = matrix_g_2[:, (self.band_length_half - 1):end]

        # 원하는 device로 이동시키기
        self.matrix_low_0 = torch.Tensor(matrix_h_0).to(self.device)
        self.matrix_low_1 = torch.Tensor(matrix_h_1).to(self.device)
        self.matrix_low_2 = torch.Tensor(matrix_h_2).to(self.device)
        self.matrix_high_0 = torch.Tensor(matrix_g_0).to(self.device)
        self.matrix_high_1 = torch.Tensor(matrix_g_1).to(self.device)
        self.matrix_high_2 = torch.Tensor(matrix_g_2).to(self.device)

    def forward(self, input):
        assert len(input.size()) == 5
        self.input_depth = input.size()[-3]
        self.input_height = input.size()[-2]
        self.input_width = input.size()[-1]
        self.get_matrix()

        # Using the matrices directly within forward
        L = torch.matmul(self.matrix_low_0, input)
        H = torch.matmul(self.matrix_high_0, input)
        
        LL = torch.matmul(L, self.matrix_low_1).transpose(dim0=2, dim1=3)
        LH = torch.matmul(L, self.matrix_high_1).transpose(dim0=2, dim1=3)
        HL = torch.matmul(H, self.matrix_low_1).transpose(dim0=2, dim1=3)
        HH = torch.matmul(H, self.matrix_high_1).transpose(dim0=2, dim1=3)

        LLL = torch.matmul(self.matrix_low_2, LL).transpose(dim0=2, dim1=3)
        LLH = torch.matmul(self.matrix_low_2, LH).transpose(dim0=2, dim1=3)
        LHL = torch.matmul(self.matrix_low_2, HL).transpose(dim0=2, dim1=3)
        LHH = torch.matmul(self.matrix_low_2, HH).transpose(dim0=2, dim1=3)
        HLL = torch.matmul(self.matrix_high_2, LL).transpose(dim0=2, dim1=3)
        HLH = torch.matmul(self.matrix_high_2, LH).transpose(dim0=2, dim1=3)
        HHL = torch.matmul(self.matrix_high_2, HL).transpose(dim0=2, dim1=3)
        HHH = torch.matmul(self.matrix_high_2, HH).transpose(dim0=2, dim1=3)

        return LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH
