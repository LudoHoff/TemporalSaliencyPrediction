import torchvision.models as models
import torch
import torch.nn as nn
from collections import OrderedDict
import sys

sys.path.append('../PNAS/')
from PNASnet import *
from genotypes import PNASNet

class PNASModel(nn.Module):

    def __init__(self, num_channels=3, train_enc=False, load_weight=1):
        super(PNASModel, self).__init__()
        self.path = '../PNAS/PNASNet-5_Large.pth'

        self.pnas = NetworkImageNet(216, 1001, 12, False, PNASNet)
        if load_weight:
            self.pnas.load_state_dict(torch.load(self.path))

        for param in self.pnas.parameters():
            param.requires_grad = train_enc

        self.padding = nn.ConstantPad2d((0,1,0,1),0)
        self.drop_path_prob = 0

        self.linear_upsampling = nn.UpsamplingBilinear2d(scale_factor=2)

        self.deconv_layer0 = nn.Sequential(
            nn.Conv2d(in_channels = 4320, out_channels = 512, kernel_size=3, padding=1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )

        self.deconv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels = 512+2160, out_channels = 256, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels = 1080+256, out_channels = 270, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer3 = nn.Sequential(
            nn.Conv2d(in_channels = 540, out_channels = 96, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer4 = nn.Sequential(
            nn.Conv2d(in_channels = 192, out_channels = 128, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer5 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = 128, out_channels = 1, kernel_size = 3, padding = 1, bias = True),
            nn.Sigmoid()
        )

    def forward(self, images):
        batch_size = images.size(0)

        s0 = self.pnas.conv0(images)
        s0 = self.pnas.conv0_bn(s0)
        out1 = self.padding(s0)

        s1 = self.pnas.stem1(s0, s0, self.drop_path_prob)
        out2 = s1
        s0, s1 = s1, self.pnas.stem2(s0, s1, 0)

        for i, cell in enumerate(self.pnas.cells):
            s0, s1 = s1, cell(s0, s1, 0)
            if i==3:
                out3 = s1
            if i==7:
                out4 = s1
            if i==11:
                out5 = s1

        out5 = self.deconv_layer0(out5)

        x = torch.cat((out5,out4), 1)
        x = self.deconv_layer1(x)

        x = torch.cat((x,out3), 1)
        x = self.deconv_layer2(x)

        x = torch.cat((x,out2), 1)
        x = self.deconv_layer3(x)
        x = torch.cat((x,out1), 1)

        x = self.deconv_layer4(x)
        
        x = self.deconv_layer5(x)
        x = x.squeeze(1)
        return x


class PNASBoostedModel(nn.Module):

    def __init__(self, device, model_path, model_vol_path, time_slices, train_model=False):
        super(PNASBoostedModel, self).__init__()
        
        model_vol = PNASVolModel(time_slices=time_slices)
        model_vol = nn.DataParallel(model_vol)
        state_dict = torch.load(model_vol_path)
        new_state_dict = OrderedDict()

        for k, v in state_dict.items():
            if 'module' not in k:
                k = 'module.' + k
            else:
                k = k.replace('features.module.', 'module.features.')
            new_state_dict[k] = v

        model_vol.load_state_dict(new_state_dict)
        self.pnas_vol = model_vol.to(device)

        for param in self.pnas_vol.parameters():
            param.requires_grad = False


        model = PNASModel()
        model = nn.DataParallel(model)
        state_dict = torch.load(model_path)
        new_state_dict = OrderedDict()

        for k, v in state_dict.items():
            if 'module' not in k:
                k = 'module.' + k
            else:
                k = k.replace('features.module.', 'module.features.')
            new_state_dict[k] = v

        model.load_state_dict(new_state_dict)
        self.pnas = model.to(device)

        for param in self.pnas.parameters():
            param.requires_grad = train_model
        

        self.deconv_mix = nn.Sequential(
            nn.Conv2d(in_channels = 1 + time_slices, out_channels = 16, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = 32, out_channels = 1, kernel_size = 3, padding = 1, bias = True),
            nn.Sigmoid()
        )


    def forward(self, images):
        pnas_pred = self.pnas(images).unsqueeze(1)
        pnas_vol_pred = self.pnas_vol(images)

        x = torch.cat((pnas_pred, pnas_vol_pred), 1)
        x = self.deconv_mix(x)
        x = x.squeeze(1)

        return x


class PNASVolModel(nn.Module):

    def __init__(self, time_slices, num_channels=3, train_enc=False, load_weight=1):
        super(PNASVolModel, self).__init__()
        self.path = '../PNAS/PNASNet-5_Large.pth'

        self.pnas = NetworkImageNet(216, 1001, 12, False, PNASNet)
        if load_weight:
            self.pnas.load_state_dict(torch.load(self.path))

        for param in self.pnas.parameters():
            param.requires_grad = train_enc

        self.padding = nn.ConstantPad2d((0,1,0,1),0)
        self.drop_path_prob = 0

        self.linear_upsampling = nn.UpsamplingBilinear2d(scale_factor=2)

        self.deconv_layer0 = nn.Sequential(
            nn.Conv2d(in_channels = 4320, out_channels = 512, kernel_size=3, padding=1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )

        self.deconv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels = 512+2160, out_channels = 256, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels = 1080+256, out_channels = 270, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer3 = nn.Sequential(
            nn.Conv2d(in_channels = 540, out_channels = 96, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer4 = nn.Sequential(
            nn.Conv2d(in_channels = 192, out_channels = 128, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )

        self.deconv_layer5 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 64, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = 32, out_channels = time_slices, kernel_size = 3, padding = 1, bias = True),
            nn.Sigmoid()
        )

        # def init_weights(m):
        #     if isinstance(m, nn.Linear):
        #         torch.nn.init.xavier_uniform_(m.weight)
        #         m.bias.data.fill_(0.01)

        # self.linear.apply(init_weights)

    def forward(self, images):
        s0 = self.pnas.conv0(images)
        s0 = self.pnas.conv0_bn(s0)
        out1 = self.padding(s0)

        s1 = self.pnas.stem1(s0, s0, self.drop_path_prob)
        out2 = s1
        s0, s1 = s1, self.pnas.stem2(s0, s1, 0)

        for i, cell in enumerate(self.pnas.cells):
            s0, s1 = s1, cell(s0, s1, 0)
            if i==3:
                out3 = s1
            if i==7:
                out4 = s1
            if i==11:
                out5 = s1

        out5 = self.deconv_layer0(out5)

        x = torch.cat((out5,out4), 1)
        x = self.deconv_layer1(x)

        x = torch.cat((x,out3), 1)
        x = self.deconv_layer2(x)

        x = torch.cat((x,out2), 1)
        x = self.deconv_layer3(x)
        x = torch.cat((x,out1), 1)

        x = self.deconv_layer4(x)

        x = self.deconv_layer5(x)
        x = x / x.max()

        return x


class SimpleNet(nn.Module):
    
        def __init__(self):
            super(SimpleNet, self).__init__()

            self.linear_upsampling = nn.UpsamplingBilinear2d(scale_factor=2)

            self.deconv_layer0 = nn.Sequential(
                nn.Conv2d(in_channels = 4320, out_channels = 512, kernel_size=3, padding=1, bias = True),
                nn.ReLU(inplace=True),
                self.linear_upsampling
            )

            self.deconv_layer1 = nn.Sequential(
                nn.Conv2d(in_channels = 512+2160, out_channels = 256, kernel_size = 3, padding = 1, bias = True),
                nn.ReLU(inplace=True),
                self.linear_upsampling
            )
            self.deconv_layer2 = nn.Sequential(
                nn.Conv2d(in_channels = 1080+256, out_channels = 270, kernel_size = 3, padding = 1, bias = True),
                nn.ReLU(inplace=True),
                self.linear_upsampling
            )
            self.deconv_layer3 = nn.Sequential(
                nn.Conv2d(in_channels = 540, out_channels = 96, kernel_size = 3, padding = 1, bias = True),
                nn.ReLU(inplace=True),
                self.linear_upsampling
            )
            self.deconv_layer4 = nn.Sequential(
                nn.Conv2d(in_channels = 192, out_channels = 128, kernel_size = 3, padding = 1, bias = True),
                nn.ReLU(inplace=True),
                self.linear_upsampling
            )

            self.deconv_layer5 = nn.Sequential(
                nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 1, bias = True),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels = 128, out_channels = 1, kernel_size = 3, padding = 1, bias = True),
                nn.Sigmoid()
            )

        def forward(self, pnas_output):
            out1, out2, out3, out4, out5 = pnas_output
            out5 = self.deconv_layer0(out5)

            x = torch.cat((out5,out4), 1)
            x = self.deconv_layer1(x)

            x = torch.cat((x,out3), 1)
            x = self.deconv_layer2(x)

            x = torch.cat((x,out2), 1)
            x = self.deconv_layer3(x)
            x = torch.cat((x,out1), 1)

            x = self.deconv_layer4(x)
            
            x = self.deconv_layer5(x)
            x = x.squeeze(1)
            return x


class VolModel(nn.Module):
    
    def __init__(self, device, time_slices, num_channels=3, train_enc=False, load_weight=1):
        super(VolModel, self).__init__()
        self.path = '../PNAS/PNASNet-5_Large.pth'
        self.time_slices = time_slices

        self.pnas = NetworkImageNet(216, 1001, 12, False, PNASNet)

        self.padding = nn.ConstantPad2d((0,1,0,1),0)
        self.drop_path_prob = 0
        
        if load_weight:
            self.pnas.load_state_dict(torch.load(self.path))

        for param in self.pnas.parameters():
            param.requires_grad = train_enc

        for i in range(time_slices):
            self.__dict__['model_' + str(i)] = SimpleNet().to(device)


    def forward(self, images):
        s0 = self.pnas.conv0(images)
        s0 = self.pnas.conv0_bn(s0)
        out1 = self.padding(s0)

        s1 = self.pnas.stem1(s0, s0, self.drop_path_prob)
        out2 = s1
        s0, s1 = s1, self.pnas.stem2(s0, s1, 0)

        for i, cell in enumerate(self.pnas.cells):
            s0, s1 = s1, cell(s0, s1, 0)
            if i==3:
                out3 = s1
            if i==7:
                out4 = s1
            if i==11:
                out5 = s1

        preds = None
        for i in range(self.time_slices):
            pred = self.__dict__['model_' + str(i)]((out1, out2, out3, out4, out5))
            pred = torch.unsqueeze(pred, 1)

            if preds == None:
                preds = pred
            else:
                preds = torch.cat((preds, pred), 1)

        return preds
    # def __init__(self, time_slices, device, num_channels=3, train_enc=False, load_weight=1):
    #     super(VolModel, self).__init__()
    #     self.models = [PNASModel(num_channels=num_channels, train_enc=train_enc, load_weight=load_weight) for _ in range(time_slices)]

    #     for model in self.models:
    #         model.to(device)

    # def forward(self, images):
    #     preds = torch.zeros((images.size()[0], len(self.models), 256, 256)).cuda()

    #     for i, pred in enumerate([model(images) for model in self.models]):
    #         preds[:,i] = pred
        
    #     return preds

    # def params(self):
    #     params_list = [model.parameters() for model in self.models]
    #     return list(filter(lambda p: p.requires_grad, [param for params in params_list for param in params]))

class DenseModel(nn.Module):

    def __init__(self, num_channels=3, train_enc=False, load_weight=1):
        super(DenseModel, self).__init__()

        self.dense = models.densenet161(pretrained=bool(load_weight)).features

        for param in self.dense.parameters():
            param.requires_grad = train_enc

        self.linear_upsampling = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv_layer0 = nn.Sequential(*list(self.dense)[:3])
        self.conv_layer1 = nn.Sequential(
        	self.dense.pool0,
        	self.dense.denseblock1,
        	*list(self.dense.transition1)[:3]
        )
        self.conv_layer2 = nn.Sequential(
        	self.dense.transition1[3],
        	self.dense.denseblock2,
        	*list(self.dense.transition2)[:3]
        )
        self.conv_layer3 = nn.Sequential(
        	self.dense.transition2[3],
        	self.dense.denseblock3,
        	*list(self.dense.transition3)[:3]
        )
        self.conv_layer4 = nn.Sequential(
        	self.dense.transition3[3],
        	self.dense.denseblock4
        )


        self.deconv_layer0 = nn.Sequential(
            nn.Conv2d(in_channels = 2208, out_channels = 512, kernel_size=3, padding=1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )

        self.deconv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels = 512+1056, out_channels = 256, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels = 384+256, out_channels = 192, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer3 = nn.Sequential(
            nn.Conv2d(in_channels = 192+192, out_channels = 96, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer4 = nn.Sequential(
            nn.Conv2d(in_channels = 96+96, out_channels = 128, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer5 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = 128, out_channels = 1, kernel_size = 3, padding = 1, bias = True),
            nn.Sigmoid()
        )

    def forward(self, images):
        batch_size = images.size(0)

        out1 = self.conv_layer0(images)
        out2 = self.conv_layer1(out1)
        out3 = self.conv_layer2(out2)
        out4 = self.conv_layer3(out3)
        out5 = self.conv_layer4(out4)


        assert out1.size() == (batch_size, 96, 128, 128)
        assert out2.size() == (batch_size, 192, 64, 64)
        assert out3.size() == (batch_size, 384, 32, 32)
        assert out4.size() == (batch_size, 1056, 16, 16)
        assert out5.size() == (batch_size, 2208, 8, 8)

        out5 = self.deconv_layer0(out5)

        x = torch.cat((out5,out4), 1)
        x = self.deconv_layer1(x)

        x = torch.cat((x,out3), 1)
        x = self.deconv_layer2(x)

        x = torch.cat((x,out2), 1)
        x = self.deconv_layer3(x)

        x = torch.cat((x,out1), 1)
        x = self.deconv_layer4(x)
        x = self.deconv_layer5(x)
        x = x.squeeze(1)
        return x

class ResNetModel(nn.Module):

    def __init__(self, num_channels=3, train_enc=False, load_weight=1):
        super(ResNetModel, self).__init__()

        self.num_channels = num_channels
        self.resnet = models.resnet50(pretrained=bool(load_weight))

        for param in self.resnet.parameters():
            param.requires_grad = train_enc

        self.conv_layer1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu
        )
        self.conv_layer2 = nn.Sequential(
            self.resnet.maxpool,
            self.resnet.layer1
        )
        self.conv_layer3 = self.resnet.layer2
        self.conv_layer4 = self.resnet.layer3
        self.conv_layer5 = self.resnet.layer4

        self.linear_upsampling = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv_layer0 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels = 2048, out_channels = 512, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels = 1024, out_channels = 256, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer3 = nn.Sequential(
            nn.Conv2d(in_channels = 512, out_channels = 64, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer4 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 64, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer5 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = 64, out_channels = 1, kernel_size = 3, padding = 1, bias = True),
            nn.Sigmoid()
        )

    def forward(self, images):
        batch_size = images.size(0)

        out1 = self.conv_layer1(images)
        out2 = self.conv_layer2(out1)
        out3 = self.conv_layer3(out2)
        out4 = self.conv_layer4(out3)
        out5 = self.conv_layer5(out4)

        out5 = self.deconv_layer0(out5)
        assert out5.size() == (batch_size, 1024, 16, 16)

        x = torch.cat((out5,out4), 1)
        assert x.size() == (batch_size, 2048, 16, 16)
        x = self.deconv_layer1(x)
        assert x.size() == (batch_size, 512, 32, 32)

        x = torch.cat((x, out3), 1)
        assert x.size() == (batch_size, 1024, 32, 32)
        x = self.deconv_layer2(x)
        assert x.size() == (batch_size, 256, 64, 64)

        x = torch.cat((x, out2), 1)
        assert x.size() == (batch_size, 512, 64, 64)
        x = self.deconv_layer3(x)
        assert x.size() == (batch_size, 64, 128, 128)

        x = torch.cat((x, out1), 1)
        assert x.size() == (batch_size, 128, 128, 128)
        x = self.deconv_layer4(x)
        x = self.deconv_layer5(x)
        assert x.size() == (batch_size, 1, 256, 256)
        x = x.squeeze(1)
        assert x.size() == (batch_size, 256, 256)
        return x

class VGGModel(nn.Module):

    def __init__(self, num_channels=3, train_enc=False, load_weight=1):
        super(VGGModel, self).__init__()

        self.num_channels = num_channels
        self.vgg = models.vgg16(pretrained=bool(load_weight)).features

        for param in self.vgg.parameters():
            param.requires_grad = train_enc

        self.conv_layer1 = self.vgg[:7]
        self.conv_layer2 = self.vgg[7:12]
        self.conv_layer3 = self.vgg[12:19]
        self.conv_layer4 = self.vgg[19:24]
        self.conv_layer5 = self.vgg[24:]

        self.linear_upsampling = nn.UpsamplingBilinear2d(scale_factor=2)

        self.deconv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels = 1024, out_channels = 512, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels = 1024, out_channels = 256, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer3 = nn.Sequential(
            nn.Conv2d(in_channels = 512, out_channels = 128, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer4 = nn.Sequential(
            nn.Conv2d(in_channels = 256, out_channels = 128, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer5 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = 128, out_channels = 1, kernel_size = 3, padding = 1, bias = True),
            nn.Sigmoid()
        )

    def forward(self, images):
        batch_size = images.size(0)

        out1 = self.conv_layer1(images)
        out2 = self.conv_layer2(out1)
        out3 = self.conv_layer3(out2)
        out4 = self.conv_layer4(out3)
        out5 = self.conv_layer5(out4)

        out5 = self.linear_upsampling(out5)
        assert out5.size() == (batch_size, 512, 16, 16)

        x = torch.cat((out5,out4), 1)
        assert x.size() == (batch_size, 1024, 16, 16)
        x = self.deconv_layer1(x)
        assert x.size() == (batch_size, 512, 32, 32)

        x = torch.cat((x, out3), 1)
        assert x.size() == (batch_size, 1024, 32, 32)
        x = self.deconv_layer2(x)
        assert x.size() == (batch_size, 256, 64, 64)

        x = torch.cat((x, out2), 1)
        assert x.size() == (batch_size, 512, 64, 64)
        x = self.deconv_layer3(x)
        assert x.size() == (batch_size, 128, 128, 128)

        x = torch.cat((x, out1), 1)
        assert x.size() == (batch_size, 256, 128, 128)
        x = self.deconv_layer4(x)
        x = self.deconv_layer5(x)
        assert x.size() == (batch_size, 1, 256, 256)
        x = x.squeeze(1)
        assert x.size() == (batch_size, 256, 256)
        return x

class MobileNetV2(nn.Module):

    def __init__(self, num_channels=3, train_enc=False, load_weight=1):
        super(MobileNetV2, self).__init__()

        self.mobilenet = torch.hub.load('pytorch/vision:v0.4.0', 'mobilenet_v2', pretrained=True).features

        for param in self.mobilenet.parameters():
            param.requires_grad = train_enc

        self.linear_upsampling = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv_layer1 = self.mobilenet[:2]
        self.conv_layer2 = self.mobilenet[2:4]
        self.conv_layer3 = self.mobilenet[4:7]
        self.conv_layer4 = self.mobilenet[7:14]
        self.conv_layer5 = self.mobilenet[14:]


        self.deconv_layer0 = nn.Sequential(
            nn.Conv2d(in_channels = 1280, out_channels = 96, kernel_size=3, padding=1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )

        self.deconv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels = 96+96, out_channels = 32, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels = 32+32, out_channels = 24, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer3 = nn.Sequential(
            nn.Conv2d(in_channels = 24+24, out_channels = 16, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer4 = nn.Sequential(
            nn.Conv2d(in_channels = 16+16, out_channels = 16, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer5 = nn.Sequential(
            nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = 16, out_channels = 1, kernel_size = 3, padding = 1, bias = True),
            nn.Sigmoid()
        )

    def forward(self, images):
        batch_size = images.size(0)

        out1 = self.conv_layer1(images)
        out2 = self.conv_layer2(out1)
        out3 = self.conv_layer3(out2)
        out4 = self.conv_layer4(out3)
        out5 = self.conv_layer5(out4)


        assert out1.size() == (batch_size, 16, 128, 128)
        assert out2.size() == (batch_size, 24, 64, 64)
        assert out3.size() == (batch_size, 32, 32, 32)
        assert out4.size() == (batch_size, 96, 16, 16)
        assert out5.size() == (batch_size, 1280, 8, 8)

        out5 = self.deconv_layer0(out5)

        x = torch.cat((out5,out4), 1)
        x = self.deconv_layer1(x)

        x = torch.cat((x,out3), 1)
        x = self.deconv_layer2(x)

        x = torch.cat((x,out2), 1)
        x = self.deconv_layer3(x)

        x = torch.cat((x,out1), 1)
        x = self.deconv_layer4(x)
        x = self.deconv_layer5(x)
        x = x.squeeze(1)
        return x
