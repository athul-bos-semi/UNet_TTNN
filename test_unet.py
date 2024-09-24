from unet_utils import *
import tt_lib

# device = ttnn.open_device(2)
device = tt_lib.device.CreateDevice(2, l1_small_size=32768, num_hw_cqs=1)

# device = ttnn.open_device_mesh(
#         ttnn.DeviceGrid(1, 1),
#         [0]
#     )

image_height = 224
image_width = 224

class UNet(nn.Module):
    def __init__(self, in_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # self.in_conv = DoubleConv(in_channels, 64)
        self.in_conv = TTNN_DoubleConv(device, image_height, image_width, in_channels, 64)
        # self.downsample_1 = Down(64, 128)
        self.downsample_1 = TTNN_Down(device, image_height, image_width, 64, 128)
        # self.downsample_2 = Down(128, 256)
        self.downsample_2 = TTNN_Down(device, image_height//2, image_width//2, 128, 256)
        # self.downsample_3 = Down(256, 512)
        self.downsample_3 = TTNN_Down(device, image_height//4, image_width//4, 256, 512)
        factor = 2 if bilinear else 1
        # self.downsample_4 = Down(512, 1024 // factor)
        self.downsample_4 = TTNN_Down(device, image_height//8, image_width//8, 512, 1024//factor)

        # self.upsample_1 = Up(1024, 512 // factor, bilinear)
        self.upsample_1 = TTNN_Up(device, 14, 14, 1024, 512 // factor, bilinear)
        # self.upsample_2 = Up(512, 256 // factor, bilinear)
        # self.upsample_3 = Up(256, 128 // factor, bilinear)
        # self.upsample_4 = Up(128, 64, bilinear)
        # self.out_conv = OutConv(64, n_classes)

    def forward(self, x):
        x = x.permute(0,2,3,1).reshape(1,1,-1,self.in_channels)
        x = ttnn.from_torch(x,ttnn.bfloat16)
        print("\n---------------------- Initial Convolution ----------------------\n")
        x1, x1_feature_height, x1_feature_width = self.in_conv(x)
        
        print("\n---------------------- Downsample 1 ----------------------\n")
        x2, x2_feature_height, x2_feature_width = self.downsample_1(x1)
        print("\n---------------------- Downsample 2 ----------------------\n")
        x3, x3_feature_height, x3_feature_width = self.downsample_2(x2)
        print("\n---------------------- Downsample 3 ----------------------\n")
        x4, x4_feature_height, x4_feature_width = self.downsample_3(x3)
        print("\n---------------------- Downsample 4 ----------------------\n")
        x5, x5_feature_height, x5_feature_width  = self.downsample_4(x4)
        
        return x5

        print("\n---------------------- Upsample 1 ----------------------\n")
        x = self.upsample_1(x5, x4)
        return x
        # x = self.upsample_2(x, x3)
        # x = self.upsample_3(x, x2)
        # x = self.upsample_4(x, x1)
        
        # logits = self.out_conv(x)
        
        # return logits

model = UNet(3, 10, True)

test_image = torch.randn(1, 3, image_height, image_width, dtype=torch.bfloat16)
# test_image = test_image.permute(0,2,3,1)#.reshape(1,1,-1,3)

output = model(test_image)

# print(output)

# ttnn.close_device(2)
