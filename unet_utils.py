import torch
import torch.nn as nn
import torch.nn.functional as F
import ttnn
import tt_lib

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.double_conv(x)
        feature_height, feature_width = out.shape[2], out.shape[3]
        return out, feature_height, feature_width

class TTNN_DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, device, input_height, input_width, in_channels, out_channels, mid_channels=None):
        super().__init__()

        self.input_height = input_height
        self.input_width = input_width
        self.in_channels = in_channels
        self.mid_channels = out_channels if not mid_channels else mid_channels
        self.out_channels = out_channels
        self.device = device
        self.device_name = "wormhole_b0" # if "wormhole_b0" in os.environ.get("ARCH_NAME", os.environ.get("TT_ARCH_NAME", "")).lower() else "grayskull"

        # Convolution Weight Shape = [out_channels, in_channels, kernel_height, kernel_width]
        self.conv1_weight_tensor = torch.randn(self.mid_channels, self.in_channels, 3, 3, dtype=torch.bfloat16)
        # self.conv1_weight_tensor = self.conv1_weight_tensor.permute(2,3,0,1).reshape(1,1,-1,self.in_channels)
        self.conv1_weight_tensor = ttnn.from_torch(self.conv1_weight_tensor, dtype=ttnn.bfloat16)
        # self.conv1_bias_tensor = torch.zeros(self.mid_channels,dtype=ttnn.bfloat16)
        # self.conv1_bias_tensor = ttnn.from_torch(self.conv1_bias_tensor)
        self.conv1_bias_tensor = None
        self.conv1_use_shallow_conv_variant = False
        self.conv1_config = ttnn.Conv2dConfig(
            dtype=ttnn.bfloat16,
            weights_dtype=ttnn.bfloat16,
            math_fidelity=ttnn.MathFidelity.LoFi,
            height_sharding=True,
            input_channels_alignment=16 if self.conv1_use_shallow_conv_variant else 32,
            fp32_dest_acc_enabled=False,
            activation="relu",
            deallocate_activation=True,
            reshard_if_not_optimal=False,
            transpose_shards=False,
            packer_l1_accum_enabled=True if self.device_name == "wormhole_b0" else False,
            act_block_h_override=32,
        )

        self.conv2_weight_tensor = torch.randn(self.out_channels, self.mid_channels, 3, 3, dtype=torch.bfloat16)
        # self.conv2_weight_tensor = self.conv1_weight_tensor.permute(0,2,3,1).reshape(1,1,-1,self.in_channels)
        self.conv2_weight_tensor = ttnn.from_torch(self.conv2_weight_tensor, dtype=ttnn.bfloat16)
        # self.conv2_bias_tensor = torch.zeros(self.out_channels,dtype=ttnn.bfloat16)
        # self.conv2_bias_tensor = ttnn.from_torch(self.conv2_bias_tensor)
        self.conv2_bias_tensor = None
        self.conv2_use_shallow_conv_variant = False
        self.conv2_config = ttnn.Conv2dConfig(
            dtype=ttnn.bfloat16,
            weights_dtype=ttnn.bfloat16,
            math_fidelity=ttnn.MathFidelity.LoFi,
            height_sharding=True,
            input_channels_alignment=16 if self.conv2_use_shallow_conv_variant else 32,
            fp32_dest_acc_enabled=False,
            activation="relu",
            deallocate_activation=True,
            reshard_if_not_optimal=False,
            transpose_shards=False,
            packer_l1_accum_enabled=True if self.device_name == "wormhole_b0" else False,
            act_block_h_override=32,
        )

    def forward(self, x):

        if x.get_legacy_shape()[2] == 3136 and x.get_legacy_shape()[3] == 128:
            # print(x.memory_config())
            reshard_shard_spec = tt_lib.tensor.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange((0,0), (5,7)),ttnn.CoreRange((6,0), (6,0))}),
                [64,128],
                tt_lib.tensor.ShardOrientation.COL_MAJOR,
                False,
            )
            reshard_mem_config = tt_lib.tensor.MemoryConfig(
                tt_lib.tensor.TensorMemoryLayout.HEIGHT_SHARDED, tt_lib.tensor.BufferType.L1, reshard_shard_spec
            )
            x = tt_lib.tensor.reshard(x, reshard_mem_config)
            print(">>> Resharded")
        elif x.get_legacy_shape()[2] == 784 and x.get_legacy_shape()[3] == 256:
            # print(x.memory_config())
            reshard_shard_spec = tt_lib.tensor.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange((0,0), (5,7)),ttnn.CoreRange((6,0), (6,0))}),
                [16,256],
                tt_lib.tensor.ShardOrientation.COL_MAJOR,
                False,
            )
            reshard_mem_config = tt_lib.tensor.MemoryConfig(
                tt_lib.tensor.TensorMemoryLayout.HEIGHT_SHARDED, tt_lib.tensor.BufferType.L1, reshard_shard_spec
            )
            x = tt_lib.tensor.reshard(x, reshard_mem_config)
            print(">>> Resharded")
        elif x.get_legacy_shape()[2] == 196 and x.get_legacy_shape()[3] == 512:
            # Assuming Bilinear Upsampling is used, and not ConvTranspose2D
            # print(x.memory_config())
            reshard_shard_spec = tt_lib.tensor.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange((0,0), (0,5))}),
                [32,512],
                tt_lib.tensor.ShardOrientation.COL_MAJOR,
                False,
            )
            reshard_mem_config = tt_lib.tensor.MemoryConfig(
                tt_lib.tensor.TensorMemoryLayout.HEIGHT_SHARDED, tt_lib.tensor.BufferType.L1, reshard_shard_spec
            )
            x = tt_lib.tensor.reshard(x, reshard_mem_config)
            print(">>> Resharded")
            self.conv1_config = ttnn.Conv2dConfig(
                dtype=ttnn.bfloat16,
                weights_dtype=ttnn.bfloat16,
                math_fidelity=ttnn.MathFidelity.LoFi,
                height_sharding=False,
                input_channels_alignment=16 if self.conv1_use_shallow_conv_variant else 32,
                fp32_dest_acc_enabled=False,
                activation="relu",
                deallocate_activation=True,
                reshard_if_not_optimal=False,
                transpose_shards=False,
                packer_l1_accum_enabled=True if self.device_name == "wormhole_b0" else False,
                # act_block_h_override=32,
            )
        elif x.get_legacy_shape()[2] == 196 and x.get_legacy_shape()[3] == 1024:
            raise Exception("Code for Resharding for not Bilinear not written")
        # try:
        #     print(x.memory_config())
        # except:
        #     pass
            
        print(">>> First Conv Input: ", x.get_legacy_shape())
        # print(x.memory_config())
        out, feature_height, feature_width, self.conv1_weight_tensor, self.conv1_bias_tensor = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.conv1_weight_tensor,
            device=self.device,
            in_channels=self.in_channels,
            out_channels=self.mid_channels,
            batch_size=1,
            input_height=self.input_height,
            input_width=self.input_width,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1,1),
            groups=1,
            bias_tensor=self.conv1_bias_tensor,
            conv_config=self.conv1_config,
            conv_op_cache={},
            debug=False,
        )
        # print(out.memory_config())
        # print(out.get_legacy_shape()[2], out.get_legacy_shape()[3])

        if out.get_legacy_shape()[2] == 3136 and out.get_legacy_shape()[3] == 256:
            # print(out.memory_config())
            reshard_shard_spec = tt_lib.tensor.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange((0,0), (5,7)),ttnn.CoreRange((6,0), (6,0))}),
                [64,256],
                tt_lib.tensor.ShardOrientation.COL_MAJOR,
                False,
            )
            reshard_mem_config = tt_lib.tensor.MemoryConfig(
                tt_lib.tensor.TensorMemoryLayout.BLOCK_SHARDED, tt_lib.tensor.BufferType.L1, reshard_shard_spec
            )
            out = tt_lib.tensor.reshard(out, reshard_mem_config)
            print(">>> Resharded again")
        elif out.get_legacy_shape()[2] == 800 and out.get_legacy_shape()[3] == 512:
            reshard_shard_spec = tt_lib.tensor.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange((0,0), (7,2)),ttnn.CoreRange((0,3), (0,3))}),
                [32,512],
                tt_lib.tensor.ShardOrientation.COL_MAJOR,
                False,
            )
            reshard_mem_config = tt_lib.tensor.MemoryConfig(
                tt_lib.tensor.TensorMemoryLayout.HEIGHT_SHARDED, tt_lib.tensor.BufferType.L1, reshard_shard_spec
            )
            out = tt_lib.tensor.reshard(out, reshard_mem_config)
            print(">>> Resharded again")

            self.conv2_config = ttnn.Conv2dConfig(
                dtype=ttnn.bfloat16,
                weights_dtype=ttnn.bfloat16,
                math_fidelity=ttnn.MathFidelity.LoFi,
                height_sharding=False,
                input_channels_alignment=16 if self.conv2_use_shallow_conv_variant else 32,
                fp32_dest_acc_enabled=False,
                activation="relu",
                deallocate_activation=True,
                reshard_if_not_optimal=False,
                transpose_shards=False,
                packer_l1_accum_enabled=True if self.device_name == "wormhole_b0" else False,
                act_block_h_override=32,
            )
        
        print(">>> Second Conv Input: ", out.get_legacy_shape())
        out, feature_height, feature_width, self.conv2_weight_tensor, self.conv2_bias_tensor = ttnn.conv2d(
            input_tensor=out,
            weight_tensor=self.conv2_weight_tensor,
            device=self.device,
            in_channels=self.mid_channels,
            out_channels=self.out_channels,
            batch_size=1,
            input_height=feature_height,
            input_width=feature_width,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1,1),
            groups=1,
            bias_tensor=self.conv2_bias_tensor,
            conv_config=self.conv2_config,
            conv_op_cache={},
            debug=False,
        )
        # print(out.memory_config())

        print(">>> Double Conv Output: ", out.get_legacy_shape())

        return out, feature_height, feature_width


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        out = self.maxpool_conv(x)
        feature_height, feature_width = out.shape[2], out.shape[3]
        return out, feature_height, feature_width

class TTNN_Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, device, input_height, input_width, in_channels, out_channels):
        super().__init__()

        self.input_height = input_height
        self.input_width = input_width
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = device
        self.device_name = "wormhole_b0" # if "wormhole_b0" in os.environ.get("ARCH_NAME", os.environ.get("TT_ARCH_NAME", "")).lower() else "grayskull"

        self.max_pool = ttnn.MaxPool2d(
                kernel_size=(2, 2),
                stride=(2, 2),
                padding=(0, 0),
                dilation=(1, 1),
                dtype=ttnn.bfloat16,
                device=self.device,
                batch_size=1,
                input_height=self.input_height,
                input_width=self.input_width,
                reader_patterns_cache={},
                deallocate_activation=True,
                parallel_config_override={},
                channels=self.in_channels,
                # mesh_mapper=self.mesh_mapper,
            )

        self.downsample_output_feature_height = self.input_height // 2
        self.downsample_output_feature_width = self.input_width // 2
        self.double_conv = TTNN_DoubleConv(self.device, self.downsample_output_feature_height, self.downsample_output_feature_width, self.in_channels, self.out_channels)


    def forward(self, x):
        print(">>> Downsampling Input: ", x.get_legacy_shape())
        # print(x.memory_config())
        if x.get_legacy_shape()[2] == 3136 and x.get_legacy_shape()[3] == 256:
            # print(x.memory_config())
            reshard_shard_spec = tt_lib.tensor.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange((0,0), (5,7)),ttnn.CoreRange((6,0), (6,0))}),
                [64,256],
                tt_lib.tensor.ShardOrientation.ROW_MAJOR,
                False,
            )
            reshard_mem_config = tt_lib.tensor.MemoryConfig(
                tt_lib.tensor.TensorMemoryLayout.HEIGHT_SHARDED, tt_lib.tensor.BufferType.L1, reshard_shard_spec
            )
            x = tt_lib.tensor.reshard(x, reshard_mem_config)
            print(">>> Resharded for MaxPool2D")
        elif x.get_legacy_shape()[2] == 896 and x.get_legacy_shape()[3] == 512:
            # print(x.memory_config())
            reshard_shard_spec = tt_lib.tensor.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange((0,0), (3,6))}),
                [32,512],
                tt_lib.tensor.ShardOrientation.ROW_MAJOR,
                False,
            )
            reshard_mem_config = tt_lib.tensor.MemoryConfig(
                tt_lib.tensor.TensorMemoryLayout.HEIGHT_SHARDED, tt_lib.tensor.BufferType.L1, reshard_shard_spec
            )
            x = tt_lib.tensor.reshard(x, reshard_mem_config)
            print(">>> Resharded for MaxPool2D")
        
        # out = self.max_pool(x)
        out = ttnn.max_pool2d_new(
                input_tensor=x,
                batch_size=1,
                input_h=self.input_height,
                input_w=self.input_width,
                channels=self.in_channels,
                kernel_size=[2,2],
                stride=[2, 2],
                padding=[0, 0],
                dilation=[1, 1],
                device=self.device,
        )
        # print(out.memory_config())
        
        print(">>> Double Conv Input: ", out.get_legacy_shape())
        out, feature_height, feature_width = self.double_conv(out)
        # print(out.memory_config())

        print(">>> Downsampling Output: ", out.get_legacy_shape())

        ttnn.deallocate(x)
        
        return out, feature_height, feature_width


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class TTNN_Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, device, input_height, input_width, in_channels, out_channels, bilinear=True):
        super().__init__()

        self.input_height = input_height
        self.input_width = input_width
        self.in_channels = in_channels
        self.mid_channels = in_channels // 2 if bilinear else out_channels
        self.out_channels = out_channels
        self.device = device
        self.device_name = "wormhole_b0" # if "wormhole_b0" in os.environ.get("ARCH_NAME", os.environ.get("TT_ARCH_NAME", "")).lower() else "grayskull"
        
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            print("Set to bilinear")
        else:
            raise Exception("ConvTranspose2d not currently supported!")
        
        self.conv1_weight_tensor = torch.randn(self.mid_channels, self.in_channels, 3, 3, dtype=torch.bfloat16)
        self.conv1_weight_tensor = ttnn.from_torch(self.conv1_weight_tensor, dtype=ttnn.bfloat16)
        self.conv1_bias_tensor = None
        self.conv1_use_shallow_conv_variant = False
        self.conv1_config = ttnn.Conv2dConfig(
            dtype=ttnn.bfloat16,
            weights_dtype=ttnn.bfloat16,
            math_fidelity=ttnn.MathFidelity.LoFi,
            height_sharding=True,
            input_channels_alignment=16 if self.conv1_use_shallow_conv_variant else 32,
            fp32_dest_acc_enabled=False,
            activation="relu",
            deallocate_activation=True,
            reshard_if_not_optimal=False,
            transpose_shards=False,
            packer_l1_accum_enabled=True if self.device_name == "wormhole_b0" else False,
            act_block_h_override=32,
            enable_act_double_buffer=False,
            enable_split_reader=False,
            enable_subblock_padding=False,
        )

    def forward(self, x1, x2):
        print(">>> Upsample Layer Input: ", x1.get_legacy_shape())
        
        if x1.get_legacy_shape()[2] == 224 and x1.get_legacy_shape()[3] == 512:
            x1 = ttnn.to_memory_config(x1, ttnn.L1_MEMORY_CONFIG)
            x1 = ttnn.to_layout(x1, ttnn.ROW_MAJOR_LAYOUT)
            x1 = ttnn.reshape(x1, [1,self.input_height,self.input_width,512])
            print(">>> Upsample Input: ", x1.get_legacy_shape())
            x1 = ttnn.upsample(x1, 2)
            print(">>> Upsample Output: ", x1.get_legacy_shape())
            x1 = ttnn.reshape(x1, [1,1,self.input_height*self.input_width*4,512])
            # print()
            # reshard_shard_spec = tt_lib.tensor.ShardSpec(
            #     ttnn.CoreRangeSet({ttnn.CoreRange((0,0), (7,2)),ttnn.CoreRange((0,3), (0,3))}),
            #     [32,512],
            #     tt_lib.tensor.ShardOrientation.COL_MAJOR,
            #     False,
            # )
            # reshard_mem_config = tt_lib.tensor.MemoryConfig(
            #     tt_lib.tensor.TensorMemoryLayout.HEIGHT_SHARDED, tt_lib.tensor.BufferType.L1, reshard_shard_spec
            # )
            # x1 = ttnn.to_memory_config(x1, reshard_mem_config)
            # print(">>> Resharded for concat")
            x2 = ttnn.to_memory_config(x2, ttnn.L1_MEMORY_CONFIG)
            x2 = ttnn.to_layout(x2, ttnn.ROW_MAJOR_LAYOUT)
            # print(x1.memory_config())
            # print(x2.memory_config())

            # print(x1.get_legacy_shape(), x1.get_layout())
            # print(x2.get_legacy_shape(), x2.get_layout())
            x = ttnn.concat([x1,x2], dim=3)
            ttnn.deallocate(x1)
            ttnn.deallocate(x2)

            print(x.memory_config())
            # x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
            # x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
            reshard_shard_spec = tt_lib.tensor.ShardSpec(
                # ttnn.CoreRangeSet({ttnn.CoreRange((0,0), (7,2)), ttnn.CoreRange((0,3), (0,3))}),
                # [32,1024],
                # ttnn.CoreRangeSet({ttnn.CoreRange((0,0), (7,5)), ttnn.CoreRange((0,6), (1,6))}),
                # [32,512],
                ttnn.CoreRangeSet({ttnn.CoreRange((0,0), (6,7))}),
                [128,128],
                tt_lib.tensor.ShardOrientation.COL_MAJOR,
                False,
            )
            reshard_mem_config = tt_lib.tensor.MemoryConfig(
                tt_lib.tensor.TensorMemoryLayout.BLOCK_SHARDED, tt_lib.tensor.BufferType.L1, reshard_shard_spec
            )
            x = ttnn.to_memory_config(x, reshard_mem_config)
            print(">>> Resharded for Convolution")
            print(x.memory_config())

            print("First Conv Input = ", x.get_legacy_shape())
            print(self.input_height * 2, self.input_width * 2)
            print(self.in_channels, self.mid_channels,)
            out, feature_height, feature_width, self.conv1_weight_tensor, self.conv1_bias_tensor = ttnn.conv2d(
                input_tensor=x,
                weight_tensor=self.conv1_weight_tensor,
                device=self.device,
                in_channels=self.in_channels,
                out_channels=self.mid_channels,
                batch_size=1,
                input_height=self.input_height * 2,
                input_width=self.input_width * 2,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                dilation=(1,1),
                groups=1,
                bias_tensor=self.conv1_bias_tensor,
                conv_config=self.conv1_config,
                conv_op_cache={},
                debug=False,
            )
            print("First Conv Output = ", out.get_legacy_shape())
            exit()

        
        return out
        
        # input is NCHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)