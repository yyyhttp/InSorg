import torch
import os
import mmcv

def featuremap_2_heatmap(feature_map):
    # 首先检查 feature_map 的维度
    print("Feature map shape:", feature_map.shape)
    
    if len(feature_map.shape) == 4:  # 如果是四维张量
        heatmap = feature_map[:, 0, :, :] * 0
    elif len(feature_map.shape) == 3:  # 如果是三维张量
        heatmap = feature_map[0, :, :] * 0
    elif len(feature_map.shape) == 2:  # 如果是二维张量
        heatmap = feature_map[:, :] * 0
    else:
        raise ValueError("Unsupported feature map dimensions")
    
    return heatmap

def draw_feature_map(model, img_path, save_dir):
    # 模型推理得到 feature_map，假设 model.forward(img) 返回 feature_map
    img = mmcv.imread(img_path)
    img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)  # 假设需要这种形状
    featuremap = model.forward(img_tensor)
    
    heatmap = featuremap_2_heatmap(featuremap)
    # 保存热力图或做进一步处理
    save_path = os.path.join(save_dir, "heatmap.png")
    mmcv.imwrite(heatmap.numpy(), save_path)

def main():
    # 模拟参数
    class Args:
        img = "path/to/image.jpg"
        save_dir = "path/to/save_dir"

    args = Args()

    # 模拟一个模型
    class DummyModel:
        def forward(self, x):
            # 返回一个模拟的 feature_map，这里我们假设是三维的 (channels, height, width)
            return torch.rand(3, 64, 64)
    
    model = DummyModel()
    draw_feature_map(model, args.img, args.save_dir)

if __name__ == "__main__":
    main()
