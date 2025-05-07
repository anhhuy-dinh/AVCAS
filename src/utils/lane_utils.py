import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CULaneDataset(Dataset):
    def __init__(self, data_root, split='train', img_size=(288, 800), 
                 griding_num=400, row_anchor=None, use_aux=True):
        """
        CULane dataset for Ultra-Fast Lane Detection
        Args:
            data_root: path to CULane dataset
            split: 'train', 'val', or 'test'
            img_size: input image size (height, width)
            griding_num: number of grid cells in lateral dimension
            row_anchor: row anchors for lane detection, default is denser than original
            use_aux: whether to use auxiliary segmentation task
        """
        super(CULaneDataset, self).__init__()
        self.data_root = data_root
        self.split = split
        self.img_height, self.img_width = img_size
        self.griding_num = griding_num
        
        # Define denser row anchors if not provided (more points than original 18)
        if row_anchor is None:
            # Create denser row anchors (40 points between 100-288)
            self.row_anchor = np.linspace(100, self.img_height - 1, 40)
            self.row_anchor = self.row_anchor.astype(int)
        else:
            self.row_anchor = row_anchor
            
        self.use_aux = use_aux
        self.num_lanes = 4  # CULane has 4 lanes
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
        # Load file list according to split
        if split == 'train':
            list_path = os.path.join(data_root, 'list', 'train_gt.txt')
            if not os.path.exists(list_path):
                list_path = os.path.join(data_root, 'list', 'train.txt')
        elif split == 'val':
            list_path = os.path.join(data_root, 'list', 'val.txt')
            if not os.path.exists(list_path):
                list_path = os.path.join(data_root, 'list', 'val_gt.txt')
        else:  # test
            list_path = os.path.join(data_root, 'list', 'test.txt')
        
        # Check if list file exists
        if not os.path.exists(list_path):
            raise FileNotFoundError(f"List file not found: {list_path}")
            
        self.img_list = []
        with open(list_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    if split == 'test':
                        self.img_list.append(line)
                    else:
                        # Parse the line format: img_path label_path exist1 exist2 exist3 exist4
                        parts = line.split()
                        
                        # Handle different line formats
                        if len(parts) >= 6:  # Standard format
                            img_path = parts[0]
                            label_path = parts[1]
                            exist = ' '.join(parts[2:6])  # Get lane existence labels
                            self.img_list.append((img_path, label_path, exist))
                        elif len(parts) >= 1:  # Only image path
                            img_path = parts[0]
                            # For training, expect segmentation label in laneseg_label_w16
                            # Convert driver_xx_xxframe/xxx/xxx.jpg to laneseg_label_w16/xxx/xxx.png
                            if len(parts) >= 2:
                                label_path = parts[1]
                            else:
                                label_parts = img_path.split('/')
                                if len(label_parts) >= 2:
                                    label_path = os.path.join('laneseg_label_w16', '/'.join(label_parts[1:]))
                                    label_path = label_path.replace('.jpg', '.png')
                                else:
                                    label_path = img_path.replace('.jpg', '.png')
                            
                            # Default existence - assume all 4 lanes exist
                            exist = '1 1 1 1'
                            self.img_list.append((img_path, label_path, exist))
        
        # Clean up paths by removing leading slashes
        cleaned_img_list = []
        for item in self.img_list:
            if isinstance(item, tuple):
                img_path, label_path, exist = item
                img_path = img_path[1:] if img_path.startswith('/') else img_path
                label_path = label_path[1:] if label_path.startswith('/') else label_path
                cleaned_img_list.append((img_path, label_path, exist))
            else:
                img_path = item
                img_path = img_path[1:] if img_path.startswith('/') else img_path
                cleaned_img_list.append(img_path)
        
        self.img_list = cleaned_img_list
        print(f"Loaded {len(self.img_list)} samples for {split}")
    
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        if self.split == 'test':
            img_path = self.img_list[idx]
            img_path = os.path.join(self.data_root, img_path)
            
            try:
                img = cv2.imread(img_path)
                if img is None:
                    raise ValueError(f"Failed to load image: {img_path}")
                
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (self.img_width, self.img_height))
                img = self.img_transform(img)
                
                sample = {'img': img, 'path': img_path}
                return sample
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                # Create a dummy image as fallback
                dummy_img = torch.zeros(3, self.img_height, self.img_width)
                return {'img': dummy_img, 'path': img_path}
        else:
            # Handle different formats
            img_path, label_path, exist_label = self.img_list[idx]
            
            # Add data_root to paths
            img_path = os.path.join(self.data_root, img_path)
            label_path = os.path.join(self.data_root, label_path)
            
            try:
                # Read image
                img = cv2.imread(img_path)
                if img is None:
                    raise ValueError(f"Failed to load image: {img_path}")
                
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (self.img_width, self.img_height))
                
                # Read lane label
                if os.path.exists(label_path):
                    label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
                    label = cv2.resize(label, (self.img_width, self.img_height), 
                                      interpolation=cv2.INTER_NEAREST)
                else:
                    print(f"Warning: Label file not found: {label_path}")
                    label = np.zeros((self.img_height, self.img_width), dtype=np.uint8)
                
                # Convert exist label to binary array [4] for 4 lanes
                if isinstance(exist_label, str):
                    exist = np.array([int(x) for x in exist_label.split()])
                    # Ensure we have 4 values (pad with zeros if needed)
                    if len(exist) < 4:
                        exist = np.pad(exist, (0, 4 - len(exist)), 'constant')
                    elif len(exist) > 4:
                        exist = exist[:4]
                else:
                    exist = np.array([0, 0, 0, 0])  # Default no lane
                
                # Process data for training
                img = self.img_transform(img)
                
                # Generate lane line classification target
                cls_label = self._get_cls_label(label)
                
                sample = {
                    'img': img,
                    'cls_label': cls_label,
                    'exist_label': torch.from_numpy(exist),
                    'path': img_path
                }
                
                # Load segmentation label if needed
                if self.use_aux:
                    seg_label = label
                    sample['seg_label'] = torch.from_numpy(seg_label.astype(np.int64))
                
                return sample
            
            except Exception as e:
                print(f"Error loading sample {idx} - {img_path}: {e}")
                # Create a dummy sample as fallback
                dummy_img = torch.zeros(3, self.img_height, self.img_width)
                dummy_cls = torch.zeros(len(self.row_anchor), self.griding_num+1, self.num_lanes)
                dummy_exist = torch.zeros(4)
                
                sample = {
                    'img': dummy_img,
                    'cls_label': dummy_cls, 
                    'exist_label': dummy_exist,
                    'path': img_path
                }
                
                if self.use_aux:
                    dummy_seg = torch.zeros(self.img_height, self.img_width, dtype=torch.long)
                    sample['seg_label'] = dummy_seg
                
                return sample
    
    def _get_cls_label(self, label):
        """
        Generate classification label for each row anchor based on segmentation label
        Args:
            label: segmentation label with shape (H, W)
        Returns:
            cls_label: classification label with shape (num_rows, num_grids, num_lanes)
        """
        # Initialize classification label
        cls_label = np.zeros((len(self.row_anchor), self.griding_num+1, self.num_lanes))
        
        # Process each lane
        for i in range(1, self.num_lanes+1):
            lane_points = np.where(label == i)
            if len(lane_points[0]) == 0:
                continue
                
            # Process each row anchor
            for j, row in enumerate(self.row_anchor):
                # Find points in this row
                row_points = np.where(lane_points[0] == row)[0]
                if len(row_points) == 0:
                    continue
                
                # Get the column value (x-coordinate) of lane at this row
                col = int(np.mean(lane_points[1][row_points]))
                
                # Convert to grid index
                col_idx = int(col / self.img_width * self.griding_num)
                
                # Set the target (i-1 because lane id starts from 1 but index from 0)
                cls_label[j, col_idx, i-1] = 1
        
        return torch.from_numpy(cls_label).float()

def get_data_loader(data_root, batch_size=32, split='train', num_workers=4, 
                    img_size=(288, 800), griding_num=400, row_anchor=None, use_aux=True,
                    distributed=False):
    """
    Get data loader for CULane dataset
    Args:
        data_root: path to CULane dataset
        batch_size: batch size
        split: 'train', 'val', or 'test'
        num_workers: number of workers for data loading
        img_size: input image size
        griding_num: number of grid cells in lateral dimension
        row_anchor: row anchors for lane detection
        use_aux: whether to use auxiliary segmentation task
        distributed: whether to use distributed training
    Returns:
        data_loader: data loader for CULane dataset
    """
    dataset = CULaneDataset(
        data_root=data_root,
        split=split,
        img_size=img_size,
        griding_num=griding_num,
        row_anchor=row_anchor,
        use_aux=use_aux
    )
    
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            sampler=sampler
        )
    else:
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=(split == 'train'),
            pin_memory=True
        )
    
    return data_loader

def visualize_dataset(data_loader, save_dir='./visualization', num_samples=5):
    """
    Visualize dataset samples
    Args:
        data_loader: data loader for CULane dataset
        save_dir: directory to save visualization results
        num_samples: number of samples to visualize
    """
    import matplotlib.pyplot as plt
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for i, data in enumerate(data_loader):
        if i >= num_samples:
            break
        
        img = data['img'][0].permute(1, 2, 0).numpy()
        # De-normalize
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        plt.figure(figsize=(12, 6))
        plt.imshow(img)
        
        # Draw lane existence
        if 'exist_label' in data:
            exist = data['exist_label'][0].numpy()
            for lane_idx, exists in enumerate(exist):
                if exists:
                    plt.text(50 + lane_idx * 100, 30, f"Lane {lane_idx+1}: Yes", 
                             color='green', fontsize=12)
                else:
                    plt.text(50 + lane_idx * 100, 30, f"Lane {lane_idx+1}: No", 
                             color='red', fontsize=12)
        
        # Draw lane points from cls_label
        if 'cls_label' in data:
            cls_label = data['cls_label'][0].numpy()
            row_anchors = data_loader.dataset.row_anchor
            
            for lane_idx in range(cls_label.shape[2]):
                lane_points = []
                for row_idx, row in enumerate(row_anchors):
                    if np.max(cls_label[row_idx, :, lane_idx]) > 0:
                        col_idx = np.argmax(cls_label[row_idx, :, lane_idx])
                        if col_idx < data_loader.dataset.griding_num:
                            x = col_idx * data_loader.dataset.img_width / data_loader.dataset.griding_num
                            lane_points.append((x, row))
                
                if lane_points:
                    lane_points = np.array(lane_points)
                    plt.scatter(lane_points[:, 0], lane_points[:, 1], color=f'C{lane_idx}', s=10)
                    plt.plot(lane_points[:, 0], lane_points[:, 1], color=f'C{lane_idx}', linewidth=2)
        
        plt.title(f"Sample {i}")
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, f"sample_{i}.png"))
        plt.close()
        
        print(f"Saved visualization for sample {i}")