# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VGGT (Visual Geometry Grounded Transformer) is a computer vision research project that performs 3D scene reconstruction from images. It's a feed-forward neural network that predicts camera poses, depth maps, 3D point clouds, and point tracks from one or multiple input images.

## Development Environment Setup

### Installation
```bash
# Install as a package (recommended)
pip install -e .

# Or install basic dependencies
pip install -r requirements.txt

# For demo functionality (Gradio, Viser visualization)
pip install -r requirements_demo.txt
```

### Python Requirements
- Python >= 3.10
- PyTorch 2.3.1, torchvision 0.18.1
- Main dependencies: numpy, Pillow, huggingface_hub, einops, safetensors

## Core Commands

### Demo Scripts
```bash
# Interactive Gradio web interface
python demo_gradio.py

# 3D visualization with Viser
python demo_viser.py --image_folder path/to/images

# Export to COLMAP format (basic)
python demo_colmap.py --scene_dir=/path/to/scene/

# Export to COLMAP with bundle adjustment
python demo_colmap.py --scene_dir=/path/to/scene/ --use_ba
```

### Training
```bash
# Fine-tune on Co3D dataset (requires setup in training/config/default.yaml)
cd training/
torchrun --nproc_per_node=4 launch.py

# Use custom config
torchrun --nproc_per_node=4 launch.py --config custom_config
```

## Architecture Overview

### Core Components

1. **VGGT Model** (`vggt/models/vggt.py`): Main model class that orchestrates all components
   - Inherits from `PyTorchModelHubMixin` for Hugging Face integration
   - Supports selective head activation (camera, depth, point, track)

2. **Aggregator** (`vggt/models/aggregator.py`): Vision transformer backbone
   - Applies alternating attention over input frames
   - Uses patch-based image encoding with rotary position embeddings
   - Supports gradient checkpointing for memory efficiency

3. **Prediction Heads** (`vggt/heads/`):
   - `camera_head.py`: Predicts camera extrinsics/intrinsics
   - `dpt_head.py`: Depth Prediction Transformer for depth maps and 3D points
   - `track_head.py`: Point tracking across frames

4. **Utility Modules**:
   - `vggt/utils/pose_enc.py`: Camera pose encoding/decoding
   - `vggt/utils/geometry.py`: 3D geometry operations
   - `vggt/utils/load_fn.py`: Image preprocessing utilities

### Data Flow
1. Images → Aggregator → Feature tokens
2. Feature tokens → Individual heads → Predictions
3. Predictions include: camera poses, depth maps, 3D points, tracks

## Training System

### Configuration
- Uses Hydra for configuration management
- Main config: `training/config/default.yaml`
- Supports multi-dataset training through `ComposedDataset`

### Key Training Parameters
- `max_img_per_gpu`: Batch size per GPU
- `accum_steps`: Gradient accumulation steps
- Learning rate tuning critical (try 5e-6 to 5e-4 range)

### Dataset Support
- Co3D dataset (primary)
- VKitti dataset
- Custom datasets (follow Co3D annotation format)

## Coordinate Systems and Conventions

- **Camera poses**: OpenCV `camera-from-world` convention
- **Depth maps**: Aligned with corresponding camera poses
- **Image preprocessing**: Images resized to 518x518, normalized to [0,1]
- **Point tracks**: Pixel coordinates format

## Performance Considerations

### GPU Memory Usage (H100)
| Frames | Time (s) | Memory (GB) |
|--------|----------|-------------|
| 1      | 0.04     | 1.88        |
| 10     | 0.14     | 3.63        |
| 100    | 3.12     | 21.15       |

### Optimization
- Use Flash Attention 3 for better performance
- bfloat16 supported on Ampere GPUs (Compute Capability 8.0+)
- Gradient checkpointing available for training

## Key File Locations

- Model weights: Auto-downloaded from `facebook/VGGT-1B` on Hugging Face
- Training data: Configured in `training/config/default.yaml`
- Demo examples: `examples/` directory with test images
- Visualization utilities: `visual_util.py`

## Integration Points

### COLMAP Export
The `demo_colmap.py` script exports predictions to standard COLMAP format:
- `cameras.bin`, `images.bin`, `points3D.bin` in `sparse/` directory
- Compatible with gsplat and other NeRF/Gaussian splatting libraries

### Hugging Face Integration
- Model hub integration through `PyTorchModelHubMixin`
- Automatic weight downloading
- Commercial and non-commercial model versions available