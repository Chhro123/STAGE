# STAGE: Segmentation-oriented Industrial Anomaly Synthesis via Graded Diffusion with Explicit Mask Alignment

**STAGE** (Segmentation-oriented Industrial Anomaly Synthesis via Graded Diffusion with Explicit Mask Alignment) is a novel **anomaly synthesis framework** tailored for **industrial anomaly segmentation**. It addresses key limitations in existing Anomaly Synthesis (AS) approaches by generating **high-fidelity**, **well-aligned**, and **fine-grained** synthetic anomalies, significantly boosting anomaly segmentation performance.

---

## 📄 Paper

If you find this work useful, please cite:
```bibtex
@inproceedings{STAGE2025,
  author = {Anonymous Author(s)},
  title = {STAGE: Segmentation-oriented Industrial Anomaly Synthesis via Graded Diffusion with Explicit Mask Alignment},
  booktitle = {Proceedings of the 39th Conference on Neural Information Processing Systems (NeurIPS)},
  year = {2025}
}
```
## 📌 Introduction
**STAGE** addresses three core challenges in anomaly synthesis for segmentation tasks:.

### 🔹 Key Challenges in Anomaly Synthesis:
- **Limited texture fidelity** – Existing methods fail to capture rich and realistic anomaly textures.
- **Misalignment between anomaly and background** – Synthetic anomalies often lack pixel-level alignment with background context.
- **Overlooked tiny anomalies** – Small or low-contrast anomalies are commonly suppressed or missed during synthesis.

### ✅ How STAGE Solves These Problems:
1. **Graded Diffusion (GD):** A dual-branch denoising strategy: the Anomaly-Aware Branch ensures contextually consistent global reconstruction and the Anomaly-Only Branch is activated during specific time windows to focus on anomaly details, And alternate sampling between branches allows small anomalies to be preserved without being overwhelmed by background information during sampling process. 
2. **Explicit Mask Alignment (EMA):** It gradually evolves the sampling mask over time steps — starting from an all-one matrix and linearly converging to the actual binary anomaly mask — enabling progressive refinement of anomaly structure and boundary alignment.
3. **Anomaly Inference (AIF):** It injects clean background features as condition into the reverse diffusion process, guiding anomaly generation within predefined regions while suppressing unnecessary background reconstruction.

STAGE achieves **state-of-the-art** (SOTA) performance on **MVTec-AD** and **BTAD** datasets, surpassing existing AS methods.

---


## 🏗️ Repository Structure
```
├── Seg_log/                 # Logs of downstream segmentation models
├── configs/                # Model configuration files
├── datasets/               # Data preprocessing scripts
├── ldm/                    # STAGE model implementation
├── taming/                    # Basic functions
├── scripts/                
├── utils/                  # Utility functions
├── main.py                 # Entry point for training and testing
├── requirements.txt        # Required Python packages
├── README.md               # This file
```

---

## 🚀 Training STAGE
To train STAGE on MVTec-AD:

```bash
python main.py --base config_file -t --actual_resume models/ldm/text2img-large/model.ckpt -n test --gpus 0, --init_word "word"  --mvtec_path='mvtec_data_path/'  --log_folder "save_log_path" 
```
For BTAD dataset:
```bash
python main.py --base config_file -t --actual_resume models/ldm/text2img-large/model.ckpt -n test --gpus 0, --init_word "word"  --mvtec_path='btad_data_path/'  --log_folder "save_log_path" 
```


## 🧐 Evaluation & Inference
Run inference on test images:

```bash
python generate_with_mask_mvtec.py --data_root='normal_data_path' --weight_idx weight_param --sample_name='save_image_folder' --init_word "word" --anomaly_name='save_image_subfolder' --pt_path='weight_path/' --mask_path='mask_path/'
```

Results will be saved in `save_image_folder/` directory.

---


## 📢 Acknowledgments
This work is built upon **Denoising Diffusion Models**, **Latent Diffusion Models (LDMs)** and **Anomaly Diffusion**. We acknowledge contributions from prior work in **anomaly segmentation**.

For questions or contributions, feel free to open an **Issue** or submit a **Pull Request**.

---

## 📜 License
This project is released under the **MIT License**.

---
