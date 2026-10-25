# Insiyab: Crowd Density Estimation & Early Warning

Insiyab estimates crowd density from camera images and gives early warnings when an area becomes too crowded. It is built for high-density environments such as the Holy Mosque in Makkah.

**Live demo:** https://insiyab.streamlit.app

---

## Problem

During Hajj and Umrah, crowd density in the Holy Mosque and Holy Sites changes quickly. Monitoring usually relies on staff watching camera screens, which makes it hard to notice the early signs of congestion across many cameras and areas. Late intervention can lead to bottlenecks that affect pilgrim safety and movement.

## Solution

Insiyab analyzes camera images with a deep learning model that produces a **density map** for each image:

- **Count estimation:** the sum of the density map gives the estimated number of people.
- **Density map:** shows *where* the crowd is concentrated in the image.
- **Early warning:** an alert is raised when an area exceeds a configurable threshold.
- **Zone comparison:** multiple cameras are compared side by side, sorted from most to least crowded.

The system is a **decision-support tool**. Final decisions remain with the crowd management team.

## Features

| Feature | Description |
|---|---|
| Multi-camera upload | Each uploaded image is treated as a camera in a different zone |
| Density levels | Low / Medium / High, with adjustable thresholds |
| Heatmap overlay | Predicted density map drawn on top of the image |
| High-resolution analysis (tiling) | Splits wide shots into 2×2, 3×3 or 4×4 tiles, analyzes each at full resolution, then adds up the counts |

## Model

- **Architecture:** simplified **CSRNet**. The front end is VGG16 pre-trained on ImageNet (up to `block4_conv3`), followed by a back end of dilated convolutions that outputs a density map 1/8 of the input size.
- **Dataset:** [ShanghaiTech Part A](https://www.kaggle.com/datasets/tthien/shanghaitech), a dataset of highly congested scenes (33 to 3,138 people per image).
- **Ground truth:** head annotations converted into density maps using Gaussian smoothing, with the total kept equal to the real count.
- **Input size:** 384 × 512
- **Training:** Adam (lr = 1e-5), MSE loss, horizontal flip augmentation, EarlyStopping on validation count MAE.
- **Split:** 270 train / 30 validation / 182 test. The test set was used only for the final evaluation.

## Results

Evaluated on the **182 unseen test images** of ShanghaiTech Part A (average of 433 people per image):

| Metric | Value |
|---|---|
| MAE | 93.71 |
| RMSE | 160.87 |
| MAPE | 23.84% |

For reference, published results on the same benchmark include MCNN (MAE ≈ 110) and the full CSRNet (MAE ≈ 68).

### Initial test on images from the Holy Mosque

The model was also tried on public images from the Holy Mosque, which it never saw during training. There is no ground truth for these images, so this is a qualitative check, not an accuracy measurement:

- **Localization works well:** density is placed on the pilgrims in the Mataf, while the Kaaba, buildings and sky stay low.
- **Wide shots are undercounted:** resizing a very wide image makes each person tiny. Tiling improves both the count and the coverage of the map.
- **Close-up shots can be misread:** with tiling, the model mistook the embroidered band of the Kiswah for heads. Tiling should be used for wide shots only.

## Limitations & Future Work

- **Fine-tuning on Holy Mosque data:** collect and annotate images from the Holy Mosque so the model learns the environment (Kiswah, marble floor, pillars, Ihram clothing).
- **Video streams:** analyze live camera feeds instead of still images.
- **Trend-based alerts:** warn when density is *rising quickly*, not only when it is high.
- **More training data:** random crops and additional datasets to reduce overfitting.

## Project Structure

```
Insiyab/
├── app.py                                # Streamlit app
├── requirements.txt                      # Python dependencies
├── insiyab_density.keras                 # Trained density model
└── Insiyab_ShanghaiTech_Training.ipynb   # Training notebook (Google Colab)
```

## Run Locally

```bash
conda create -n insiyab python=3.11 -y
conda activate insiyab
pip install -r requirements.txt
streamlit run app.py
```

To retrain the model, open `Insiyab_ShanghaiTech_Training.ipynb` in Google Colab with a **T4 GPU** runtime and run all cells.

## Background

This project extends a crowd counting model I developed in a Machine Learning course project, moving from a regression CNN on the Mall dataset to a density map model trained on high-density crowd scenes.

## Author

**Meran Al-Daadi**, Artificial Intelligence student, Umm Al-Qura University
Built for the Tuwaiq Club (UQU) camp, Emerging Technologies track.
