# [TSSI: Time Series as Screenshot Images for Multivariate Time Series Classification using Convolutional Neural Networks](https://www.sciencedirect.com/science/article/pii/S036083522500539X)

This repository contains the official implementation of the paper:
> YongKyung Oh, Heeyoung Kim, and Sungil Kim. TSSI: Time Series as Screenshot Images for multivariate time series classification using convolutional neural networks. Computers & Industrial Engineering, 209:111393, November 2025.

With the remarkable success of convolutional neural networks (CNNs) in computer vision, researchers have made significant efforts to leverage advanced CNN models for multivariate time series classification (MTSC). This involves the development of encoding methods that transform time series data into image inputs suitable for CNNs. However, existing encoding methods often struggle to preserve critical information, as their complex transformations can distort intra-series temporal correlations and fail to effectively capture inter-series correlations. In this paper, we propose a simple yet effective encoding method that can preserve both intra-series and inter-series correlations in multivariate time series, enabling an intuitive conversion of time series data into images and facilitating a practical and effective implementation of MTSC using CNNs. The proposed method, called Time Series as Screenshot Images (TSSI), encodes each univariate time series as a binary image by taking a screenshot of a time series plot, capturing the intra-series temporal correlations. The resulting binary images are then concatenated to form a single multi-channel image, capturing the inter-series correlations. In contrast to methods that generate abstract numerical matrices, our method directly generates intuitive visual representations designed to leverage the powerful pattern recognition capabilities of modern CNNs. As a CNN-based approach, our method enables the utilization of well-established networks such as ResNet18 and ResNet50, thereby reducing the challenges associated with developing and optimizing new architectures. We conduct extensive experiments on 26 benchmark and six real-world datasets and demonstrate the state-of-the-art classification performance of the proposed method. 

---

## Repository Structure

```
.
├── ts2img/                 # Time series to image library
├── adaption/               # Other time series-to-image encoding baselines
├── model.py                # Main code for TSSI-CNN model
├── benchmark.py            # 1D CNN-based baseline models
├── function.py             # Critical Difference Diagram code
```

---

## Citation

If you use this code or method in your research, please cite:

```bibtex
@article{oh_tssi_2025,
	title        = {{TSSI}: {Time} {Series} as {Screenshot} {Images} for multivariate time series classification using convolutional neural networks},
	author       = {Oh, YongKyung and Kim, Heeyoung and Kim, Sungil},
	year         = 2025,
	month        = nov,
	journal      = {Computers \& Industrial Engineering},
	volume       = 209,
	pages        = 111393,
	doi          = {10.1016/j.cie.2025.111393}
}
```

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
