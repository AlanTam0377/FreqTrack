\# FreqTrack: A Frequency-Enhanced Spatiotemporal Network for UAV Multi-Object Tracking

\[!\[Under Review](https://img.shields.io/badge/Status-Under\_Review-yellow.svg)]()

\[!\[Python 3.8](https://img.shields.io/badge/Python-3.8-blue.svg)]()

\[!\[PyTorch](https://img.shields.io/badge/PyTorch-1.10+-red.svg)]()



> \*\*Note:\*\* This repository contains the official core implementation of \*\*FreqTrack\*\*, which is currently under review for \*IEEE Transactions on Intelligent Transportation Systems (T-ITS)\*. 

> 

> 🚧 \*\*The full training pipeline, comprehensive evaluation scripts, and pretrained weights are undergoing internal cleanup and will be fully released upon paper acceptance.\*\* Currently, we provide the core modules (`SpectralGating`, `SIGB`, and `SGMC`) for review and reference.



\## 🌟 Architecture

!\[FreqTrack Architecture](docs/architecture.png)

\*Brief description: FreqTrack utilizes a streamlined Spectral Gating mechanism and a Spectral-Interactive Gated Block (SIGB) to achieve robust multi-object tracking in challenging UAV scenarios.\*



\## 🚀 Core Modules (Available Now)

To facilitate understanding of our proposed frequency-domain mechanisms, we have released the following core components in the `models/` directory:

\- `SpectralGating.py`: Implementation of the FFT-based global context modeling.

\- `SIGB.py`: Contains the Frequency Band Modulation (FBM) and Kernel Spatial Modulation (KSM).

\- `SGMC.py`: The Phase Correlation-based Spectral Global Motion Compensation module.



\## 📊 Visual Results

!\[Qualitative Results](docs/results.png)



\## 📩 Contact

If you have any questions, please feel free to open an issue or contact us.

