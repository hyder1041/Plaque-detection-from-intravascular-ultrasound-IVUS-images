# Plaque-detection-from-intravascular-ultrasound-IVUS-images
As part of my undergraduate project, I designed and implemented a semi-automated algorithm in C++ for detecting plaque in intravascular ultrasound (IVUS) images, aiming to assist in the early diagnosis and treatment planning of cardiovascular diseases.
- Problem Context
  - IVUS imaging provides detailed cross-sectional views of coronary arteries, but manual plaque detection is labor-intensive, prone to observer bias, and affected by noise/artifacts. To address these challenges, I developed an image-processing pipeline that reduces noise, extracts lumen/media boundaries, and detects plaque morphology with higher consistency.
- Methodology
  - Dataset: 50 IVUS images from human coronary artery pullbacks (Boston Scientific iLab IVUS system, 40 MHz Atlantis SR Pro catheter).
  - Peprocessing: Resized images (512×512 → 256×256), grayscale quantization (8-bit → 3-bit), and intensity inversion to highlight lumen regions.
  - Noise Reduction: Applied erosion and dilation filters to enhance lumen and adventitia boundaries.
  - ROI Extraction: Implemented adaptive border detection and convex hull fitting around lumen to minimize the search space for plaque detection.
  - Plaque Segmentation: Applied gradient thresholding and intensity-based contour analysis to distinguish concentric vs. eccentric plaque morphology.
  - Implementation: Entire pipeline built in C++ with optimized memory handling for medical image processing.

- Results & Contributions
  - Successfully segmented lumen, media, and plaque regions with improved consistency compared to manual segmentation.
  - Demonstrated detection of both concentric and eccentric plaque across diverse IVUS images. Reduced noise and artifacts while preserving diagnostic features of images.
- Tech Stack: C++ · Medical Imaging · Image Processing · Digital Image Processing · Data Structures · Convex hull analysis · Morphological Operations · Image Segmentation

