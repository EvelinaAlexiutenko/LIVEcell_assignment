# LIVEcell dataset semantic and instance segmentation with U-net and watershed.

Based on and inspired by the study published in <a href="https://www.nature.com/articles/s41592-021-01249-6" target="_blank">Nature</a>.

The <a href="https://sartorius-research.github.io/LIVECell/">dataset</a>.
<br>
## Requirements:
Python version 3.10.11

```pip install requirements.txt```
<br>
## Details:
![image](https://github.com/user-attachments/assets/3bfaada9-9b88-4187-a600-a9376987d38b)
![image](https://github.com/user-attachments/assets/26d91195-ad5e-4590-86ec-8945537f0363)

<details>

  <summary><em>Converting the annotation from jsons in coco format to masks.</em></summary>
  As a selected model for a baseline experiment is U-net the input required to be an image and corresponding mask images. To convert .json file with annotations to mask images run:
  
  ```python convert_json_to_masks.py --json_file path/to/annotations.json --mask_output_folder path/to/mask/output --image_output_folder path/to/image/output --original_image_dir path/to/original/image```
</details>
<details>
  <summary><em>Exploratory data analysis</em></summary>
  
  Discusses the rationale behind specific train-validation-test splits for different cell types, focusing on SKOV3.
Mentions standard split ratios and includes reasoning about model performance with varying data splits. Discovering and fixing validation leakage problem.
  
You can find it in <a href="notebooks/EDA.ipynb">EDA.ipynb</a>
</details>
<details>
  <summary><em>Model selection</em></summary>
  
  U-Net was trained to perform semantic segmentation on the LIVECell dataset, generating binary masks that differentiate cell regions from the background. The loss function combined Binary Cross-Entropy with Dice loss, optimizing for both boundary accuracy and class balance.

  Following semantic segmentation, a watershed-based postprocessing pipeline was applied to achieve instance separation. 
  </details>
<details>
  <summary><em>Training model and RAM issue</em></summary>
  The training was done in Google Colab with <a href="notebooks\training_unet_watershed_keras.ipynb">training_unet_watershed_keras.ipynb</a> on T4 GPU.

  The LIVECell dataset, including its masks, totals approximately 7GB in size. During dataset preparation, a significant challenge was encountered due to the high memory usage: the dataset caused RAM to run out quickly, interrupting the data loading process. To address this issue, a data generator was implemented.

  The data generator reads and processes images in batches rather than loading the entire dataset into memory at once. This approach minimizes memory usage by only keeping a subset of data in RAM, allowing for efficient handling of large datasets. The generator iteratively loads and augments images and masks on-the-fly during model training, which ensures that the system’s memory capacity is not exceeded, even when working with the full dataset.

  This solution enabled smooth and efficient data handling, allowing the model training to proceed without memory interruptions while preserving the integrity of the entire dataset.
  During training, various configurations were tested to optimize the model's performance on the LIVECell dataset. This included experimenting with the number of epochs to find the ideal training duration that balances learning and overfitting.
  
  Weights can be found here: <a href="https://drive.google.com/drive/folders/1o1wLv-IYSjfC9_u4cjiuZnZo2VC-7BKV?usp=sharing">here</a>.
  
  As a platform for monitoring - the <a href="https://wandb.ai/site">weights and biases</a>. 
  
  The results which you can see in the preview photo can be found here: <a href="https://api.wandb.ai/links/aleksiutenko-evelina-odessa-i-i-mechnikov-national-unive/8ilo205t">here</a>.
  ![image](https://github.com/user-attachments/assets/66e8fec9-4cd4-4b40-aa6f-1c3ace1b3824)

  </details>

<details>
  <summary><em>Evaluation the model</em></summary>
  Model evaluation details can be found in the notebook <a href="notebooks\training_unet_watershed_keras.ipynb">training_unet_watershed_keras.ipynb</a>.
  
  The Average Precision (AP) metrics for the U-Net semantic segmentation model are as follows:
  
  For the model trained for 9 epochs, the AP metric was 0.8886.
  For the model trained for 29 epochs, the AP metric improved to 0.9025.
</details>
<details>
  <summary><em>Converting semantic segmentation labels to instance segmentation with the watershed.</em></summary>
  As a postprocessing technique was used in a watershed, this includes:
  
  Thresholding: The semantic segmentation output was binarized to clearly distinguish cell regions.
  
  Distance Transform: Distance transform was calculated on the binary mask, enhancing cell interiors and creating regions well-suited for watershed segmentation.
  Watershed Algorithm: Watershed segmentation was performed on the distance-transformed mask, effectively separating touching cells based on intensity valleys.
  
  Connected Component Analysis: Each segmented cell region was uniquely labelled for instance differentiation.
  
  Small-Object Removal: Small regions, likely noise, were removed to reduce false positives.
  
  Gaussian Blurring: Boundaries were smoothed with Gaussian blurring to produce cleaner final instance masks.
</details>
<details>

  <summary><em>Running a Streamlit demo to create a user-friendly interface to play around.</em></summary>
    To serve a streamlit app run:

  ```python serve_app_ngrok.py```
</details>
<details>
    <summary><em>Improvements.</em></summary>
    <ol>
        <li>Experiment with different split percentages.</li>
        <li>Evaluate the model performance after postprocessing with watershed. Compare to the benchmark.</li>
        <li>Fix AFNR metric function.</li>
        <li>Try to experiment with StratifiedShuffleSplit to balance classes in splits and change overall split percentage.</li>
        <li>Experiment with loss function.</li>
    </ol>
</details>
<details>
  <summary><em>References.</em></summary>
<ul>
    <li><a href="https://paperswithcode.com/method/u-net">U-net</a></li>
    <li><a href="https://arxiv.org/abs/1611.08303">Watershed</a></li>
    <li><a href="https://cs230.stanford.edu/projects_fall_2018/reports/12353505.pdf">More info about selected architecture (U-net and Watershed)</a></li>
    <li><a href="https://www.linkedin.com/pulse/in-depth-exploration-loss-functions-deep-learning-kiran-dev-yadav/">More info about combined loss function</a></li>
    <li><a href="https://arxiv.org/pdf/2110.08322">More info about combined loss function (PDF)</a></li>
    <li><a href="https://youtu.be/csFGTLT6_WQ?si=tAKU7mWl1aVS2zDs">Recommended video 1</a></li>
    <li><a href="https://youtu.be/lOZDTDOlqfk?si=-xD7d-Lro0EuVFMk">Recommended video 2</a></li>
</ul>


</details>
