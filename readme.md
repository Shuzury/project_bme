# Brain MRI Classification and Automated Scan Sorting

## Project Description

This repository contains a Jupyter notebook, [project.ipynb](./project.ipynb), that trains and applies a convolutional neural network (CNN) for binary classification of brain MRI images. The notebook performs two main tasks:

1. It trains a TensorFlow/Keras model on MRI images stored in class-separated folders under `dataset/`.
2. It loads the trained model and automatically classifies new images placed in `patient/`, moving them into `results/benign/` or `results/malignant/`.

The notebook is best understood as a compact end-to-end prototype for image-based classification and automated file routing. It is suitable for learning, experimentation, and small-scale demonstrations. It is not a production-ready or clinically validated medical system.

## Repository Contents

```text
BME/
|-- dataset/
|   |-- no/              # 1,500 JPG images
|   `-- yes/             # 1,500 JPG images
|-- patient/             # Incoming images for inference
|-- results/
|   |-- benign/          # Files predicted as negative
|   `-- malignant/       # Files predicted as positive
|-- test/                # Present in the repository but not used by the notebook
|-- model.h5             # Saved trained model produced by the notebook
|-- project.ipynb        # Main notebook
`-- readme.md            # Project documentation
```

## Notebook Scope

The notebook contains five code cells and no markdown cells. The execution flow is linear:

1. Import libraries and create output directories.
2. Load the dataset, define the CNN, train the model, and save `model.h5`.
3. Load `model.h5`, classify images from `patient/`, and move them into the appropriate `results/` subfolder.
4. Display one example image from `results/malignant/`.
5. Display one example image from `results/benign/`.

## Prerequisites

Before running the notebook, ensure that the following software is available:

- Python 3.9 or later
- Jupyter Notebook or JupyterLab
- TensorFlow 2.x
- NumPy
- Matplotlib
- Pillow

The notebook was previously executed in an environment that reported TensorFlow `2.20.0`.

## Installation

### 1. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install tensorflow numpy matplotlib pillow notebook
```

If you prefer JupyterLab, you may install it as well:

```bash
pip install jupyterlab
```

## Dataset

### Local Dataset Used by the Notebook

The notebook expects a directory named `dataset/` with exactly two class folders:

```text
dataset/
|-- no/
`-- yes/
```

At the time of review, the repository contains:

- `dataset/no`: 1,500 `.jpg` files
- `dataset/yes`: 1,500 `.jpg` files

All discovered training files are JPEG images. During training, the notebook reads them from disk, rescales them to `150 x 150` pixels, and normalizes pixel values to the range `[0, 1]`.

### Dataset Source

The repository does not document the original source or citation for the dataset. For that reason, the provenance of the included images cannot be verified from repository contents alone.

If this project is to be reused for academic, research, or regulated work, the dataset source should be documented explicitly before publication or redistribution. At minimum, the following should be recorded:

- Original dataset name
- Source URL or DOI
- Licensing terms
- Class definitions for `yes` and `no`
- Any preprocessing or balancing steps performed before the data was added to this repository

### Dataset Structure and Label Semantics

The model is trained with `flow_from_directory('dataset/', class_mode='binary')`. In Keras, binary class indices are assigned alphabetically by folder name, so the effective mapping is expected to be:

- `no -> 0`
- `yes -> 1`

The notebook later interprets predictions greater than `0.5` as `Malignant` and predictions less than or equal to `0.5` as `Benign`. This means the implementation assumes:

- `yes` corresponds to the positive class routed to `results/malignant/`
- `no` corresponds to the negative class routed to `results/benign/`

That assumption should be reviewed carefully. A folder named `no` may represent "no tumor" rather than "benign tumor," which is not medically equivalent.

## Methodology and Algorithms

### Data Preparation

The notebook uses `ImageDataGenerator` with:

- `rescale=1./255`
- `validation_split=0.2`

Only the training subset is actually instantiated:

```python
train_data = datagen.flow_from_directory(
    'dataset/',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='training'
)
```

As written, the notebook trains on the 80 percent training portion of the dataset and does not create or use a validation generator, even though `validation_split` is configured.

### Model Architecture

The CNN is defined with Keras `Sequential` and contains the following layers:

1. `Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3))`
2. `MaxPooling2D(2, 2)`
3. `Conv2D(64, (3, 3), activation='relu')`
4. `MaxPooling2D(2, 2)`
5. `Flatten()`
6. `Dense(128, activation='relu')`
7. `Dropout(0.5)`
8. `Dense(1, activation='sigmoid')`

This is a straightforward CNN for binary image classification:

- Convolution layers learn local visual features.
- Pooling layers reduce spatial resolution and computation.
- The dense layer combines extracted features into a final decision representation.
- The dropout layer reduces overfitting pressure during training.
- The sigmoid output produces a scalar probability-like score for the positive class.

### Optimization

The model is compiled with:

- Optimizer: `adam`
- Loss: `binary_crossentropy`
- Metric: `accuracy`

Training is performed for `10` epochs:

```python
model.fit(train_data, epochs=10)
```

The trained model is then saved as:

```python
model.save('model.h5')
```

## Detailed Notebook Walkthrough

### Cell 0: Environment Setup

Purpose:

- Imports standard library modules: `os`, `shutil`
- Imports scientific and visualization libraries: `numpy`, `matplotlib`
- Imports TensorFlow/Keras utilities for modeling and image loading
- Creates `results/benign` and `results/malignant` if they do not already exist
- Prints the TensorFlow version

Practical effect:

- Ensures the notebook can safely write classification outputs without failing because of missing directories
- Confirms the TensorFlow runtime is available

Expected output:

```text
Libraries loaded. TensorFlow version: <version>
```

### Cell 1: Data Loading, Model Definition, Training, and Model Export

Purpose:

- Builds the image data pipeline from `dataset/`
- Resizes all images to `150 x 150`
- Normalizes pixel values
- Creates the CNN architecture
- Trains the model for 10 epochs
- Saves the trained model to `model.h5`

Key implementation details:

- `batch_size=32`
- Binary classification mode
- Only the training subset is loaded
- No explicit validation or test evaluation is performed

Expected console behavior:

- Keras reports the number of discovered training images
- Epoch-by-epoch loss and accuracy are printed
- A save confirmation message is displayed

Stored notebook output indicates a prior successful run on `2,400` training images. Accuracy rose substantially across epochs, but because the notebook does not evaluate on a held-out validation or test set, those values should be interpreted as training performance only.

### Cell 2: Patient Folder Inference and Automated File Sorting

Purpose:

- Loads the saved model from `model.h5`
- Scans the `patient/` directory for files ending in `.png`, `.jpg`, or `.jpeg`
- Preprocesses each image to the expected input size and scale
- Runs inference on one image at a time
- Moves each processed image into either `results/benign/` or `results/malignant/`

How the classification decision is made:

```python
if prediction[0] > 0.5:
    label = "Malignant"
else:
    label = "Benign"
```

Operational behavior:

- If `patient/` is empty, the cell prints a message and exits
- If files are present, each file is moved out of `patient/` after classification
- The original filename is preserved

Expected output examples:

```text
No new images found in the 'patient' folder.
```

or

```text
Processed example.jpg: Classified as Malignant
Processed sample.png: Classified as Benign
```

Important consequence:

This step is destructive in the sense that it relocates files from `patient/` into `results/`. It does not keep a duplicate in the source folder.

### Cell 3: Visual Verification of the Malignant Output Folder

Purpose:

- Reads the first file found in `results/malignant/`
- Displays it with Matplotlib

Expected behavior:

- If the folder contains at least one file, an image window or inline notebook plot is shown
- If the folder is empty, the notebook prints a message instead

Interpretation:

This cell is a simple qualitative check that the inference pipeline has populated the malignant output folder and that images remain readable after being moved.

### Cell 4: Visual Verification of the Benign Output Folder

Purpose:

- Reads the first file found in `results/benign/`
- Displays it with Matplotlib

Expected behavior:

- If the folder contains at least one file, an image is displayed
- If the folder is empty, a message is printed

Note:

The comment and fallback message in this cell still mention the malignant folder, even though the code actually reads from `results/benign/`. This is a documentation inconsistency inside the notebook code, not a functional difference in the folder path being used.

## How to Run the Notebook From Start to Finish

### Recommended Clean-Run Procedure

1. Confirm that `dataset/no` and `dataset/yes` exist and contain the intended training images.
2. Ensure your Python environment has the required dependencies installed.
3. Launch Jupyter Notebook or JupyterLab from the repository root.
4. Open [project.ipynb](./project.ipynb).
5. Run Cell 0 to load libraries and create output folders.
6. Run Cell 1 to train the CNN and generate `model.h5`.
7. Place one or more new MRI images into `patient/`.
8. Run Cell 2 to classify and move those images.
9. Run Cell 3 and Cell 4 to visually inspect one sample from each output folder.

### Launch Commands

Jupyter Notebook:

```bash
jupyter notebook
```

JupyterLab:

```bash
jupyter lab
```

### Reusing the Existing Model

The repository already contains a `model.h5` file. If you do not want to retrain the model, you may:

1. Start the notebook.
2. Run Cell 0.
3. Skip Cell 1 if the existing `model.h5` is trusted and compatible with your TensorFlow version.
4. Add images to `patient/`.
5. Run Cell 2, then Cell 3 and Cell 4.

For full reproducibility, running Cell 1 again is preferable.

## Expected Outputs and How to Interpret Them

### Training Output

During training, you should expect:

- A message indicating how many images were found
- Ten epoch logs with loss and accuracy
- Creation or overwrite of `model.h5`

Interpretation:

- Rising training accuracy suggests the model is fitting the training data
- Low training loss indicates the model is becoming more confident on the training subset
- These values do not prove good real-world performance because there is no notebook-based validation or test evaluation

### Inference Output

During scan processing, you should expect:

- Either a message indicating that `patient/` is empty
- Or one status line per processed image

Interpretation:

- Files routed to `results/malignant/` are positive-class predictions according to the model threshold
- Files routed to `results/benign/` are negative-class predictions according to the model threshold
- These folder names represent the notebook's chosen labels, not independently verified clinical conclusions

### Visualization Output

The final two cells display one image from each results folder when available. These displays are intended for quick manual inspection only. They do not provide probability scores, confidence intervals, saliency maps, or diagnostic explanation.

## Assumptions

The notebook assumes all of the following:

- `dataset/` exists in the repository root
- The training folders are named exactly `yes` and `no`
- Images can be read by Keras image utilities
- Input images should be treated as RGB images resized to `150 x 150`
- `model.h5` exists before Cell 2 is executed, unless Cell 1 is run first
- Positive predictions should be routed to `results/malignant/`
- Negative predictions should be routed to `results/benign/`

## Limitations

- The notebook performs binary classification only.
- The training pipeline does not instantiate a validation generator, despite defining `validation_split=0.2`.
- No separate test evaluation, confusion matrix, ROC analysis, or classification report is included.
- No data augmentation beyond rescaling is used.
- The model is a basic CNN and may underperform compared with modern transfer-learning approaches.
- The output terminology may be medically imprecise if `no` means "no tumor" rather than "benign."
- The scan-sorting step moves files instead of copying them.
- Existing files in `results/` can make visual verification ambiguous across repeated runs.
- The notebook contains no error handling for corrupted images, duplicate filenames, or incompatible model files.
- `model.h5` is saved in the legacy HDF5 format; modern Keras generally recommends the native `.keras` format.

## Suggested Interpretation of This Project

This project should be interpreted as a compact educational workflow demonstrating:

- Folder-based image classification with Keras
- CNN training in a notebook environment
- Automated inference over newly added files
- File-system-based routing of prediction results

It should not be interpreted as a validated medical diagnosis pipeline.

## Troubleshooting

### `No new images found in the 'patient' folder.`

Cause:

- `patient/` contains no supported image files

Resolution:

- Add `.png`, `.jpg`, or `.jpeg` files to `patient/` and rerun Cell 2

### `model.h5` cannot be loaded

Cause:

- The model has not been trained yet, was deleted, or is incompatible with the current TensorFlow/Keras version

Resolution:

- Rerun Cell 1 to generate a fresh model in the current environment

### Training appears very accurate but predictions are unreliable

Cause:

- The notebook reports only training accuracy and does not evaluate on a separate validation or test set

Resolution:

- Add a validation generator and a true test workflow before drawing performance conclusions

## Disclaimer

This repository is intended for educational and prototyping purposes. It must not be used as a sole basis for clinical decision-making, diagnosis, treatment planning, or any regulated medical workflow.
