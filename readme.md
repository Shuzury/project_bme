# Brain MRI Classification and Automated Scan Sorting

## Project Description

This repository contains a Jupyter notebook, `detector.ipynb`, that trains and evaluates a convolutional neural network (CNN) for binary classification of brain MRI images and then applies that model to new scans in `patient/`.

The notebook performs these main tasks:

1. Train a CNN on MRI images stored under `dataset/`.
2. Evaluate the model using validation metrics and plots.
3. Load the saved model and classify incoming images from `patient/`.
4. Automatically move classified images into `results/benign/` or `results/malignant/`.

This project is intended as a compact prototype for image-based classification and file routing. It is suitable for learning and experimentation only, not for clinical use.

## Repository Contents

```text
Biomedical/
|-- dataset/
|   |-- no/              # negative class images
|   `-- yes/             # positive class images
|-- patient/             # incoming images for inference
|-- results/
|   |-- benign/          # files predicted as negative
|   `-- malignant/       # files predicted as positive
|-- detector.ipynb       # Main notebook
|-- model.h5             # Saved trained model produced by the notebook
`-- readme.md            # Project documentation
```

## Notebook Scope

`detector.ipynb` includes the following workflow:

1. Import libraries and create required output folders.
2. Load training and validation data from `dataset/`.
3. Define and train the CNN, then save `model.h5`.
4. Plot training and validation accuracy/loss.
5. Generate a confusion matrix.
6. Compute precision and recall.
7. Plot the ROC curve.
8. Process new images from `patient/` and sort them into `results/`.
9. Display one example from each results folder.

## Prerequisites

Before running the notebook, make sure the following software is installed:

- Python 3.9 or later
- Jupyter Notebook or JupyterLab
- TensorFlow 2.x
- NumPy
- Matplotlib
- Pillow
- scikit-learn

The notebook was previously executed in an environment with TensorFlow `2.20.0`.

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
pip install tensorflow numpy matplotlib pillow notebook scikit-learn
```

If you prefer JupyterLab, you may also install it:

```bash
pip install jupyterlab
```

## Dataset

### Local dataset structure

The notebook expects a directory named `dataset/` containing exactly two subfolders:

```text
dataset/
|-- no/
`-- yes/
```

During training, all images are resized to `150 x 150` pixels and normalized to the range `[0, 1]`.

### Label mapping

Keras assigns binary labels alphabetically, so the effective mapping is:

- `no -> 0`
- `yes -> 1`

The notebook interprets model outputs as:

- `prediction > 0.5`: `Malignant`
- `prediction <= 0.5`: `Benign`

This means the implementation assumes:

- `yes` corresponds to the positive/malignant class
- `no` corresponds to the negative/benign class

Review these class semantics carefully before using the results.

## Notebook Details

### Model training and evaluation

The notebook uses `ImageDataGenerator` with:

- `rescale=1./255`
- `validation_split=0.2`

It constructs a CNN with the following layers:

1. `Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3))`
2. `MaxPooling2D(2, 2)`
3. `Conv2D(64, (3, 3), activation='relu')`
4. `MaxPooling2D(2, 2)`
5. `Flatten()`
6. `Dense(128, activation='relu')`
7. `Dropout(0.5)`
8. `Dense(1, activation='sigmoid')`

The model is compiled with:

- Optimizer: `adam`
- Loss: `binary_crossentropy`
- Metric: `accuracy`

Training runs for `10` epochs and saves the model as `model.h5`.

### Evaluation outputs

The notebook includes:

- Training and validation accuracy and loss plots
- A confusion matrix for validation data
- Precision and recall scores
- ROC curve and AUC estimation

### Inference and file routing

The notebook loads `model.h5`, scans `patient/` for `.png`, `.jpg`, and `.jpeg` files, and moves each image to:

- `results/malignant/` if the prediction is positive
- `results/benign/` if the prediction is negative

The displayed samples from `results/malignant/` and `results/benign/` are intended for quick visual checks only.

## How to run the notebook

1. Confirm `dataset/no` and `dataset/yes` exist and contain training images.
2. Ensure dependencies are installed.
3. Start Jupyter Notebook or JupyterLab from the repository root.
4. Open `detector.ipynb`.
5. Run the cells in order from top to bottom.
6. Place one or more new images in `patient/`.
7. Run the inference cell to process the new scans.
8. Inspect the sample image displays in the final cells.

## Notes and limitations

- This repository is a prototype and not clinically validated.
- The notebook uses a simple CNN and does not include advanced augmentation.
- The model is evaluated only on a validation split from the training data.
- Files are moved from `patient/` into `results/`; the originals are not retained.
- The class semantics depend on the naming of `dataset/no` and `dataset/yes`.

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
