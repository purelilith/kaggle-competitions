# Flower Classification — ResNet50 (PyTorch)

Solution for the **Petals to the Metal** competition (TPU Flowers dataset, 104 classes), based on fine-tuning a pretrained **ResNet50** model.

**Public score: `0.92104`**

## Data

The source data is stored in **TFRecord** format (224x224 JPEG images, 104 flower classes):

```
/kaggle/input/competitions/tpu-getting-started/tfrecords-jpeg-224x224/{train,val,test}/*.tfrec
```

Since TFRecord is not directly usable with a PyTorch `Dataset`, the first stage of the pipeline parses the TFRecord files with `tf.data.TFRecordDataset`, extracts the images to disk as `.jpeg` files, and stores the labels in `train.csv` / `val.csv`.

## Pipeline

1. **Extracting data from TFRecord**
   - Parsing `id`/`class` (train, val) and `id`/`image` (train, val, test)
   - Saving images to `/kaggle/working/dataset/images/{train,val,test}`
   - Building `train.csv` and `val.csv`

2. **Dataset and augmentations (PyTorch)**
   - `train`: `RandomResizedCrop(224)` + `RandomHorizontalFlip` + normalization using ImageNet statistics
   - `val`/`test`: `Resize(224, 224)` + normalization, no augmentations

3. **Model**
   - `ResNet50` with pretrained `ImageNet` weights (`ResNet50_Weights.DEFAULT`)
   - The final `fc` layer is replaced with `Linear(in_features, 104)` to match the number of classes

4. **Training**
   - Loss: `CrossEntropyLoss`
   - Optimizer: `Adam(lr=1e-4)`
   - Scheduler: `ReduceLROnPlateau` (`factor=0.1`, `patience=1`)
   - Custom **EarlyStopping** (`patience=3`) that saves the best weights (`best_model.pth`)
   - Up to 20 epochs, with the best checkpoint reloaded after training

5. **Inference**
   - Running the test loader through the best model
   - Building `submission.csv` (`id`, `label`)

## Stack

`Python`, `PyTorch`, `torchvision`, `TensorFlow` (used only for reading TFRecord), `pandas`, `PIL`

## Result

| Metric | Value |
|---|---|
| Public score | **0.92104** |
