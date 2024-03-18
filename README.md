## VideoMAE: Fine-tuned for UCFCrime Full Datasethis model is a fine-tuned version of [MCG-NJU/videomae-base](https://huggingface.co/MCG-NJU/videomae-base) .

This model is a fine-tuned version of [MCG-NJU/videomae-base](https://huggingface.co/MCG-NJU/videomae-base) on the [UCF-CRIME](https://paperswithcode.com/dataset/ucf-crime) dataset.
Get the trainging summary of the model from [wandb summary](https://wandb.ai/dumbal/huggingface/runs/3xelrisz) and donwload it from [huggingface repository](https://huggingface.co/archit11/videomae-base-finetuned-ucfcrime-full) or just clone this repository and run the required steps.



## Model Description

The `videomae-base-finetuned-ucfcrime-full` model is a fine-tuned version of the `MCG-NJU/videomae-base` model, which is a video representation learning model based on the Masked Autoencoders (MAE) approach. The model is designed to learn general video representations from large-scale unlabeled video data, which can be transferred to various downstream video understanding tasks through fine-tuning.

## Intended Uses & Limitations

The `videomae-base-finetuned-ucfcrime-full` model is intended for video understanding tasks related to criminal activity detection and recognition. It can be used for tasks such as video classification, action recognition, and anomaly detection in surveillance videos.

However, it's important to note that the model's performance and limitations are dependent on the specific dataset and task it was fine-tuned on. Additionally, as with any machine learning model, it may exhibit biases or limitations related to the training data or the underlying architecture.

## Training and Evaluation Data

The model was fine-tuned on the UCFCrime Full2 dataset, which contains real-world surveillance videos capturing various criminal activities. However, more information about the specific dataset split, preprocessing steps, and evaluation metrics used is needed for a complete understanding of the model's training and evaluation procedure.

## Training Procedure
procedure : (./videomae-classification-finetuneing.ipynb)[./videomae-classification-finetuneing.ipynb]
### Training Hyperparameters

The following hyperparameters were used during training:

- learning_rate: 5e-05
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- optimizer: Adam with betas=(0.9, 0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_ratio: 0.1
- training_steps: 700

### Training Results

The following table summarizes the training results for the `videomae-base-finetuned-ucfcrime-full2` model:

| Training Loss | Epoch | Step | Validation Loss | Accuracy |
|:-------------:|:-----:|:----:|:---------------:|:--------:|
| 2.5836 | 0.13 | 88 | 2.4944 | 0.2080 |
| 2.3212 | 1.13 | 176 | 2.5855 | 0.1773 |
| 2.2333 | 2.13 | 264 | 2.6270 | 0.1046 |
| 1.985 | 3.13 | 352 | 2.4058 | 0.2109 |
| 2.194 | 4.13 | 440 | 2.3654 | 0.2235 |
| 1.9796 | 5.13 | 528 | 2.2609 | 0.2235 |
| 1.8786 | 6.13 | 616 | 2.2725 | 0.2341 |
| 1.71 | 7.12 | 700 | 2.2228 | 0.2226 |


## Model description

Classifies video input into 13 Classes

## Intended uses & limitations

Can be used to detect instances of vandalism in CCTV footage.


9## Demo

![Video Demo](./media/demo_vid.mp4)

## License


This project is licensed under the [Creative Commons Attribution-NonCommercial 5.0 International License](https://creativecommons.org/licenses/by-nc/4.0/).


## TO run 
 install dependences :
`` pip install -r requirements.txt``

 run :
```python app.py```


