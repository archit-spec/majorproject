# videomae-base-finetuned-fight-nofight-subset2


This model is a fine-tuned version of [MCG-NJU/videomae-base](https://huggingface.co/MCG-NJU/videomae-base) on the [Acts of Agression (CCTV footage fights)](https://huggingface.co/datasets/Pinwheel/ActsOfAgression) dataset.


Get the trainging summary of the model from [here](https://wandb.ai/dumbal/huggingface/runs/hxktifdo?workspace=user-dumbal) and donwload it from [here](https://huggingface.co/archit11/videomae-base-finetuned-fight-nofight) or just clone this repository and run the required steps.

It achieves the following results on the evaluation set:
- Loss: 0.5190
- Accuracy: 0.7435

## Model description

Classifies video input into "Fight" or "No Fight" Class

## Intended uses & limitations

Can be used to detect fights/crime in CCTV footage.

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_ratio: 0.1
- training_steps: 252
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch | Step | Validation Loss | Accuracy |
|:-------------:|:-----:|:----:|:---------------:|:--------:|
| 0.5145        | 0.25  | 64   | 0.7845           | 0.5075   |
| 0.607         | 1.25  | 128  | 0.6886           | 0.6343   |
| 0.3986        | 2.25  | 192  | 0.5106           | 0.7463   |
| 0.3632        | 3.24  | 252  | 0.7408           | 0.6716   |

### Framework versions

- Transformers 4.37.0
- PyTorch 1.2.1
- Datasets 2.1.0
- Tokenizers 0.15.1

## Demo

![Video Demo](./media/demo_vid.mp4)

## License


This project is licensed under the [Creative Commons Attribution-NonCommercial 5.0 International License](https://creativecommons.org/licenses/by-nc/4.0/).


## TO run 
 install dependences :
`` pip install -r requirements.txt``

 run :
```python app.py```


