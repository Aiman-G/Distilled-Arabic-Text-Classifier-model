# Distilled-Arabic-Text-Classifier-model


This model ( [on HuggingFace](https://huggingface.co/AimanGh/distilled_arabic_text_classifier) ) was produced in two steps: first, a BERT model was fine-tuned on an Arabic dataset ( [SANAD dataset](https://www.kaggle.com/datasets/haithemhermessi/sanad-dataset)). 
Then, the fine-tuned model was distilled.
The goal is to reduce model size and inference time while maintaining as much accuracy as possible.

![Model Accuracy](./evaluation_metrics_comparison.jpg)


![Size Comparision](./model_size_comparison.jpg)


![Inference Time Comparision](./model_time_comparison.jpg)
