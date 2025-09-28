import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 使用预训练好的情感分析模型作为教师
print("加载预训练的教师模型...")
teacher_model = AutoModelForSequenceClassification.from_pretrained(
    "textattack/bert-base-uncased-imdb",  # 已在IMDB上训练好的BERT
).to(device)
teacher_tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-imdb")

# 学生模型
print("加载学生模型...")
student_model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
).to(device)
student_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# 数据集
print("加载数据集...")
dataset = load_dataset("imdb", split="train[:2000]")
eval_dataset = load_dataset("imdb", split="test[:500]")

def tokenize_function(examples):
    return student_tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)

tokenized_train = dataset.map(tokenize_function, batched=True)
tokenized_eval = eval_dataset.map(tokenize_function, batched=True)

# 蒸馏训练器
class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, temperature=3.0, alpha=0.7, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self.temperature = temperature
        self.alpha = alpha
        if self.teacher:
            self.teacher.eval()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs.loss

        if self.teacher is None:
            return (loss, outputs) if return_outputs else loss

        with torch.no_grad():
            teacher_outputs = self.teacher(**inputs)

        # KL散度蒸馏损失
        distill_loss = F.kl_div(
            F.log_softmax(outputs.logits / self.temperature, dim=-1),
            F.softmax(teacher_outputs.logits / self.temperature, dim=-1),
            reduction='batchmean'
        ) * (self.temperature ** 2)

        total_loss = self.alpha * distill_loss + (1 - self.alpha) * loss
        return (total_loss, outputs) if return_outputs else total_loss

# 评估函数
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}

# 训练参数
training_args = TrainingArguments(
    output_dir="./distilled_model",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    warmup_steps=100,
    logging_steps=50,
    save_strategy="epoch",  # 改为epoch以匹配eval_strategy
    eval_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=torch.cuda.is_available(),
    report_to="none"
)

# 训练
trainer = DistillationTrainer(
    model=student_model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    tokenizer=student_tokenizer,
    compute_metrics=compute_metrics,
    teacher_model=teacher_model,
    temperature=4.0,
    alpha=0.7
)

print("\n开始蒸馏训练...")
trainer.train()

# 评估
print("\n评估结果:")
eval_results = trainer.evaluate()
print(f"准确率: {eval_results['eval_accuracy']:.4f}")

# 保存
trainer.save_model("./final_distilled_model")
print("\n模型已保存!")

# 测试
def test_model(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        pred = torch.argmax(probs, dim=-1).item()
    return "正面" if pred == 1 else "负面", probs[0][pred].item()

test_texts = [
    "This movie is terrible!",
    "I love this film, it's amazing!",
]

print("\n测试预测:")
for text in test_texts:
    sentiment, conf = test_model(student_model, student_tokenizer, text)
    print(f"{text[:30]}: {sentiment} ({conf:.3f})")