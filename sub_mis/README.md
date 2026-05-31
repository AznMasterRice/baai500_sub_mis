# Synthetic Misalignment Observer

## Setup

```bash
pip install -r requirements.txt
```


---

## Run pipeline

### Generate synthetic data

```bash
python3 scr/generate.py --config configs/creative.yaml
```

### Train student

```bash
python3 scr/train_student.py --config configs/creative.yaml
```

### Evaluate base vs trained student

```bash
python3 scr/evaluate.py --config configs/creative.yaml
```

### Summarize results

```bash
python3 scr/summarize_results.py --input results/creative_eval.jsonl --trained-model-path results/student_creative_lora --outdir results/summary_creative
```

---

## Main experiments

### Numbers

```bash
python3 scr/generate.py --config configs/numbers.yaml
python3 scr/train_student.py --config configs/numbers.yaml
python3 scr/evaluate.py --config configs/numbers.yaml
```

### Creative

```bash
python3 scr/generate.py --config configs/creative.yaml
python3 scr/train_student.py --config configs/creative.yaml
python3 scr/evaluate.py --config configs/creative.yaml
```

### Finance

```bash
python3 scr/generate.py --config configs/finance.yaml
python3 scr/train_student.py --config configs/finance.yaml
python3 scr/evaluate.py --config configs/finance.yaml
```

### Extreme sports

```bash
python3 scr/generate.py --config configs/ex_sport.yaml
python3 scr/train_student.py --config configs/ex_sport.yaml
python3 scr/evaluate.py --config configs/ex_sport.yaml
```
