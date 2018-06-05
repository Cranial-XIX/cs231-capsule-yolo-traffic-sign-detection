import os
cmd_template = "python main.py --model {} --train_frac {}"
models = ['capsule']
fracs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

for model in models:
    for frac in fracs:
        cmd = cmd_template.format(model, frac)
        print("Executing: ", cmd)
        os.system(cmd)

