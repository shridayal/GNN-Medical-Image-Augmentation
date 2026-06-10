# In train_pipeline.py, replace this line:
# trainer = TrainerGNNVAE(
#     device=DEVICE,
#     learning_rate=TRAINING_CONFIG['learning_rate']
# )

# With this:
trainer = TrainerGNNVAE(
    device=DEVICE,
    learning_rate=TRAINING_CONFIG.get('learning_rate', 1e-3)
)
