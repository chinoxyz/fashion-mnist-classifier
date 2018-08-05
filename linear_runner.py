from trainer import run

# Best Params Linear Runner


training_batch_size = 512
validation_batch_size = 100000
test_batch_size = 10000
epochs = 20
learning_rate = 1e-2
training_show_every = 20
check_validation_every = 40
validation_split = 0.2
model_params = {
    'model_name': 'linear',
    'l2_param': 1e-5
}

run(
    training_batch_size,
    validation_batch_size,
    test_batch_size,
    model_params,
    epochs,
    learning_rate,
    training_show_every,
    check_validation_every,
    validation_split
)
