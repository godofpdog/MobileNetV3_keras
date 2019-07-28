def learning_rate_scheduler(epoch):
    if epoch > 50:
        lr = 1e-3
    elif epoch > 100:
        lr = 1e-4
    elif epoch > 300:
        lr = 5e-5
    elif epoch > 500:
        lr = 1e-5
    elif epoch > 800:
        lr = 1e-6
    else:
        lr = 3e-2
    return lr