import torch


def train_one_epoch(model,
                    train_dl,
                    loss_fn,
                    optimizer,
                    device):
    model.train()
    model.to(device)

    loss_train = 0
    for _, (nums, sins) in enumerate(train_dl):

        optimizer.zero_grad()

        nums = nums.to(device)
        sins = sins.to(device)

        outputs = model(nums)
        loss = loss_fn(outputs, sins)

        loss_train += loss.item()

        loss.backward()

        optimizer.step()

    return loss_train/len(train_dl)


def test_one_epoch(model,
                    test_dl,
                    loss_fn,
                    device):
    model.eval()
    model.to(device)
    loss_val = 0

    with torch.inference_mode():

        for _, (nums, sins) in enumerate(test_dl):

            nums = nums.to(device)
            sins = sins.to(device)

            outputs = model(nums)
            loss = loss_fn(outputs, sins)

            loss_val += loss.item()

    return loss_val/len(test_dl)


def train(model,
          train_dl,
          test_dl,
          loss_fn,
          optimizer,
          device,
          epochs):

    results = {
        'train_loss':[],
        'test_loss':[]
    }

    for _ in range(epochs):

        train_loss = train_one_epoch(model=model,
                                     train_dl=train_dl,
                                     loss_fn=loss_fn,
                                     optimizer=optimizer,
                                     device=device)

        test_loss = test_one_epoch(model=model,
                                   test_dl=test_dl,
                                   loss_fn=loss_fn,
                                   device=device)

        results['train_loss'].append(train_loss)
        results['test_loss'].append(test_loss)

    return results
