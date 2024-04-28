import torch


def train_one_epoch(model,
                    train_dl,
                    loss_fn,
                    optimizer,
                    device):
    model.train()
    model.to(device)

    loss_train = 0
    for _, (images, labels) in enumerate(train_dl):

        optimizer.zero_grad()

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = loss_fn(outputs, labels)

        loss_train += loss.item()

        loss.backward()

        optimizer.step()

    return loss_train/len(train_dl)


def test_one_epoch(model,
                    val_dl,
                    loss_fn,
                    device):
    model.eval()
    model.to(device)

    loss_val = 0
    with torch.inference_mode():

        for _, (images, labels) in enumerate(val_dl):

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            loss_val += loss.item()

    return loss_val/len(val_dl)


def train(model,
          train_dl,
          val_dl,
          loss_fn,
          optimizer,
          device,
          epochs):

    results = {
        'train_loss':[],
        'val_loss':[]
    }

    for _ in range(epochs):

        train_loss = train_one_epoch(model=model,
                                     train_dl=train_dl,
                                     loss_fn=loss_fn,
                                     optimizer=optimizer,
                                     device=device)

        val_loss = test_one_epoch(model=model,
                                   val_dl=val_dl,
                                   loss_fn=loss_fn,
                                   device=device)

        results['train_loss'].append(train_loss)
        results['val_loss'].append(val_loss)

    return results
