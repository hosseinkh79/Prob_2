import torch

from make_modular.utils import calculate_f1Score_recall_precision


def train_one_epoch(model,
                    train_dl,
                    loss_fn,
                    optimizer,
                    device):
    model.train()
    model.to(device)

    train_loss, train_acc = 0, 0
    train_f1_score, train_recall, train_precision = 0, 0, 0
    num_classes = 10

    for i, (images, labels) in enumerate(train_dl):

        optimizer.zero_grad()

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = loss_fn(outputs, labels)


        # calculate accuracy
        _, indices = torch.max(outputs, dim=1)
        acc = torch.sum(indices == labels)/len(images)
        train_acc += acc.item()

        # calculate f1_score, recall, precision
        labels = labels.detach().cpu().numpy()
        indices = indices.detach().cpu().numpy()
        f1_score, recall, precision = calculate_f1Score_recall_precision(preds=indices, 
                                                                         labels=labels, 
                                                                         num_classes=num_classes)
        train_f1_score += f1_score
        train_recall += recall
        train_precision += precision

        # if i % 30 == 0 :
        #     print(f'loss in {i} {loss.item()}')
            
        train_loss += loss.item()

        loss.backward()

        optimizer.step()

    train_loss = train_loss/len(train_dl)
    train_acc = train_acc/len(train_dl)
    train_f1_score = train_f1_score/(len(train_dl))

    return train_loss, train_acc, train_f1_score


def test_one_epoch(model,
                    val_dl,
                    loss_fn,
                    device):
    model.eval()
    model.to(device)

    val_loss, val_acc = 0, 0
    val_f1_score, val_recall, val_precision = 0, 0, 0
    num_classes = 10

    with torch.inference_mode():

        for _, (images, labels) in enumerate(val_dl):

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            # calculate accuracy
            _, indices = torch.max(outputs, dim=1)
            acc = torch.sum(indices == labels)/len(images)
            val_acc += acc.item()

            # calculate f1_score, recall, precision
            labels = labels.detach().cpu().numpy()
            indices = indices.detach().cpu().numpy()
            f1_score, recall, precision = calculate_f1Score_recall_precision(preds=indices, 
                                                                             labels=labels, 
                                                                             num_classes=num_classes)
            val_f1_score += f1_score
            val_recall += recall
            val_precision += precision


            val_loss += loss.item()

    val_loss = val_loss/len(val_dl)
    val_acc = val_acc/len(val_dl)
    val_f1_score = val_f1_score/len(val_dl)

    return val_loss, val_acc, val_f1_score

# import wandb
import wandb

def train(model,
          train_dl,
          val_dl,
          loss_fn,
          optimizer,
          device,
          epochs,
          save_wandb=None,
          project_name=None,
          experiment_name=None,
          hyper_param_config=None):
    
# ------------------------------------------------------------------------------------------------
    if save_wandb : 
        wandb.init(
        # Set the project where this run will be logged
        project=project_name, 
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=f"exp_{experiment_name}", 
        # Track hyperparameters and run metadata
        config=hyper_param_config
        )
# --------------------------------------------------------------------------------

    results = {
        'train_loss':[],
        'val_loss':[],
        'train_acc':[],
        'val_acc':[],
        'train_f1_score':[],
        'val_f1_score':[]
    }

    for i in range(epochs):

        train_loss, train_acc, train_f1_score = train_one_epoch(model=model,
                                     train_dl=train_dl,
                                     loss_fn=loss_fn,
                                     optimizer=optimizer,
                                     device=device)

        val_loss, val_acc, val_f1_score = test_one_epoch(model=model,
                                   val_dl=val_dl,
                                   loss_fn=loss_fn,
                                   device=device)
# --------------------------------------------------------------------------------
        if save_wandb : 
            wandb.log({"train_loss": train_loss, 
                        "val_loss": val_loss,
                        "train_acc": train_acc,
                        "val_acc": val_acc,
                        "train_f1_score": train_f1_score,
                        "val_f1_score": val_f1_score,
                        })
            
# --------------------------------------------------------------------------------

        print(f'epoch {i+1}/{epochs} | '
              f'train_loss:{train_loss:.2f} | '
              f'val_loss:{val_loss:.2f} | '
              # f'train_acc:{train_acc:.3f} | '
              # f'val_acc:{val_acc:.3f} | '
              f'train_f1_score:{train_f1_score:.3f} | '
              f'val_f1_score:{val_f1_score:.3f}'
              )
    
        results['train_loss'].append(train_loss)
        results['val_loss'].append(val_loss)    
        results['train_acc'].append(train_acc)
        results['val_acc'].append(val_acc)
        results['train_f1_score'].append(train_f1_score)
        results['val_f1_score'].append(val_f1_score)

# --------------------------------------------------------------------------------
    if save_wandb :
        wandb.finish()
# --------------------------------------------------------------------------------

    return results
