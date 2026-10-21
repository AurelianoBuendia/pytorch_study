
import copy

import numpy as np
import torch
import tqdm


def model_train(model, device,
                X_train, y_train, X_val, y_val,
                n_epochs, batch_size, loss_fn, optimizer):
    model = model.to(device)
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)
    batch_start = torch.arange(0, len(X_train), batch_size)
    #Hold the best model
    best_acc = - np.inf #init to negative infinity
    best_weights = None
    for epoch in range(n_epochs):
        model.train()
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                #take a batch
                X_batch = X_train[start:start+batch_size]
                y_batch = y_train[start:start+batch_size]
                #forward pass
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                #backward pass
                optimizer.zero_grad()
                loss.backward()
                #update weights
                optimizer.step()
                #print progress
                acc = (y_pred.round() == y_batch).float().mean()
                bar.set_postfix(
                    loss=float(loss.detach().cpu().numpy()),
                    acc=float(acc.detach().cpu().numpy())
                )
        #evaluate accuracy at end of each epoch
        model.eval()
        y_pred = model(X_val)
        acc = (y_pred.round() == y_val).float().mean()
        acc = float(acc)
        if acc > best_acc:
            best_acc = acc
            best_weights = copy.deepcopy(model.state_dict())
    #restore model and return best accuracy
    model.load_state_dict(best_weights)
    return best_acc


def model_train_with_history(model, device, X_train, y_train, X_test, y_test,
          n_epochs, batch_size,
          loss_fn, optimizer):
    model = model.to(device)
    batches_per_epoch = len(X_train) // batch_size
    best_acc = -np.inf
    best_weights = None
    train_loss_hist = []
    train_acc_hist = []
    test_loss_hist = []
    test_acc_hist = []
    for epoch in range(n_epochs):
        epoch_loss = []
        epoch_acc = []
        model.train()
        with tqdm.trange(batches_per_epoch, unit="batch", mininterval=0) as bar:
            bar.set_description(f"Epoch {epoch + 1}/{n_epochs}")
            for i in bar:
                start = i * batch_size
                end = start + batch_size
                X_batch = X_train[start:end]
                y_batch = y_train[start:end]
                # forward pass
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                # compute and store metrics
                acc = (torch.argmax(y_pred, 1) == torch.argmax(y_batch, 1)).float().mean()
                epoch_loss.append(float(loss))
                epoch_acc.append(float(acc))
                bar.set_postfix(loss=float(loss), acc=float(acc))
        # compute test metrics for the epoch
        model.eval()
        y_pred = model(X_test)
        test_loss = loss_fn(y_pred, y_test)
        test_acc = (torch.argmax(y_pred, 1) == torch.argmax(y_test, 1)).float().mean()
        test_loss = float(test_loss)
        test_acc = float(test_acc)
        # store metrics for the epoch
        train_loss_hist.append(np.mean(epoch_loss))
        train_acc_hist.append(np.mean(epoch_acc))
        test_loss_hist.append(test_loss)
        test_acc_hist.append(test_acc)
        # update best model
        if test_acc > best_acc:
            best_acc = test_acc
            best_weights = copy.deepcopy(model.state_dict())
    # load best model weights
    model.load_state_dict(best_weights)
    return model, train_loss_hist, train_acc_hist, test_loss_hist, test_acc_hist
