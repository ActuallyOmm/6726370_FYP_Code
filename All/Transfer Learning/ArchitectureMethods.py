import torch
from tqdm import tqdm
import numpy as np
import copy

# Accuracy Function
def accuracy(output, target):
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(target.view_as(pred)).sum().item()
    return correct / len(target)

def train(model, device, train_loader, val_loader, optimizer, lr_scheduler, num_epochs, criterion, name):
    # Early Stopping Variables
    # Article used for implementation: https://medium.com/@vrunda.bhattbhatt/a-step-by-step-guide-to-early-stopping-in-tensorflow-and-pytorch-59c1e3d0e376
    best_loss = float('inf') 
    best_weights = None
    patience = 5

    # Training Loop
    train_losses, val_losses  = [], []
    train_accuracy, val_accuracy  = [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss_train = 0.0
        running_acc_train = 0.0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            running_loss_train += loss.item() * images.size(0)
            running_acc_train += accuracy(outputs, labels)

        # Validation Loop
        model.eval()
        running__loss_val = 0.0
        running_acc_val = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running__loss_val  += loss.item() * images.size(0)
                running_acc_val += accuracy(outputs, labels)

        train_loss = running_loss_train / len(train_loader.dataset)
        train_losses.append(train_loss)

        val_loss = running__loss_val / len(val_loader.dataset)
        val_losses.append(val_loss)

        train_acc = running_acc_train/len(train_loader)
        train_accuracy.append(train_acc)
        
        val_acc = running_acc_val/len(val_loader)
        val_accuracy.append(val_acc)
        # Early Stopping
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())    
            patience = 5 
        else:
            patience -= 1
            if patience == 0:
                break

        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        print(f"Epoch {epoch+1}/{num_epochs}, Training Acc: {running_acc_train / len(train_loader) *100}, Validation Accuracy: {running_acc_val / len(val_loader)*100}")

        if lr_scheduler:
           lr_scheduler.step(val_loss)
    torch.save(model.state_dict(), str(name))
    print("Model saved successfully!")

    return train_losses, val_losses, train_accuracy, val_accuracy

def test(myModel, device, criterion, data_loader, split):
    myModel.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    y_true = torch.tensor([])
    y_score = torch.tensor([])
    y_pred =  torch.tensor([])
    with torch.no_grad():
        for data, targets in data_loader:
            data, targets = data.to(device), targets.to(device)

            y_true, y_score = y_true.to(device), y_score.to(device)
            y_pred = y_pred.to(device)

            outputs = myModel(data)
            test_loss += criterion(outputs, targets).item()
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            _, pred = torch.max(outputs, 1)
            pred = pred.to(device)
            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)
            y_pred = torch.cat((y_pred, pred), 0)

        y_true = y_true.cpu()
        y_score = y_score.cpu()
        y_true = y_true.numpy()
        y_score = y_score.detach().numpy()
        y_pred = y_pred.cpu().numpy()


    avg_test_loss = test_loss / len(data_loader)
    test_accuracy = 100 * correct / total
    print(f"{split} Loss: {avg_test_loss:.4f}, {split} Accuracy: {test_accuracy:.2f}%")
    return y_true, y_pred, y_score

def weighted_loss(device, train_dataset, emotionTotal):
  # Due to class imbalance, and to prevent train and val loss diverging, weight loss used.
  # Resource used to do: https://medium.com/@ravi.abhinav4/improving-class-imbalance-with-class-weights-in-machine-learning-af072fdd4aa4
  weight_dict = train_dataset.getNoImagesInClass()
  print(weight_dict)
  print(len(train_dataset))

  for x in weight_dict:
    weight_dict[x] = (len(train_dataset) / (emotionTotal * weight_dict[x]))
  print(weight_dict)

  weight_arr = np.array([weight_dict[x] for x in sorted(weight_dict.keys())])
  print(weight_arr)
  weight_tensor = torch.tensor(weight_arr, dtype=torch.float32).to(device)
  return weight_tensor