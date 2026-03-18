import torch
from tqdm import tqdm
import numpy as np

# Accuracy Function
def accuracy(output, target):
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(target.view_as(pred)).sum().item()
    return correct / len(target)

# Original Train method for Testing pre-existing architectures
def train(model, device, train_loader, val_loader, optimizer, lr_scheduler, num_epochs, criterion, name):
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

# Overloaded Train Method, for Custom Architecture Research
def train(model_a, model_b, device, train_loader, val_loader, optimizer_a, optimizer_b, lr_scheduler_a, lr_scheduler_b, num_epochs, criterion, name_a, name_b):
    # Training Loop
    train_losses_a, val_losses_a  = [], []
    train_accuracy_a, val_accuracy_a  = [], []

    train_losses_b, val_losses_b  = [], []
    train_accuracy_b, val_accuracy_b  = [], []

    for epoch in range(num_epochs):
        model_a.train()
        model_b.train()

        running_loss_train_a = 0.0
        running_acc_train_a = 0.0
        running_loss_train_b = 0.0
        running_acc_train_b = 0.0

        for images, labels in tqdm(train_loader):

            images, labels = images.to(device), labels.to(device)

            optimizer_a.zero_grad()
            outputs_a = model_a(images)
            loss_a = criterion(outputs_a, labels)
            loss_a.backward()
            optimizer_a.step()
            running_loss_train_a += loss_a.item() * images.size(0)
            running_acc_train_a += accuracy(outputs_a, labels)

            optimizer_b.zero_grad()
            outputs_b = model_b(images)
            loss_b = criterion(outputs_b, labels)
            loss_b.backward()
            optimizer_b.step()
            running_loss_train_b += loss_b.item() * images.size(0)
            running_acc_train_b += accuracy(outputs_b, labels)
        
        train_loss_a = running_loss_train_a / len(train_loader.dataset)
        train_losses_a.append(train_loss_a)

        train_loss_b = running_loss_train_b / len(train_loader.dataset)
        train_losses_b.append(train_loss_b)

        train_acc_a = running_acc_train_a/len(train_loader)
        train_accuracy_a.append(train_acc_a)

        train_acc_b = running_acc_train_b/len(train_loader)
        train_accuracy_b.append(train_acc_b)

        # Validation Loop
        model_a.eval()
        model_b.eval()
        running_loss_val_a = 0.0
        running_acc_val_a = 0.0
        running_loss_val_b = 0.0
        running_acc_val_b = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs_a = model_a(images)
                loss_a = criterion(outputs_a, labels)
                running_loss_val_a  += loss_a.item() * images.size(0)
                running_acc_val_a += accuracy(outputs_a, labels)

                outputs_b = model_b(images)
                loss_b = criterion(outputs_b, labels)
                running_loss_val_b  += loss_b.item() * images.size(0)
                running_acc_val_b += accuracy(outputs_b, labels)

        
        val_loss_a = running_loss_val_a / len(val_loader.dataset)
        val_losses_a.append(val_loss_a)

        val_loss_b = running_loss_val_b / len(val_loader.dataset)
        val_losses_b.append(val_loss_b)

        val_acc_a = running_acc_val_a/len(val_loader)
        val_accuracy_a.append(val_acc_a)

        val_acc_b = running_acc_val_b/len(val_loader)
        val_accuracy_b.append(val_acc_b)

        print('Adam')
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss_a:.4f}, Validation Loss: {val_loss_a:.4f}")
        print(f"Epoch {epoch+1}/{num_epochs}, Training Acc: {running_acc_train_a / len(train_loader) *100}, Validation Accuracy: {running_acc_val_a / len(val_loader)*100}")

        print('SGD')
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss_b:.4f}, Validation Loss: {val_loss_b:.4f}")
        print(f"Epoch {epoch+1}/{num_epochs}, Training Acc: {running_acc_train_b / len(train_loader) *100}, Validation Accuracy: {running_acc_val_b / len(val_loader)*100}")

        if lr_scheduler_a:
           lr_scheduler_a.step(val_loss_a)
           lr_scheduler_b.step(val_loss_b)

    torch.save(model_a.state_dict(), str(name_a))
    torch.save(model_b.state_dict(), str(name_b))
    print("Models saved successfully!")

    return (train_losses_a, val_losses_a, train_accuracy_a, val_accuracy_a,  
            train_losses_b, val_losses_b, train_accuracy_b, val_accuracy_b)

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