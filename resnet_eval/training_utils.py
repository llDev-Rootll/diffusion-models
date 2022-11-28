import torch
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

# Training function.
def train(model, trainloader, optimizer, criterion, device):
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    for _, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # Forward pass.
        outputs = model(image)
        # Calculate the loss.
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        # Calculate the accuracy.
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        # Backpropagation
        loss.backward()
        # Update the weights.
        optimizer.step()
    
    # Loss and accuracy for the complete epoch.
    epoch_loss = train_running_loss / counter
    # epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc

# Validation function.
def validate(model, testloader, criterion, device, nb_classes=10):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    cf = torch.zeros(nb_classes, nb_classes)
    y_true = []
    y_pred = []
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            y_true.extend(labels.cpu().numpy())
            # Forward pass.
            outputs = model(image)
            # Calculate the loss.
            loss = criterion(outputs, labels)

            valid_running_loss += loss.item()
            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)
            y_pred.extend(preds.cpu().numpy())
            valid_running_correct += (preds == labels).sum().item()

            for t, p in zip(labels.view(-1), preds.view(-1)):
                cf[t.long(), p.long()] += 1
                
    cf_matrix = confusion_matrix(y_true, y_pred)
    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / counter
    # print(len(testloader.dataset))
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))
    # print("Per class loss: ",running_loss_per_class)
    per_class_accu = cf.diag()/cf.sum(1)
    # print("Per class accuracy: ",confusion_matrix.diag()/confusion_matrix.sum(1))
    return epoch_loss, epoch_acc, per_class_accu, cf_matrix
