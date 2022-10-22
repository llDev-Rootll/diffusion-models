import torch

from tqdm import tqdm
from torchvision.transforms import ToTensor

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
        # break
    
    # Loss and accuracy for the complete epoch.
    epoch_loss = train_running_loss / counter
    # epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc

# Validation function.
def validate(model, testloader, criterion, device):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    # running_loss_per_class = [0]*10
    running_loss_per_class = [0 for c in range(10)]
    # print(running_loss_per_class)

    nb_classes = 10

    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    # with torch.no_grad():
    #     for i, (inputs, classes) in enumerate(testloader):
    #         inputs = inputs.to(device)
    #         classes = classes.to(device)
    #         outputs = model_ft(inputs)
    #         _, preds = torch.max(outputs, 1)

    # print(confusion_matrix)

    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            # Forward pass.
            outputs = model(image)
            # Calculate the loss.
            loss = criterion(outputs, labels)
            # print(i)
            # print("labels: ",labels)
            # print("labels: ",labels.shape)
            valid_running_loss += loss.item()
            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)
            # print("pred:",preds.shape)
            # print("pred:",preds)
            valid_running_correct += (preds == labels).sum().item()

            for t, p in zip(labels.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
        # print(confusion_matrix)
        # break



            # print(labels.type)
            # for c in range(10):
            #     num = ((preds == labels) * (labels == c)).float()
            #     # denom = (torch.max(labels == c).sum(), 1)
            #     denom = torch.tensor([max(labels == c).sum()])
                
            #     if(denom==0 or torch.sum(num)==0):
            #         continue
            #     print("num:",num.shape)
            #     print("num:",num)

            #     print("num sum:",torch.sum(num))
                
            #     print("denom:", denom)
            #     print("denom:",denom.shape)
            #     running_loss_per_class[c] += torch.div(num, denom)
            #     break
            # break
        
    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / counter
    # print(len(testloader.dataset))
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))
    # print("Per class loss: ",running_loss_per_class)
    per_class_accu = confusion_matrix.diag()/confusion_matrix.sum(1)
    # print("Per class accuracy: ",confusion_matrix.diag()/confusion_matrix.sum(1))
    return epoch_loss, epoch_acc, per_class_accu
