import torch


def train(
        model,
        optimizer,
        criterion,
        accuracy,
        learning_rate,
        train_loader,
        val_loader,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        test_loader=None,
        epochs=10,
        early_stopping=False,
        patience=10,
        model_name='model',
):

    model = model.to(device)
    optimizer = optimizer(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")

    """
    with open(f'../logs/log_{model_name}.csv', 'w') as log:
        log.write('epoch,train_loss,val_loss,train_acc,val_acc\n')
    """

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        val_loss = 0

        es_count = 0

        accuracy.reset()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            accuracy.update(outputs, labels)

        train_acc = accuracy.compute()

        accuracy.reset()

        with torch.no_grad():
            model.eval()

            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                accuracy.update(outputs, labels)

            val_acc = accuracy.compute()

        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)

        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Loss:\t{train_loss:.4f}')
        print(f'Val Loss:\t{val_loss:.4f}',
              f'Accuracy:\t{val_acc:.4f}',
              end='\n\n')

        """
        with open(f'../logs/log_{model_name}.csv', 'w') as log:
            log.write(f'{epoch},{train_loss},{val_loss},'
                      f'{train_acc},{val_acc}\n')
        """

        if not early_stopping:
            continue

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            es_count = 0
            #torch.save(model.state_dict(), f'../models/{model_name}.pt')
        else:
            es_count += 1
            if es_count == patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

    #if not early_stopping:
        #torch.save(model.state_dict(), f'../models/{model_name}.pt')

    if test_loader is not None:
        model.eval()
        test_loss = 0
        accuracy.reset()
        with torch.no_grad():

            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                accuracy.update(outputs, labels)

            test_acc = accuracy.compute()

            test_loss = test_loss / len(test_loader)

            print(f'Test Loss:\t{test_loss:.4f}',
                  f'Test Accuracy:\t{test_acc:.4f}', sep='\n')

    #model.load_state_dict(torch.load(f'../models/{model_name}.pt'))
    return model




