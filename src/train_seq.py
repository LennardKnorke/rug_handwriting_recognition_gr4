import torch


def train_sequence(
        model,
        ensemble,
        loss_function,
        optimizer_class,
        criterion,
        loader,
        epochs,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
):
    model = model.to(device)
    optimizer = optimizer_class(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        criterion.reset()
        cum_loss = 0.0
        for data in loader:
            images, target, end, dist = data
            images = images.to(device)
            target = target.to(device)
            end = end.to(device)
            dist = dist.to(device)

            letters = ensemble.forward(images)
            predictions = model.forward(letters)

            loss = loss_function(predictions, target, end, dist)
            cum_loss += loss.item()
            criterion.update(predictions[0], target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch: {epoch}, Loss: {(cum_loss / len(loader)):.4f}',
              f'Acc: {criterion.compute():.4f}', sep=' ')

