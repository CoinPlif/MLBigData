import torch
import yaml
import json
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from utils import train_epoch, setup_model, setup_loaders, validate, upload_to_minio


def main():
    with open("params.yaml") as f:
        cfg = yaml.safe_load(f)["continue_train"]

    exp_name = cfg['exp_name']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model, criterion, optimizer, scheduler = setup_model(cfg)
    model.load_state_dict(torch.load(cfg["prev_weights"], map_location=device))
    model.to(device)
    
    train_loader, test_loader = setup_loaders(cfg)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])

    writer = SummaryWriter(log_dir=f"logs/runs/{exp_name}")

    torch.backends.cudnn.enabled = False
    last_epoch = cfg["start_epoch"] + cfg["epochs_to_add"]
    
    for epoch in range(cfg["start_epoch"], last_epoch):
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('F1/train', train_f1, epoch)

        writer.flush()

    test_loss, test_acc, test_f1, test_precision, test_recall = validate(model, test_loader, criterion, device)
    
    model_path = f"logs/artifacts/{exp_name}/model_v2.pth"
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)

    upload_to_minio(
        file_path=str(model_path),
        bucket_name="models",
        object_name=f"{exp_name}/model_v2.pth"
    )

    writer.add_scalar('Loss/test', test_loss, last_epoch)
    writer.add_scalar('Accuracy/test', test_acc, last_epoch)
    writer.add_scalar('F1/test', test_f1, epoch, last_epoch)
    writer.add_scalar('Precision/test', test_precision, last_epoch)
    writer.add_scalar('Recall/test', test_recall, last_epoch)

    writer.close()
    
    metrics = {
        "loss": test_loss,
        "accuracy": test_acc,
        "f1": test_f1,
        "precision": test_precision,
        "recall": test_recall
    }

    with open(f"logs/artifacts/{exp_name}/metrics_v2.json", "w") as f:
        json.dump(metrics, f)

    print('Another train complited')

if __name__ == "__main__":
    main()