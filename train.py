import torch
import yaml
import json
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from utils import setup_model, setup_loaders, train_epoch, validate, upload_to_minio


def main():
    with open("params.yaml") as f:
        cfg = yaml.safe_load(f)["train"]
    
    exp_name = cfg['exp_name']

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    
    model, criterion, optimizer, scheduler = setup_model(cfg)
    model.to(device)
    train_loader, test_loader = setup_loaders(cfg)

    writer = SummaryWriter(log_dir=f"logs/runs/{exp_name}")

    torch.backends.cudnn.enabled = False
    num_epochs = cfg["epochs"]

    for epoch in range(num_epochs):
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('F1/train', train_f1, epoch)

        writer.flush()

    test_loss, test_acc, test_f1, test_precision, test_recall = validate(model, test_loader, criterion, device)

    model_path = f"logs/artifacts/{exp_name}/model_init.pth"
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)

    upload_to_minio(
        file_path=str(model_path),
        bucket_name="models",
        object_name=f"{exp_name}/model_init.pth"
    )

    writer.add_scalar('Loss/test', test_loss, num_epochs)
    writer.add_scalar('Accuracy/test', test_acc, num_epochs)
    writer.add_scalar('F1/test', test_f1, epoch, num_epochs)
    writer.add_scalar('Precision/test', test_precision, num_epochs)
    writer.add_scalar('Recall/test', test_recall, num_epochs)

    writer.close()
    
    metrics = {
        "loss": test_loss,
        "accuracy": test_acc,
        "f1": test_f1,
        "precision": test_precision,
        "recall": test_recall
    }
    with open(f"logs/artifacts/{exp_name}/metrics_v1.json", "w") as f:
        json.dump(metrics, f, indent=2)
        
    print("Training and test finished. Artifacts saved.")

if __name__ == "__main__":
    main()