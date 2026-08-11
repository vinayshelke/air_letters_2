import sys
import os

sys.path.insert(0, 'src')

OUTPUT_DIR = r'C:\Users\shelk\Downloads\air_letter_2\outputs\test'

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    import json
    import torch
    from torch import nn
    from airletters.config import load_config
    from airletters.data import create_dataloaders
    from airletters.models import create_model
    from airletters.utils.checkpointing import load_checkpoint
    from airletters.utils.train_eval import evaluate
    from airletters.utils.plots import save_evaluation_plots

    config = load_config('configs/digits.yaml')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    dataloaders, _ = create_dataloaders(config)
    ckpt = load_checkpoint('checkpoints/best.pt', None, device, map_only=True)
    num_classes = len(ckpt['label_to_index'])
    model = create_model(config, num_classes=num_classes).to(device)
    load_checkpoint('checkpoints/best.pt', model, device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    metrics = evaluate(
        model=model,
        dataloader=dataloaders['test'],
        criterion=criterion,
        device=device,
        split_name='test',
        collect_predictions=True,
    )

    acc = metrics['accuracy']
    n = metrics['num_samples']
    print(f'Test accuracy: {acc:.4f}  ({n} samples)')

    index_to_label = {i: l for l, i in ckpt['label_to_index'].items()}
    save_evaluation_plots(
        targets=metrics['targets'],
        predictions=metrics['predictions'],
        index_to_label=index_to_label,
        output_dir=OUTPUT_DIR,
        split_name='test',
    )
    result = {
        'accuracy': metrics['accuracy'],
        'loss': metrics['loss'],
        'num_samples': metrics['num_samples'],
    }
    with open(os.path.join(OUTPUT_DIR, 'test_metrics.json'), 'w') as f:
        json.dump(result, f, indent=2)
    print(f'Saved to {OUTPUT_DIR}')
