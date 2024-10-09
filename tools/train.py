import os
import argparse
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from tqdm import tqdm

from logger.logger import Logger
from dataset.CustomDataset import CustomDataset
from network.pseudo_point_generation import VILPPGen
from network.loss import LossAggregator
from metric.CustomEvaluator import CustomEvaluator
from utils import read_config
from utils.train_tools import init_path_for_training, TrainingParams
from utils.load_data import load_batch_dict_to_gpu
from utils.vis_tools import gray2jet


def build_dataset_and_dataloader(dataset_train_cfg, dataloader_train_cfg, dataset_val_cfg, dataloader_val_cfg, logger):
    logger.log_text('=' * 100)
    logger.log_text('Build Dataset And Dataloader')

    dataset_train = CustomDataset(config=dataset_train_cfg)
    logger.log_text('The length of dataset_train: {}'.format(len(dataset_train)))

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=dataloader_train_cfg['batch_size'],
        collate_fn=dataset_train.collate_batch,
        shuffle=True,
        num_workers=dataloader_train_cfg['num_workers'],
        drop_last=True,
        pin_memory=False
    )
    logger.log_text('The length of dataloader_train: {}'.format(len(dataloader_train)))

    dataset_val = CustomDataset(config=dataset_val_cfg)
    logger.log_text('The length of dataset_val: {}'.format(len(dataset_val)))

    dataloader_val = DataLoader(
        dataset_val,
        batch_size=dataloader_val_cfg['batch_size'],
        collate_fn=dataset_val.collate_batch,
        shuffle=False,
        num_workers=dataloader_val_cfg['num_workers'],
        drop_last=True,
        pin_memory=False
    )
    logger.log_text('The length of dataloader_val: {}'.format(len(dataloader_val)))

    return dataset_train, dataloader_train, dataset_val, dataloader_val


def build_network(network_cfg, devices, checkpoint, logger):
    logger.log_text('=' * 100)
    logger.log_text('Build Network')

    model = VILPPGen(config=network_cfg, devices=devices)

    if checkpoint is not None:
        logger.log_text("Load model's weights from checkpoints")
        model.load_state_dict(checkpoint['model'])

    losser = LossAggregator(weight=network_cfg['loss_weight'])

    return model, losser


def build_optimizer_and_scheduler(optimizer_cfg, scheduler_cfg, model, dataloader, max_epoch, checkpoint, logger):
    logger.log_text('=' * 100)
    logger.log_text('Build Optimizer And Scheduler')

    if optimizer_cfg['type'] == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=optimizer_cfg['lr'],
            betas=optimizer_cfg['betas'],
            eps=optimizer_cfg['eps'],
            weight_decay=optimizer_cfg['weight_decay']
        )
    else:
        raise NotImplementedError

    if scheduler_cfg['type'] == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=scheduler_cfg['max_lr'],
            total_steps=len(dataloader) * max_epoch,
            pct_start=scheduler_cfg['pct_start'],
            anneal_strategy=scheduler_cfg['anneal_strategy'],
            cycle_momentum=scheduler_cfg['cycle_momentum'],
            base_momentum=scheduler_cfg['base_momentum'],
            max_momentum=scheduler_cfg['max_momentum'],
            div_factor=scheduler_cfg['div_factor']
        )
    else:
        raise NotImplementedError

    if checkpoint is not None:
        logger.log_text("Load optimizer from checkpoints")
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    return optimizer, scheduler


def build_evaluator(evaluator_cfg, logger):
    evaluator = None

    if evaluator_cfg['enable']:
        logger.log_text('=' * 100)
        logger.log_text('Build Indicator')

        evaluator = CustomEvaluator(config=evaluator_cfg)

    return evaluator


def record_vis(batch_dict, depth_min, depth_max, epoch, batch_size, mode, logger):
    with torch.no_grad():
        images = torch.cat((
            batch_dict['vis_image'],
            gray2jet(batch_dict['vis_sparse_depth'], vmin=depth_min, vmax=depth_max),
            batch_dict['inf_image'].tile((1, 3, 1, 1)),
            gray2jet(batch_dict['inf_sparse_depth'], vmin=depth_min, vmax=depth_max),

            gray2jet(batch_dict['vis_coarse_depth'], vmin=depth_min, vmax=depth_max),
            gray2jet(batch_dict['vis_refined_depth'], vmin=depth_min, vmax=depth_max),
            gray2jet(batch_dict['vis_gt_depth'], vmin=depth_min, vmax=depth_max),

            gray2jet(batch_dict['inf_coarse_depth'], vmin=depth_min, vmax=depth_max),
            gray2jet(batch_dict['inf_refined_depth'], vmin=depth_min, vmax=depth_max),
            gray2jet(batch_dict['inf_gt_depth'], vmin=depth_min, vmax=depth_max)
        ), dim=0)

        images = F.interpolate(images, size=(256, 512), mode='bilinear')

        logger.log_image(
            mode=mode, tag=f'dataloader_{mode}[-1]', step=epoch,
            image=torchvision.utils.make_grid(images, nrow=batch_size)
        )


def save_checkpoint(checkpoints_dir, epoch, model, optimizer, scheduler, max_ckpt_save_num):
    if max_ckpt_save_num > 0:
        # check and determine whether to delete previous checkpoints or not
        checkpoints_list = [item for item in os.listdir(checkpoints_dir)
                            if not os.path.islink(os.path.join(checkpoints_dir, item))]

        if len(checkpoints_list) >= max_ckpt_save_num:
            checkpoints_list.sort(key=lambda x: (int(x.split('.')[0].replace('epoch', ''))))
            checkpoints_to_delete_list = checkpoints_list[:-max_ckpt_save_num]
            for checkpoints_to_delete in checkpoints_to_delete_list:
                os.remove(os.path.join(checkpoints_dir, checkpoints_to_delete))

    # save
    checkpoints_path = os.path.join(checkpoints_dir, 'epoch{}.pth'.format(epoch))
    state = {
        'epoch': epoch,
        'model': model,
        'optimizer': optimizer,
        'scheduler': scheduler
    }
    torch.save(state, checkpoints_path)

    # build link
    latest_path = os.path.join(checkpoints_dir, 'latest.pth')
    try:
        os.remove(latest_path)
    except:
        pass
    os.symlink(os.path.relpath(checkpoints_path, checkpoints_dir), latest_path)


def validate_one_epoch(epoch, devices, train_params, dataloader, model, losser,
                       evaluator, logger, force_vis=False, period_vis=-1):
    with torch.no_grad():
        model.eval()
        training = False

        logger.log_text('>>>> Validate Epoch')

        train_params.ready_for_next_epoch(training, epoch)

        for step, batch_dict in tqdm(enumerate(dataloader), total=len(dataloader)):
            train_params.ready_for_next_step(training, step)

            batch_dict = load_batch_dict_to_gpu(batch_dict, devices[0])

            batch_dict = model(batch_dict, training=training, need_loss=True)

            loss_dict = model.get_loss(batch_dict)

            loss, loss_aggregated_dict = losser(loss_dict)

            train_params.dump_step(training, loss_aggregated_dict)

            if evaluator is not None:
                evaluator.record(batch_dict)

        if evaluator is not None:
            metric_dict = evaluator.evaluate()
            train_params.dump_epoch(training, metric_dict)
        else:
            train_params.dump_epoch(training)

        logger.log_text(train_params.format_res(training))
        logger.log_scalars_by_dict(train_params.res_epoch_val, mode='val', step=epoch)

        if force_vis or train_params.is_epoch(period_epoch=period_vis):
            record_vis(
                batch_dict=batch_dict,
                depth_min=dataloader.dataset.depth_min,
                depth_max=dataloader.dataset.depth_max,
                epoch=epoch,
                batch_size=dataloader.batch_size,
                logger=logger,
                mode='val'
            )


def train_one_epoch(epoch, devices, train_params, dataloader, model, losser,
                    optimizer, evaluator, scheduler, logger, force_vis=False, period_vis=-1):
    model.train()
    training = True

    logger.log_text('>>>> Train Epoch {}/{}'.format(epoch + 1, train_params.epoch_max))

    train_params.ready_for_next_epoch(training, epoch)

    for step, batch_dict in tqdm(enumerate(dataloader), total=len(dataloader)):

        train_params.ready_for_next_step(training, step)

        batch_dict = load_batch_dict_to_gpu(batch_dict, devices[0])

        batch_dict = model(batch_dict, training=training, need_loss=True)

        loss_dict = model.get_loss(batch_dict)

        loss, loss_aggregated_dict = losser(loss_dict)

        train_params.dump_step(training, loss_aggregated_dict)

        if evaluator is not None:
            evaluator.record(batch_dict)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()  # step based update

        lr = optimizer.state_dict()['param_groups'][0]['lr']
        logger.log_scalar('train', 'lr', lr, train_params.epoch * len(dataloader) + step)
        scheduler.step()  # step based update

    if evaluator is not None:
        metric_dict = evaluator.evaluate()
        train_params.dump_epoch(training, metric_dict)
    else:
        train_params.dump_epoch(training)

    logger.log_text(train_params.format_res(training))
    logger.log_scalars_by_dict(train_params.res_epoch, mode='train', step=epoch)

    if force_vis or train_params.is_epoch(period_epoch=period_vis):
        record_vis(
            batch_dict=batch_dict,
            depth_min=dataloader.dataset.depth_min,
            depth_max=dataloader.dataset.depth_max,
            epoch=epoch,
            batch_size=dataloader.batch_size,
            logger=logger,
            mode='train'
        )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='../config/VIL-PPGen.yaml', help='path of config')
    return parser.parse_args()


def main():
    args = get_args()

    config = read_config(args.config_path)

    # create folders and complete file_cfg
    config['file_cfg'] = init_path_for_training(
        file_cfg=config['file_cfg'],
        config_path=args.config_path
    )

    # create logger
    logger = Logger(logs_dir=config['file_cfg']['logs_dir'])

    # log config
    logger.log_text('=' * 100)
    logger.log_text('Config\n' + yaml.dump(config))

    # select devices
    devices_list = config['gpus']
    devices = []
    logger.log_text('=' * 100)
    logger.log_text('Select Devices')
    for i, idx_gpu in enumerate(devices_list):
        devices.append(torch.device('cuda:{}'.format(idx_gpu)))
        logger.log_text('({}/{}) Select cuda:{}'.format(i + 1, len(devices_list), idx_gpu))

    tps = TrainingParams(
        file_cfg=config['file_cfg'],
        epoch_max=config['epoch_max']
    )
    logger.log_text('=' * 100)
    logger.log_text('Init TrainingParams')
    logger.log_text(tps.info_init)

    dataset_train, dataloader_train, dataset_val, dataloader_val = build_dataset_and_dataloader(
        dataset_train_cfg=config['dataset_train_cfg'],
        dataloader_train_cfg=config['dataloader_train_cfg'],
        dataset_val_cfg=config['dataset_val_cfg'],
        dataloader_val_cfg=config['dataloader_val_cfg'],
        logger=logger
    )

    model, losser = build_network(
        network_cfg=config['network_cfg'],
        devices=devices,
        checkpoint=tps.checkpoint,
        logger=logger
    )

    optimizer, scheduler = build_optimizer_and_scheduler(
        optimizer_cfg=config['optimizer_cfg'],
        scheduler_cfg=config['scheduler_cfg'],
        model=model,
        dataloader=dataloader_train,
        max_epoch=tps.epoch_max,
        checkpoint=tps.checkpoint,
        logger=logger
    )

    evaluator = build_evaluator(
        evaluator_cfg=config['evaluator_cfg'],
        logger=logger
    )

    logger.log_text('=' * 100)
    logger.log_text('Start To Train And Validate')
    for epoch in range(tps.epoch_tostart, tps.epoch_max):
        train_one_epoch(
            epoch=epoch,
            devices=devices,
            train_params=tps,
            dataloader=dataloader_train,
            model=model,
            losser=losser,
            optimizer=optimizer,
            scheduler=scheduler,
            evaluator=evaluator,
            logger=logger,
            period_vis=config['period_vis']
        )

        if tps.is_epoch(period_epoch=config['period_save']):
            save_checkpoint(
                checkpoints_dir=config['file_cfg']['checkpoints_dir'],
                epoch=epoch,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                max_ckpt_save_num=config['max_ckpt_save_num']
            )

        if tps.is_epoch(period_epoch=config['period_val']):
            validate_one_epoch(
                epoch=epoch,
                devices=devices,
                train_params=tps,
                dataloader=dataloader_val,
                model=model,
                losser=losser,
                evaluator=evaluator,
                logger=logger,
                force_vis=True
            )

    logger.log_text('Training Completed', color='green')


if __name__ == '__main__':
    if not torch.cuda.is_available():
        print('CUDA devices is not avaliable')
        exit()

    main()
