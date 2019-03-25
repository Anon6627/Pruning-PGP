from utils import train, eval
import time

def btrain(model, device, trainloader, testloader, optimizer, epochs, **kwargs):

    logger = kwargs["logger"]
    if not "logger_id" in kwargs:
        logger_id = ""
    else:
        logger_id = kwargs["logger_id"]

    scheduler = None
    if "scheduler"in kwargs:
        scheduler = kwargs["scheduler"]


    for epoch in range(1, epochs + 1):
        if scheduler is not None:
            scheduler.step()

        start = time.clock()
        total_loss = train(model, optimizer, device, trainloader)
        end = time.clock()
        acc = eval(model, device, testloader)
        if logger is not None:
            logger.log_scalar("baseline_{}_epoch_time".format(logger_id), time.clock() - end, epoch)
            logger.log_scalar("baseline_{}_training_loss".format(logger_id), total_loss, epoch)
            logger.log_scalar("baseline_{}_before_target_val_acc".format(logger_id), acc, epoch)

    return model, optimizer