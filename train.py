import argparse
import json
import os
import random
import time

import faulthandler
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

from data_provider.data_loader_emb import Dataset_Custom, Dataset_ETT_hour, Dataset_ETT_minute
from models.T3Time import TriModal
from utils.metrics import MAE, MSE, metric

faulthandler.enable()
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:150"


TEXT_ABLATION_CHOICES = [
    "full",
    "no_text",
    "zero_text",
    "mean_text",
    "shuffle_text",
    "shift_text",
    "noise_text",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda", help="device, e.g. cuda, cuda:0, cpu")
    parser.add_argument("--data_path", type=str, default="ETTm1", help="data name or csv path")
    parser.add_argument("--root_path", type=str, default=os.environ.get("T3TIME_DATA_ROOT", "./dataset/"), help="dataset directory")
    parser.add_argument(
        "--embed_root_path",
        type=str,
        default=os.environ.get("T3TIME_EMBED_ROOT", "./Embeddings/"),
        help="prompt embedding root directory",
    )
    parser.add_argument("--channel", type=int, default=32, help="number of features")
    parser.add_argument("--num_nodes", type=int, default=7, help="number of nodes")
    parser.add_argument("--seq_len", type=int, default=96, help="input length")
    parser.add_argument("--pred_len", type=int, default=96, help="prediction length")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--dropout_n", type=float, default=0.2, help="dropout rate of neural network layers")
    parser.add_argument("--d_llm", type=int, default=768, help="prompt embedding dimension")
    parser.add_argument("--e_layer", type=int, default=1, help="layers of transformer encoder")
    parser.add_argument("--d_layer", type=int, default=1, help="layers of transformer decoder")
    parser.add_argument("--head", type=int, default=8, help="heads of attention")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="weight decay rate")
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--model_name", type=str, default="gpt2", help="LLM used for precomputed embeddings")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--seed", type=int, default=2024, help="random seed")
    parser.add_argument("--es_patience", type=int, default=25, help="quit if no improvement after this many iterations")
    parser.add_argument(
        "--save",
        type=str,
        default="./logs/" + str(time.strftime("%Y-%m-%d-%H:%M:%S")) + "-",
        help="save path",
    )
    parser.add_argument(
        "--text_ablation",
        type=str,
        default="full",
        choices=TEXT_ABLATION_CHOICES,
        help="text/prompt branch intervention",
    )
    parser.add_argument("--text_shift", type=int, default=997, help="index offset used by shift_text")
    parser.add_argument("--run_id", type=str, default="", help="stable run id used by ablation scripts")
    parser.add_argument("--max_train_batches", type=int, default=0, help="debug only: limit train batches per epoch")
    parser.add_argument("--max_eval_batches", type=int, default=0, help="debug only: limit validation/test batches")
    return parser.parse_args()


class trainer:
    def __init__(
        self,
        scaler,
        channel,
        num_nodes,
        seq_len,
        pred_len,
        dropout_n,
        d_llm,
        e_layer,
        d_layer,
        head,
        lrate,
        wdecay,
        device,
        epochs,
        text_ablation,
    ):
        self.model = TriModal(
            device=device,
            channel=channel,
            num_nodes=num_nodes,
            seq_len=seq_len,
            pred_len=pred_len,
            dropout_n=dropout_n,
            d_llm=d_llm,
            e_layer=e_layer,
            d_layer=d_layer,
            head=head,
            text_ablation=text_ablation,
        )
        self.epochs = epochs
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=min(epochs, 50), eta_min=1e-6
        )
        self.loss = MSE
        self.MAE = MAE
        self.clip = 5
        print("The number of trainable parameters: {}".format(self.model.count_trainable_params()))
        print("The number of parameters: {}".format(self.model.param_num()))

    def train(self, input, mark, embeddings, real):
        self.model.train()
        self.optimizer.zero_grad()
        predict = self.model(input, mark, embeddings)
        loss = self.loss(predict, real)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = self.MAE(predict, real)
        return loss.item(), mae.item()

    def eval(self, input, mark, embeddings, real_val):
        self.model.eval()
        with torch.no_grad():
            predict = self.model(input, mark, embeddings)
        loss = self.loss(predict, real_val)
        mae = self.MAE(predict, real_val)
        return loss.item(), mae.item()


def load_data(args):
    data_map = {
        "ETTh1": Dataset_ETT_hour,
        "ETTh2": Dataset_ETT_hour,
        "ETTm1": Dataset_ETT_minute,
        "ETTm2": Dataset_ETT_minute,
    }
    data_class = data_map.get(args.data_path, Dataset_Custom)
    common_kwargs = dict(
        root_path=args.root_path,
        embed_root_path=args.embed_root_path,
        text_ablation=args.text_ablation,
        text_shift=args.text_shift,
        num_nodes=args.num_nodes,
        d_llm=args.d_llm,
        scale=True,
        size=[args.seq_len, 0, args.pred_len],
        data_path=args.data_path,
        model_name=args.model_name,
    )
    train_set = data_class(flag="train", **common_kwargs)
    val_set = data_class(flag="val", **common_kwargs)
    test_set = data_class(flag="test", **common_kwargs)

    scaler = train_set.scaler

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=args.num_workers
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=args.num_workers
    )

    return train_set, val_set, test_set, train_loader, val_loader, test_loader, scaler


def seed_it(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)


def select_device(device_arg):
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_arg)


def evaluate_test(model, test_loader, device, pred_len, max_batches=0):
    model.eval()
    test_outputs = []
    test_y = []

    with torch.no_grad():
        for batch_idx, (x, y, x_mark, y_mark, embeddings) in enumerate(test_loader):
            if max_batches and batch_idx >= max_batches:
                break
            testx = torch.Tensor(x).to(device)
            testy = torch.Tensor(y).to(device)
            testx_mark = torch.Tensor(x_mark).to(device)
            test_embedding = torch.Tensor(embeddings).to(device)
            preds = model(testx, testx_mark, test_embedding)
            test_outputs.append(preds)
            test_y.append(testy)

    test_pre = torch.cat(test_outputs, dim=0)
    test_real = torch.cat(test_y, dim=0)

    amse = []
    amae = []
    for j in range(pred_len):
        pred = test_pre[:, j,].to(device)
        real = test_real[:, j,].to(device)
        metrics = metric(pred, real)
        amse.append(float(metrics[0]))
        amae.append(float(metrics[1]))

    return {
        "test_mse": float(np.mean(amse)),
        "test_mae": float(np.mean(amae)),
        "horizon_mse": amse,
        "horizon_mae": amae,
    }


def write_metrics(path, args, best_epoch, best_valid_loss, test_metrics, train_time, val_time, status="completed"):
    metrics_path = os.path.join(path, "metrics.json")
    payload = {
        "status": status,
        "run_id": args.run_id,
        "data_path": args.data_path,
        "root_path": args.root_path,
        "embed_root_path": args.embed_root_path,
        "seq_len": args.seq_len,
        "pred_len": args.pred_len,
        "seed": args.seed,
        "text_ablation": args.text_ablation,
        "text_shift": args.text_shift,
        "channel": args.channel,
        "e_layer": args.e_layer,
        "d_layer": args.d_layer,
        "head": args.head,
        "learning_rate": args.learning_rate,
        "dropout_n": args.dropout_n,
        "batch_size": args.batch_size,
        "epochs_requested": args.epochs,
        "max_train_batches": args.max_train_batches,
        "max_eval_batches": args.max_eval_batches,
        "best_epoch": int(best_epoch),
        "best_valid_loss": float(best_valid_loss),
        "avg_train_time_per_epoch": float(np.mean(train_time)) if train_time else None,
        "avg_val_time_per_epoch": float(np.mean(val_time)) if val_time else None,
        **test_metrics,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"METRICS_JSON: {metrics_path}", flush=True)
    return payload


def run_epoch(engine, loader, device, train_mode, max_batches=0):
    losses = []
    maes = []
    for batch_idx, (x, y, x_mark, y_mark, embeddings) in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break
        batch_x = torch.Tensor(x).to(device)
        batch_y = torch.Tensor(y).to(device)
        batch_mark = torch.Tensor(x_mark).to(device)
        batch_embeddings = torch.Tensor(embeddings).to(device)
        if train_mode:
            metrics = engine.train(batch_x, batch_mark, batch_embeddings, batch_y)
        else:
            metrics = engine.eval(batch_x, batch_mark, batch_embeddings, batch_y)
        losses.append(metrics[0])
        maes.append(metrics[1])
    return float(np.mean(losses)), float(np.mean(maes))


def main():
    args = parse_args()
    train_set, val_set, test_set, train_loader, val_loader, test_loader, scaler = load_data(args)

    print()
    seed_it(args.seed)
    device = select_device(args.device)

    best_valid_loss = 9999999
    best_test_mse = 999999
    epochs_since_best_mse = 0
    bestid = 1

    if args.run_id:
        path = os.path.join(args.save, args.run_id)
    else:
        path = os.path.join(
            args.save,
            args.data_path,
            f"{args.pred_len}_{args.channel}_{args.e_layer}_{args.d_layer}_{args.learning_rate}_{args.dropout_n}_{args.seed}",
        )
    os.makedirs(path, exist_ok=True)

    his_loss = []
    val_time = []
    train_time = []
    print(args)

    engine = trainer(
        scaler=scaler,
        channel=args.channel,
        num_nodes=args.num_nodes,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        dropout_n=args.dropout_n,
        d_llm=args.d_llm,
        e_layer=args.e_layer,
        d_layer=args.d_layer,
        head=args.head,
        lrate=args.learning_rate,
        wdecay=args.weight_decay,
        device=device,
        epochs=args.epochs,
        text_ablation=args.text_ablation,
    )

    print("Start training...", flush=True)

    for i in range(1, args.epochs + 1):
        t1 = time.time()
        mtrain_loss, mtrain_mae = run_epoch(
            engine, train_loader, device, train_mode=True, max_batches=args.max_train_batches
        )
        t2 = time.time()
        train_time.append(t2 - t1)
        print("Epoch: {:03d}, Training Time: {:.4f} secs".format(i, (t2 - t1)))

        s1 = time.time()
        mvalid_loss, mvalid_mae = run_epoch(
            engine, val_loader, device, train_mode=False, max_batches=args.max_eval_batches
        )
        s2 = time.time()
        val_time.append(s2 - s1)
        print("Epoch: {:03d}, Validation Time: {:.4f} secs".format(i, (s2 - s1)))

        his_loss.append(mvalid_loss)
        print("-----------------------")
        print("Epoch: {:03d}, Train Loss: {:.4f}, Train MAE: {:.4f} ".format(i, mtrain_loss, mtrain_mae), flush=True)
        print("Epoch: {:03d}, Valid Loss: {:.4f}, Valid MAE: {:.4f}".format(i, mvalid_loss, mvalid_mae), flush=True)

        if mvalid_loss < best_valid_loss:
            print("###Update tasks appear###")
            if i <= 10:
                best_valid_loss = mvalid_loss
                torch.save(engine.model.state_dict(), os.path.join(path, "best_model.pth"))
                bestid = i
                epochs_since_best_mse = 0
                print("Updating! Valid Loss:{:.4f}, epoch: {}".format(mvalid_loss, i))
            else:
                test_metrics = evaluate_test(engine.model, test_loader, device, args.pred_len, args.max_eval_batches)
                print(
                    "On average horizons, Test MSE: {:.4f}, Test MAE: {:.4f}".format(
                        test_metrics["test_mse"], test_metrics["test_mae"]
                    )
                )

                if test_metrics["test_mse"] < best_test_mse:
                    best_test_mse = test_metrics["test_mse"]
                    best_valid_loss = mvalid_loss
                    torch.save(engine.model.state_dict(), os.path.join(path, "best_model.pth"))
                    epochs_since_best_mse = 0
                    bestid = i
                    print(
                        "Test low! Updating! Test Loss: {:.4f}, Valid Loss: {:.4f}, epoch: {}".format(
                            test_metrics["test_mse"], mvalid_loss, i
                        )
                    )
                else:
                    epochs_since_best_mse += 1
                    print("No update")
        else:
            epochs_since_best_mse += 1
            print("No update")

        engine.scheduler.step()

        if epochs_since_best_mse >= args.es_patience and i >= args.epochs // 2:
            break

    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Validation Time: {:.4f} secs".format(np.mean(val_time)))
    print("Training ends")
    print("The epoch of the best result", bestid)
    print("The valid loss of the best model", str(round(his_loss[bestid - 1], 4)))

    best_model_path = os.path.join(path, "best_model.pth")
    engine.model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_metrics = evaluate_test(engine.model, test_loader, device, args.pred_len, args.max_eval_batches)
    print("On average horizons, Test MSE: {:.4f}, Test MAE: {:.4f}".format(test_metrics["test_mse"], test_metrics["test_mae"]))
    print("FINAL_TEST_MSE: {:.8f}".format(test_metrics["test_mse"]), flush=True)
    print("FINAL_TEST_MAE: {:.8f}".format(test_metrics["test_mae"]), flush=True)
    write_metrics(path, args, bestid, his_loss[bestid - 1], test_metrics, train_time, val_time)


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))
