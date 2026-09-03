from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_LATENT_DIM = 64
DEFAULT_HIDDEN_DIM = 128


# ============================================================
# ENCODERS
# ============================================================

class Encoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=DEFAULT_HIDDEN_DIM,
        latent_dim=DEFAULT_LATENT_DIM
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, x):
        return self.network(x)


# ============================================================
# LEARNED ROUTER
# ============================================================

class LearnedRouter(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


# ============================================================
# CONTEXTUAL ATTENUATION
# ============================================================

class ContextualAttenuation(nn.Module):
    def __init__(
        self,
        input_dim,
        latent_dim=DEFAULT_LATENT_DIM
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
            nn.Sigmoid()
        )

    def forward(self, z, context):
        scale = self.network(context)
        return z * scale


# ============================================================
# PSEUDO-CLONE GENERATOR
# ============================================================

class PseudoCloneGenerator(nn.Module):
    def __init__(self, noise_std=0.05):
        super().__init__()
        self.noise_std = noise_std

    def forward(self, z):
        noise = torch.randn_like(z) * self.noise_std
        return z + noise


# ============================================================
# CRUSH-NET
# ============================================================

class CRUSHNet(nn.Module):

    def __init__(
        self,
        input_dim,
        num_classes=2,
        hidden_dim=DEFAULT_HIDDEN_DIM,
        latent_dim=DEFAULT_LATENT_DIM,
        clone_noise=0.05
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.latent_dim = latent_dim

        self.encoder_major = Encoder(
            input_dim,
            hidden_dim,
            latent_dim
        )

        self.encoder_minor = Encoder(
            input_dim,
            hidden_dim,
            latent_dim
        )

        self.router = LearnedRouter(
            input_dim
        )

        self.attn = ContextualAttenuation(
            input_dim,
            latent_dim
        )

        self.pseudo_gen = PseudoCloneGenerator(
            clone_noise
        )

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(32, num_classes)
        )

    def encode(self, x):
        z_major = self.encoder_major(x)
        z_minor = self.encoder_minor(x)

        gate = self.router(x)

        z = (
            (1.0 - gate) * z_major
            + gate * z_minor
        )

        z = self.attn(z, x)

        return z, gate

    def forward(
        self,
        x,
        y=None,
        generate_clones=False
    ):

        z, gate = self.encode(x)

        logits = self.classifier(z)

        output = {
            "logits": logits,
            "latent": z,
            "gate": gate
        }

        if (
            generate_clones
            and y is not None
        ):

            minority_mask = y == 1

            if minority_mask.any():

                minority_z = z[minority_mask]

                clone_z = self.pseudo_gen(
                    minority_z
                )

                clone_logits = self.classifier(
                    clone_z
                )

                clone_targets = y[
                    minority_mask
                ]

                output["clone_logits"] = clone_logits
                output["clone_targets"] = clone_targets

            else:

                output["clone_logits"] = None
                output["clone_targets"] = None

        return output


# ============================================================
# LOSS
# ============================================================

def compute_loss(
    output,
    targets,
    clone_weight=1.0,
    router_weight=0.1
):

    main_logits = output["logits"]

    main_loss = F.cross_entropy(
        main_logits,
        targets
    )

    clone_loss = torch.tensor(
        0.0,
        device=targets.device
    )

    if (
        output.get("clone_logits") is not None
        and output.get("clone_targets") is not None
    ):

        clone_loss = F.cross_entropy(
            output["clone_logits"],
            output["clone_targets"]
        )

    router_target = targets.float().unsqueeze(1)

    router_loss = F.binary_cross_entropy(
        output["gate"],
        router_target
    )

    total_loss = (
        main_loss
        + clone_weight * clone_loss
        + router_weight * router_loss
    )

    return (
        total_loss,
        main_loss.detach(),
        clone_loss.detach(),
        router_loss.detach()
    )


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_crushnet(
    X_train,
    y_train,
    X_val,
    y_val,
    input_dim,
    epochs=100,
    batch_size=64,
    learning_rate=0.001,
    clone_weight=1.0,
    router_weight=0.1,
    patience=12,
    device=None,
    seed=42,
    save_path=None,
    verbose=True
):

    torch.manual_seed(seed)
    np.random.seed(seed)

    if device is None:

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    model = CRUSHNet(
        input_dim=input_dim
    ).to(device)

    X_train = torch.as_tensor(
        X_train,
        dtype=torch.float32
    )

    y_train = torch.as_tensor(
        y_train,
        dtype=torch.long
    )

    X_val = torch.as_tensor(
        X_val,
        dtype=torch.float32
    )

    y_val = torch.as_tensor(
        y_val,
        dtype=torch.long
    )

    train_dataset = TensorDataset(
        X_train,
        y_train
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    history = []

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):

        model.train()

        train_total = 0.0
        train_main = 0.0
        train_clone = 0.0
        train_router = 0.0
        train_batches = 0

        for xb, yb in train_loader:

            xb = xb.to(device)
            yb = yb.to(device)

            output = model(
                xb,
                yb,
                generate_clones=True
            )

            (
                loss,
                main_loss,
                clone_loss,
                router_loss
            ) = compute_loss(
                output,
                yb,
                clone_weight,
                router_weight
            )

            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0
            )

            optimizer.step()

            train_total += loss.item()
            train_main += main_loss.item()
            train_clone += clone_loss.item()
            train_router += router_loss.item()

            train_batches += 1

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        model.eval()

        with torch.no_grad():

            Xv = X_val.to(device)
            yv = y_val.to(device)

            val_output = model(
                Xv,
                yv,
                generate_clones=False
            )

            val_loss = F.cross_entropy(
                val_output["logits"],
                yv
            ).item()

        row = {
            "epoch": epoch,
            "loss": train_total / train_batches,
            "main_loss": train_main / train_batches,
            "clone_loss": train_clone / train_batches,
            "router_loss": train_router / train_batches,
            "val_loss": val_loss
        }

        history.append(row)

        if verbose:

            print(
                f"Epoch {epoch:03d}/{epochs} | "
                f"Loss={row['loss']:.4f} | "
                f"Val={val_loss:.4f}"
            )

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:

            if verbose:
                print(
                    f"Early stopping at epoch {epoch}."
                )

            break

    if best_state is not None:

        model.load_state_dict(
            best_state
        )

    history_df = pd.DataFrame(
        history
    )

    if save_path is not None:

        save_path = Path(
            save_path
        )

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        history_df.to_csv(
            save_path,
            index=False
        )

    return model, history_df


# ============================================================
# PREDICTION
# ============================================================

def predict_crushnet(
    model,
    X,
    device=None
):

    if device is None:

        device = next(
            model.parameters()
        ).device

    model.eval()

    X = torch.as_tensor(
        X,
        dtype=torch.float32
    ).to(device)

    with torch.no_grad():

        output = model(X)

        logits = output["logits"]

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        predictions = torch.argmax(
            probabilities,
            dim=1
        )

        gate = output["gate"]

    return (
        predictions.cpu().numpy(),
        probabilities.cpu().numpy(),
        gate.cpu().numpy().ravel()
    )


# ============================================================
# LOAD MODEL
# ============================================================

def save_model(
    model,
    path
):

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save(
        model.state_dict(),
        path
    )


def load_model(
    input_dim,
    path,
    device=None
):

    if device is None:

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    model = CRUSHNet(
        input_dim=input_dim
    ).to(device)

    state = torch.load(
        path,
        map_location=device
    )

    model.load_state_dict(
        state
    )

    model.eval()

    return model