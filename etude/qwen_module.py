import lightning as L
import torch
from transformers import AutoModelForCausalLM, get_cosine_schedule_with_warmup

from etude.common import COMPUTE_DTYPE
from etude.hf_flash_attention import flash_attention_4_forward


class QwenRustPretrainer(L.LightningModule):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.5-0.8B-Base",
        learning_rate: float = 5e-5,
        warmup_steps: int = 500,
        weight_decay: float = 0.01,
        total_steps: int = 50_000,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=COMPUTE_DTYPE,
            attn_implementation="flash_attention_4",
        )

    def forward(self, **batch):
        return self.model(**batch)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log("train/ppl", torch.exp(loss), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.log("val/ppl", torch.exp(loss), prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        # Separate weight-decay groups
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        params = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            params, lr=self.hparams.learning_rate, betas=(0.9, 0.95)
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.hparams.total_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
