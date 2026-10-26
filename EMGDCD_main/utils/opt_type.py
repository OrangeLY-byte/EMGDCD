from dataclasses import dataclass


@dataclass
class MSCDopt:

    @dataclass
    class MSCDargs:
        n_nodes: int
        # Populated by the experiment entrypoint before model construction.
        n_sub: int
        input_step: int
        batch_size: int
        data_dim: int
        total_epoch: int
        max_grad_norm: float

        @dataclass
        class decoder:
            lr_dec: float
            concat_h: bool
            shared_weights_decoder: bool
            mlp_hid: int
            gru_layers: int
            weight_decay: float

        @dataclass
        class encoder:
            lr_enc: float
            hid_dim: int
            lambda_1: float
            max_backward_tau: float
