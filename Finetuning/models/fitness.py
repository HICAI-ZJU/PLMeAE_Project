from asyncore import read
from collections import OrderedDict

import torch
from fairseq.models import FairseqEncoderDecoderModel
from fairseq.models import register_model, register_model_architecture

from .encoder import ProteinBertEncoder
from .decoder import ProteinRegressionDecoder


@register_model("fitness_model")
class DownstreamModel(FairseqEncoderDecoderModel):
    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--layer_num", default=33, type=int, metavar="N", help="number of layers"
        )
        parser.add_argument(
            "--embed_dim", default=1280, type=int, metavar="N", help="embedding dimension"
        )
        parser.add_argument(
            "--ffn_embed_dim",
            default=5120,
            type=int,
            metavar="N",
            help="embedding dimension for FFN",
        )
        parser.add_argument(
            "--attention_head_num",
            default=20,
            type=int,
            metavar="N",
            help="number of attention heads",
        )
        parser.add_argument("--max_position_num", default=1024, type=int, help="number of positional embeddings to learn")
        parser.add_argument("--emb_layer_norm_before", default=True, type=bool)
        parser.add_argument("--checkpoint_path", type=str)

    @classmethod
    def build_model(cls, args, task):
        encoder = ProteinBertEncoder(args, args.max_position_num, args.layer_num, args.attention_head_num, args.embed_dim, args.ffn_embed_dim, task.alphabet)
        decoder = ProteinRegressionDecoder(task.alphabet, args.embed_dim)
        model = DownstreamModel(encoder, decoder)

        
        with torch.no_grad():
            with open('/data/qm/model/esm1b_t33_650M_UR50S.pt', 'rb') as f:
            # with open('/data/wzy/workspace/MA-tuning/esm1v_t33_650M_UR90S_1.pt', 'rb') as f:
                pretrain_net_dict = torch.load(f, map_location=torch.device('cpu'))
                new_state_dict = OrderedDict()
                for k, v in pretrain_net_dict['model'].items():
                    name = k.replace(".sentence_encoder", "")
                    if 'lm_head' in name:
                        name = name.replace("encoder", "decoder")
                    # elif name == 'decoder.lm_head.weight':
                    #     decoder.lm_head.weight[:33, :] = v
                    new_state_dict[name] = v
            model.load_state_dict(new_state_dict, strict=False)
            print("load esm1b successfully!")


        return model

    def forward(self, tokens):
        encoder_out = self.encoder.forward(tokens)
        decoder_out = self.decoder.forward(encoder_out['logits'], tokens)
        return decoder_out
    def forward_encoder(self, tokens):
        return self.encoder.forward(tokens)
        
   
    def forward_decoder(self, prev_result, tokens, readout='rnn'):

        # import pdb; pdb.set_trace()

        if readout == 'rnn':
            decoder_out = self.decoder.forward_rnn(prev_result['logits'], tokens)
        elif readout == 'mlp':
            decoder_out = self.decoder.forward_mlp(prev_result['logits'], tokens)
        elif readout == 'mean':
            decoder_out = self.decoder.forward_mean(prev_result['logits'], tokens)
        elif readout == 'sum':
            decoder_out = self.decoder.forward_sum(prev_result['logits'], tokens)
        else:
            raise NotImplementedError("This readout is not implemented!")
        return decoder_out


@register_model_architecture('fitness_model', 'fitness_esm1b')
def esm_1b(args):
    args.layer_num = getattr(args, 'layer_num', 33)
    args.embed_dim = getattr(args, 'embed_dim', 1280)
    args.ffn_embed_dim = getattr(args, 'ffn_embed_dim', 5120)
    args.attention_head_num = getattr(args, 'attention_head_num', 20)
    args.max_position_num = getattr(args, 'max_position_num', 1024)
    args.emb_layer_norm_before = getattr(args, 'emb_layer_norm_before', True)