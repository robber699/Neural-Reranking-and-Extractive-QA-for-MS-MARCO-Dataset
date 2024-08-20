from typing import Dict, Iterator, List

import torch
import torch.nn as nn

from allennlp.modules.text_field_embedders import TextFieldEmbedder

from positional_encoding import PositionalEncoding


class TK(nn.Module):
    '''
    Paper: S. Hofstätter, M. Zlabinger, and A. Hanbury 2020. Interpretable & Time-Budget-Constrained Contextualization for Re-Ranking. In Proc. of ECAI 
    '''

    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 n_kernels: int,
                 n_layers: int,
                 n_tf_dim: int,
                 n_tf_heads: int):

        super(TK, self).__init__()

        self.device = ("cuda"
                       if torch.cuda.is_available()
                       else "cpu"
                       )

        self.word_embeddings = word_embeddings

        # static - kernel size & magnitude variables
        mu = torch.FloatTensor(self.kernel_mus(n_kernels)).view(1, 1, 1, n_kernels).to(self.device)
        sigma = torch.FloatTensor(self.kernel_sigmas(n_kernels)).view(1, 1, 1, n_kernels).to(self.device)

        self.register_buffer('mu', mu)
        self.register_buffer('sigma', sigma)

        # positional encoder
        self.positional_encoder = PositionalEncoding(d_model=n_tf_dim)

        # initialize transformer for contextualization
        encoder_layer = nn.TransformerEncoderLayer(d_model=n_tf_dim, nhead=n_tf_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.a = nn.Parameter(torch.full([1, 1, 1], 0.5, requires_grad=True, device=self.device))

        self.mlp = nn.Linear(n_kernels, 1, bias=False)

    def forward(self, query: Dict[str, torch.Tensor], document: Dict[str, torch.Tensor]) -> torch.Tensor:
        # pylint: disable=arguments-differ

        #
        # prepare embedding tensors & paddings masks
        # -------------------------------------------------------

        # shape: (batch, query_max)
        query_pad_oov_mask = (query["tokens"]["tokens"] > 0).float()  # > 1 to also mask oov terms
        # shape: (batch, doc_max)
        document_pad_oov_mask = (document["tokens"]["tokens"] > 0).float()

        # shape: (batch, query_max,emb_dim)
        query_embeddings = self.word_embeddings(query)
        # shape: (batch, document_max,emb_dim)
        document_embeddings = self.word_embeddings(document)

        #
        # contextualization phase
        # -------------------------------------------------------

        # mask oov terms
        # unsqueeze(-1) to match dimension of embeddings, append an extra dimension
        # shape: (batch, query_max,emb_dim)
        query_embeddings = query_embeddings * query_pad_oov_mask.unsqueeze(-1)
        # shape: (batch, document_max,emb_dim)
        document_embeddings = document_embeddings * document_pad_oov_mask.unsqueeze(-1)

        # create positional encodings
        # query_positional shape: (batch, query_max, emb_dim)
        # transpose since positional_encoder expects shape [seq_len, batch_size, embedding_dim]
        query_positional = self.positional_encoder(query_embeddings.transpose(0, 1)).transpose(0, 1)
        # query_positional shape: (batch, document_max, emb_dim)
        document_positional = self.positional_encoder(document_embeddings.transpose(0, 1)).transpose(0, 1)

        # contextualize with transformer
        # query_embeddings_context shape: (batch, query_max, emb_dim)
        query_embeddings_context = self.transformer(query_embeddings + query_positional)
        # document_embeddings_context shape: (batch, document_max, emb_dim)
        document_embeddings_context = self.transformer(document_embeddings + document_positional)

        # We regulate the influence of the contextualization by the end-to-end
        # learned parameter a. This allows the model to decide the intensity
        # of the contextualization
        query_embeddings = (self.a * query_embeddings + (
                1 - self.a) * query_embeddings_context) * query_pad_oov_mask.unsqueeze(-1)
        document_embeddings = (self.a * document_embeddings + (
                1 - self.a) * document_embeddings_context) * document_pad_oov_mask.unsqueeze(-1)

        #
        # similarity matrix
        # -------------------------------------------------------

        # compute the similarity matrix
        similarity_matrix = torch.bmm(query_embeddings, document_embeddings.transpose(1, 2))


        #
        # Kernel Pooling
        # -------------------------------------------------------

        # transform each entry in the similarity matrix with a set of RBF-kernels
        # each kernel focuses on a specific similarity range with center mu
        # K shape: (batch, query_max, doc_max, n_kernels)
        K = torch.exp(- torch.pow(similarity_matrix.unsqueeze(-1) - self.mu, 2) / (2 * torch.pow(self.sigma, 2)))

        query_mask = query_pad_oov_mask.unsqueeze(2).unsqueeze(3)
        document_mask = document_pad_oov_mask.unsqueeze(1).unsqueeze(3)

        # Mask out padding and OOV positions in both query and document
        # K_masked shape: (batch, query_max, doc_max, n_kernels)
        K_masked = K * query_mask * document_mask

        # summing document dimension for each query and kernel
        # query_kernel shape: (batch, query_max, n_kernels)
        query_kernel = torch.sum(K_masked, 2)

        #
        # scoring
        # -------------------------------------------------------

        # log normalization
        # apply a logarithm to each query term
        log_query_kernel = torch.log(query_kernel)
        log_query_kernel_masked = log_query_kernel * query_mask.squeeze(-1)

        # sum up query terms for each kernel
        # per_kernel shape: (batch, n_kernels)
        per_kernel = torch.sum(log_query_kernel_masked, 1)
        # calculate weight and sum up kernel scores
        # score_log shape: (batch)
        score_log = self.mlp(per_kernel).squeeze(1)

        # length normalization
        # divide query kernels by doc len
        len_norm_per_kernel_query = query_kernel / K_masked.size()[2]
        # sum up query terms for each kernel
        # len_norm_per_kernel_query shape: (batch, n_kernels)
        len_norm_per_kernel_query = torch.sum(len_norm_per_kernel_query, 1)
        # calculate weight and sum up kernel scores
        # score_len shape: (batch)
        score_len = self.mlp(len_norm_per_kernel_query).squeeze(1)

        # sum up scores
        # output shape: (batch)
        output = score_log + score_len
        return output


    def kernel_mus(self, n_kernels: int):
        """
        get the mu for each guassian kernel. Mu is the middle of each bin
        :param n_kernels: number of kernels (including exact match). first one is exact match
        :return: l_mu, a list of mu.
        """
        l_mu = [1.0]
        if n_kernels == 1:
            return l_mu

        bin_size = 2.0 / (n_kernels - 1)  # score range from [-1, 1]
        l_mu.append(1 - bin_size / 2)  # mu: middle of the bin
        for i in range(1, n_kernels - 1):
            l_mu.append(l_mu[i] - bin_size)
        return l_mu

    def kernel_sigmas(self, n_kernels: int):
        """
        get sigmas for each guassian kernel.
        :param n_kernels: number of kernels (including exactmath.)
        :param lamb:
        :param use_exact:
        :return: l_sigma, a list of simga
        """
        bin_size = 2.0 / (n_kernels - 1)
        l_sigma = [0.0001]  # for exact match. small variance -> exact match
        if n_kernels == 1:
            return l_sigma

        l_sigma += [0.5 * bin_size] * (n_kernels - 1)
        return l_sigma
