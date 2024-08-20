from typing import Dict, Iterator, List
import torch
import torch.nn as nn
from torch.autograd import Variable
from allennlp.modules.text_field_embedders import TextFieldEmbedder


class KNRM(nn.Module):
    '''
    Paper: End-to-End Neural Ad-hoc Ranking with Kernel Pooling, Xiong et al., SIGIR'17
    '''

    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 n_kernels: int,
                 device: torch.device = None):

        super(KNRM, self).__init__()

        self.word_embeddings = word_embeddings
        self.device = device

        # static - kernel size & magnitude variables
        mu = torch.FloatTensor(self.kernel_mus(n_kernels)).view(1, 1, 1, n_kernels)
        sigma = torch.FloatTensor(self.kernel_sigmas(n_kernels)).view(1, 1, 1, n_kernels)

        self.register_buffer('mu', mu)
        self.register_buffer('sigma', sigma)

        #todo#
        # Initialize the linear transformation layer for kernel pooling
        self.mlp = nn.Linear(n_kernels, 1, bias=False)


    def forward(self, query: Dict[str, torch.Tensor], document: Dict[str, torch.Tensor]) -> torch.Tensor:
        # pylint: disable=arguments-differ

        #
        # prepare embedding tensors & paddings masks
        # -------------------------------------------------------

        query_pad_oov_mask = (query["tokens"]["tokens"] > 0).float().to(self.device)
        #Query mask shape: torch.Size([batch_size, max_query_length])  
        document_pad_oov_mask = (document["tokens"]["tokens"] > 0).float().to(self.device)
        #Document mask shape: torch.Size([batch_size, max_document_length])
    
        # Get the embeddings for the query and document
        query_embeddings = self.word_embeddings(query).to(self.device)   
        #query_embeddings shape: torch.Size([batch, query_max, emb_dim])
        document_embeddings = self.word_embeddings(document).to(self.device) 
        #document embedding shape: torch.Size([batch, doc_max, emb_dim])

        
        #todo###

        # Compute the similarity matrix
        similarity_matrix = torch.bmm(query_embeddings, document_embeddings.transpose(1, 2))
        #Similarity matrix shape: torch.Size([batch, query_max, doc_max])


        # Kernel Pooling
        # -------------------------------------------------------

        # Calculate similarity matrix using Gaussian kernel
        
        #expand mu and sigma to match batch size and sequence lengths
        mu = self.mu.expand(similarity_matrix.size(0), -1, -1, -1)
        # mu: (batch_size, 1, query_length, n_kernels)
        sigma = self.sigma.expand(similarity_matrix.size(0), -1, -1, -1)
        # sigma: (batch_size, 1, document_length, n_kernels)

        # Compute Gaussian kernel scores
        K = torch.exp(-0.5 * ((similarity_matrix.unsqueeze(3) - mu) ** 2 / (sigma ** 2)))
        #K shape: torch.Size([batch_size, query_length, document_length, n_kernels]) 

        # Mask out padding and OOV positions in both query and document
        query_mask = query_pad_oov_mask.unsqueeze(2).unsqueeze(3)
        document_mask = document_pad_oov_mask.unsqueeze(1).unsqueeze(3)
        #Query mask shape unsqueezed: torch.Size([Batch_size, query_length, 1, 1])
        #Document mask shape unsqueezed: torch.Size([Batch_size, 1, doc_length, 1])

        # Perform element-wise multiplication with correctly shaped masks
        K = K * query_mask  # torch.Size(batch_size, query_length, document_length, n_kernels)
        K = K * document_mask # torch.Size(batch_size, query_length, document_length, n_kernels)
        
        #K = query_masked * document_masked
        K = K * query_mask * document_mask
        #K shape: torch.Size([batch_size, query_length, document_length, n_kernels])

        # Sum over document length and take logarithm
        log_sum_exp = torch.sum(torch.log1p(K), dim=2)
        #log_sum_exp shape: torch.Size([batch_size, query_length, n_kernels])
        
        # Apply MLP to aggregate kernel scores across different kernels
        output = self.mlp(log_sum_exp).squeeze(-1)
        #output shape: torch.Size([batch_size, query_length])

        #sum along the query_length dimension
        return output.sum(dim=1)   #torch.Size([batch_size])

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
