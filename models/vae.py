import os
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

class VAE(nn.Module):
    def __init__(self, input_size, z_size):
        super(VAE, self).__init__()

        self.input_size = input_size
        self.z_size = z_size

        # Encoder
        self.fc1 = nn.Linear(self.input_size, 400)
        
        # Bottlenecks layers for mean & variance
        self.fc21 = nn.Linear(400, self.z_size)  # mu layer
        self.fc22 = nn.Linear(400, self.z_size)  # logvariance layer

        # # ENCODER
        # # 28 x 28 pixels = 784 input pixels, 400 outputs
        # self.fc1 = nn.Linear(self.input_size, 400)
        # # rectified linear unit layer from 400 to 400
        # # max(0, x)
        # self.relu = nn.ReLU()
        # self.fc21 = nn.Linear(400, self.z_size)  # mu layer
        # self.fc22 = nn.Linear(400, self.z_size)  # logvariance layer
        # # this last layer bottlenecks through ZDIMS connections

        # Decoder
        self.fc3 = nn.Linear(self.z_size, 400)
        self.fc4 = nn.Linear(400, self.input_size)

    def encode(self, x: Variable) -> (Variable, Variable):
        """Input vector x -> fully connected 1 -> ReLU -> (fully connected
        21, fully connected 22)

        Parameters
        ----------
        x : [128, 784] matrix; 128 digits of 28x28 pixels each

        Returns
        -------

        (mu, logvar) : ZDIMS mean units one for each latent dimension, ZDIMS
            variance units one for each latent dimension

        """
        h1 = F.relu(self.fc1(x))
        mu = self.fc21(h1)
        logvar = self.fc22(h1)
        return mu, logvar

    def reparameterize(self, mu: Variable, logvar: Variable) -> Variable:
        """THE REPARAMETERIZATION IDEA:

        For each training sample (we get 128 batched at a time)

        - take the current learned mu, stddev for each of the ZDIMS
          dimensions and draw a random sample from that distribution
        - the whole network is trained so that these randomly drawn
          samples decode to output that looks like the input
        - which will mean that the std, mu will be learned
          *distributions* that correctly encode the inputs
        - due to the additional KLD term (see loss_function() below)
          the distribution will tend to unit Gaussians

        Parameters
        ----------
        mu : [128, ZDIMS] mean matrix
        logvar : [128, ZDIMS] variance matrix

        Returns
        -------

        During training random sample from the learned ZDIMS-dimensional
        normal distribution; during inference its mean.

        """

        if self.training:
            # multiply log variance with 0.5, then in-place exponent
            # yielding the standard deviation
            # std = logvar.mul(0.5).exp_()
            std = torch.exp(0.5*logvar)

            # - std.data is the [128,ZDIMS] tensor that is wrapped by std
            # - so eps is [128,ZDIMS] with all elements drawn from a mean 0
            #   and stddev 1 normal distribution that is 128 samples
            #   of random ZDIMS-float vectors
            # eps = Variable(std.data.new(std.size()).normal_())
            eps = torch.randn_like(std)

            # - sample from a normal distribution with standard
            #   deviation = std and mean = mu by multiplying mean 0
            #   stddev 1 sample with desired std and mu, see
            #   https://stats.stackexchange.com/a/16338
            # - so we have 128 sets (the batch) of random ZDIMS-float
            #   vectors sampled from normal distribution with learned
            #   std and mu for the current input
            # return eps.mul(std).add_(mu)
            return mu + eps*std
        else:
            # During inference, we simply spit out the mean of the
            # learned distribution for the current input.  We could
            # use a random sample from the distribution, but mu of
            # course has the highest probability.
            return mu

    def decode(self, z: Variable) -> Variable:
        h3 = F.relu(self.fc3(z))
        # return self.sigmoid(self.fc4(h3))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x: Variable) -> (Variable, Variable, Variable):
        mu, logvar = self.encode(x.view(-1, self.input_size))
        z = self.reparameterize(mu, logvar)

        # if self.training:
        return self.decode(z), mu, logvar
        # else:
        #     return z, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar) -> Variable:
        # how well do input x and output recon_x agree?
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, self.input_size))

        # KLD is Kullback–Leibler divergence -- how much does one learned
        # distribution deviate from another, in this specific case the
        # learned distribution from the unit Gaussian

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # - D_{KL} = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # note the negative D_{KL} in appendix B of the paper
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
        # # Normalise by same number of elements as in reconstruction
        # KLD /= BATCH_SIZE * (self.input_size)

        # BCE tries to make our reconstruction as accurate as possible
        # KLD tries to push the distributions as close as possible to unit Gaussian
        return BCE + KLD