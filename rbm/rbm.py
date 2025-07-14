from tinygrad import Tensor

class RBM:
    """
    Implémentation d'une Machine de Boltzmann Restreinte (RBM) utilisant tinygrad.
    Cette classe permet l'entraînement d'un modèle RBM pour des tâches comme
    la recommandation de livres.
    """

    def __init__(self, visible_units, hidden_units, learning_rate=0.6, batch_size=1):
        """
        Initializes a new RBM instance.
        """
        self.visible_units = visible_units
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Initialization of weights and biases
        self.vb = Tensor.zeros(visible_units)  # Bias for visible units
        self.hb = Tensor.zeros(hidden_units)   # Bias for hidden units
        self.W = Tensor.randn(visible_units, hidden_units) * 0.01  # Weight matrix

    def sample_hidden(self, v):
        """
        Samples the state of the hidden layer given the state of the visible layer.
        """
        # Calculation of hidden layer activations
        h_activation = (Tensor.matmul(v, self.W) + self.hb).sigmoid()
        # Gibbs sampling
        h_sample = (Tensor.sign(h_activation - Tensor.uniform(Tensor.size(h_activation)))).relu()

        return h_activation, h_sample

    def sample_visible(self, h):
        """
        Samples the state of the visible layer given the state of the hidden layer.
        """
        # Calculation of visible layer activations
        v_activation = (Tensor.matmul(h, self.W.T) + self.vb).sigmoid()
        # Gibbs sampling
        v_sample = (Tensor.sign(v_activation - Tensor.uniform(Tensor.size(v_activation)))).relu()

        return v_activation, v_sample

    def contrastive_divergence(self, input_data):
        """
        Performs a contrastive divergence step for learning.
        """
        # Create a tensor from input data
        v0 = input_data

        # Phase positive (forward pass)
        h0_activation, h0 = self.sample_hidden(v0)

        # Negative phase (reconstruction)
        v1_activation, v1 = self.sample_visible(h0)
        h1_activation, h1 = self.sample_hidden(v1)

        # Gradient calculation
        w_pos_grad = Tensor.matmul(v0.T, h0)
        w_neg_grad = Tensor.matmul(v1.T, h1)

        # Calculation of contrastive divergence
        cd = (w_pos_grad - w_neg_grad) / (Tensor.size(v0)[0])

        # Updating weights and biases
        self.W = self.W + self.learning_rate * cd
        self.vb = self.vb + self.learning_rate * (v0 - v1).mean(axis=0)
        self.hb = self.hb + self.learning_rate * (h0 - h1).mean(axis=0)

        # Error calculation
        err = v0 - v1
        err_sum = (err * err).mean(axis=0)

        return err_sum.mean()



    def free_energy(self, v_sample):
        """
        Calculate the free energy of a sample.
        """
        wx_b = Tensor.matmul(v_sample, self.W) + self.hb
        vbias_term = Tensor.matmul(v_sample, self.vb.reshape(-1, 1))
        hidden_term = Tensor.log(1 + Tensor.exp(wx_b)).sum(axis=1)

        return -hidden_term - vbias_term

    def reconstruct(self, v):
        """
        Reconstructs the state of the visible layer from the state of the hidden layer.
        """
        _, h = self.sample_hidden(v)
        v_reconstructed, _ = self.sample_visible(h)
        return v_reconstructed