import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()
    
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

# START_TAG = "<START>"
# STOP_TAG = "<STOP>"
START_IDX = 74
STOP_IDX = 75
class BiLSTM_CRF(nn.Module):

    def __init__(self, config):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = config.embed_size
        self.hidden_dim = config.hidden_size
        self.vocab_size = config.vocab_size
        self.tag_pad_idx = config.tag_pad_idx
        # self.tag_to_ix = tag_to_ix
        self.tagset_size = config.num_tags + 2

        self.word_embed = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[START_IDX, :] = -10000
        self.transitions.data[:, STOP_IDX] = -10000

        self.hidden = self.init_hidden()
    def init_hidden(self, batch_size=32):
        return (torch.randn(2, batch_size, self.hidden_dim // 2),
                torch.randn(2, batch_size, self.hidden_dim // 2))

    def forward(self, batch):
        tag_ids = batch.tag_ids
        tag_mask = batch.tag_mask
        input_ids = batch.input_ids
        lengths = batch.lengths
        

        lstm_feats = self._get_lstm_features(input_ids)
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

    def neg_log_likelihood(self, batch):
        sentence = batch.input_ids
        tags = batch.tag_ids
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        print("n3")
        gold_score = self._score_sentence(feats, tags)
        print("n4")
        return torch.mean(forward_score - gold_score)


    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][START_IDX] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        # print(feats.shape)
        alphas = []

        for sentence in feats:
            for feat in sentence:
                alphas_t = []  # The forward tensors at this timestep
                for next_tag in range(self.tagset_size):
                    # broadcast the emission score: it is the same regardless of
                    # the previous tag
                    emit_score = feat[next_tag].view(
                        1, -1).expand(1, self.tagset_size)
                    # the ith entry of trans_score is the score of transitioning to
                    # next_tag from i
                    trans_score = self.transitions[next_tag].view(1, -1)
                    # The ith entry of next_tag_var is the value for the
                    # edge (i -> next_tag) before we do log-sum-exp
                    next_tag_var = forward_var + trans_score + emit_score
                    # The forward variable for this tag is log-sum-exp of all the
                    # scores.
                    alphas_t.append(log_sum_exp(next_tag_var).view(1))
                forward_var = torch.cat(alphas_t).view(1, -1)
            terminal_var = forward_var + self.transitions[STOP_IDX]
            alphas.append(log_sum_exp(terminal_var).view(1))
        # print(alphas)
        alpha = torch.cat(alphas)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden(sentence.shape[0])
        print(self.hidden.shape)
        embeds = self.word_embed(sentence)
        # print(embeds.shape)
        # print(self.hidden[0].shape)
        print("n1")
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(sentence.shape[0], sentence.shape[1], self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        print("n2")
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        # print(tags.shape)
        a = torch.full((tags.shape[0], 1), START_IDX, dtype=torch.long)
        # print(a.shape)
        tags = torch.cat([a, tags], dim=1)
        # print(tags.shape)
        # print(feats.shape) # [bsz, len, 74]
        scores = []
        for item in range(feats.shape[0]):
            for i, feat in enumerate(feats[item]):
                score += self.transitions[tags[item][i + 1], tags[item][i]] + feat[tags[item][i + 1]]
            score += self.transitions[STOP_IDX, tags[item][-1]]
            scores.append(score.view(1))
        return torch.cat(scores)

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][START_IDX] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[STOP_IDX]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == START_IDX  # Sanity check
        best_path.reverse()
        return path_score, best_path