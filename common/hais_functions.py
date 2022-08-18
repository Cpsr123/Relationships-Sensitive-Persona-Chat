import numpy as np
import torch
import torch.nn.functional as f



def get_matching_output(matching_margin, input_tensor, response_tensor):
    """Get loss for the matching prediction."""
    batch_size = input_tensor.shape[0]
    if batch_size == 1:
        raise ValueError("Batch size must be greater than 2.")

    # make negative
    response_tensor_1, response_tensor_2 = torch.split(response_tensor, [batch_size - 1, 1], dim=0)
    nega_response_tensor = torch.cat((response_tensor_2, response_tensor_1), dim=0)

    posi_cosine_sims = f.cosine_similarity(input_tensor, response_tensor)
    nega_cosine_sims = f.cosine_similarity(input_tensor, nega_response_tensor)

    print('input_tensor:', input_tensor)
    print('response_tensor:', response_tensor)
    print('posi_cosine_sims:', posi_cosine_sims)
    print('nega_cosine_sims:', nega_cosine_sims)

    losses = f.relu(matching_margin - posi_cosine_sims + nega_cosine_sims)
    print('losses:', losses)
    return losses.mean()

def evaluate(contexts, responses, context_index):

    assert len(contexts) == len(responses)

    permutation = np.arange(len(contexts)) # [0, 1, 2, 3]

    np.random.shuffle(permutation)
    context_shuffled = [contexts[j] for j in permutation] 
    context_shuffled_index = [context_index[j] for j in permutation] 
    contexts_shuffled_matrix = []
    responses_matrix = []
    for context, response in zip(context_shuffled, responses):
        contexts_shuffled_matrix.append(context)
        responses_matrix.append(response)

    predictions = rank_responses(
        contexts_shuffled_matrix, responses_matrix) # [1, 3, 0, 2]
    
    predictions_index = [context_index[j] for j in predictions]

    recall = np.equal(predictions, permutation)
    return recall, predictions_index, context_shuffled_index

def rank_responses(contexts_matrix, responses_matrix):
    """Rank the responses for each context, using cosine similarity."""
    contexts_matrix /= np.linalg.norm(
        contexts_matrix, axis=1, keepdims=True)
    responses_matrix /= np.linalg.norm(
        responses_matrix, axis=1, keepdims=True)
    similarities = np.matmul(contexts_matrix, responses_matrix.T)
    return np.argmax(similarities, axis=1)
