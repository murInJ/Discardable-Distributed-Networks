import torch


def calculate_ifc(feature_blocks):
    num_blocks = len(feature_blocks)
    ifc_scores = torch.zeros(num_blocks, num_blocks)

    for i in range(num_blocks):
        for j in range(num_blocks):
            if i != j:
                original_block = torch.tensor(feature_blocks[i])
                distorted_block = torch.tensor(feature_blocks[j])

                # Calculate mean and standard deviation for each channel
                original_mean = torch.mean(original_block, dim=(0, 1))
                distorted_mean = torch.mean(distorted_block, dim=(0, 1))
                original_std = torch.std(original_block, dim=(0, 1))
                distorted_std = torch.std(distorted_block, dim=(0, 1))

                # Calculate IFC for each channel
                ifc_channel = 1 - torch.abs(original_mean - distorted_mean) / (original_std + distorted_std)

                # Average IFC scores across channels
                ifc_score = torch.mean(ifc_channel)
                ifc_scores[i, j] = ifc_score

    return ifc_scores