# =========================================================================
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================


import torch
from torch import nn
from fuxictr.pytorch.layers import FeatureEmbedding


class LogisticRegression(nn.Module):
    """Logistic Regression layer that learns a linear weighted sum of input features.

    ``LogisticRegression`` uses 1-dimensional embeddings as feature weights and sums them
    with an optional bias term to produce a linear prediction.

    Args:
        feature_map (FeatureMap): A ``FeatureMap`` instance that provides feature metadata.
        use_bias (bool, optional): Whether to include a learnable bias term. Default: ``True``.

    Example::

        lr = LogisticRegression(feature_map, use_bias=True)
        output = lr(X)
    """

    def __init__(self,
                 feature_map,
                 use_bias=True,
                 numeric_embedding_dim=None):
        super(LogisticRegression, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1), requires_grad=True) if use_bias else None
        self.numeric_embedding_dim = (
            int(numeric_embedding_dim)
            if numeric_embedding_dim is not None
            else None
        )
        if self.numeric_embedding_dim is not None and self.numeric_embedding_dim <= 0:
            raise ValueError("numeric_embedding_dim must be a positive integer.")

        self.feature_names = [
            feature
            for feature, spec in feature_map.features.items()
            if spec["type"] != "meta"
        ]
        self.numeric_features = [
            feature
            for feature in self.feature_names
            if feature_map.features[feature]["type"] == "numeric"
        ]
        self.numeric_feature_indices = [
            self.feature_names.index(feature)
            for feature in self.numeric_features
        ]

        if self.numeric_embedding_dim is None:
            # Original LR path: numeric values are multiplied by scalar weights.
            self.embedding_layer = FeatureEmbedding(
                feature_map,
                1,
                use_pretrain=False,
                use_sharing=False,
            )
            self.numeric_weight = None
        else:
            # Keep native scalar lookup for non-numeric fields. Numeric fields
            # come from the model's shared, full-dimensional embedding tensor.
            non_numeric_features = [
                feature
                for feature in self.feature_names
                if feature not in self.numeric_features
            ]
            self.embedding_layer = (
                FeatureEmbedding(
                    feature_map,
                    1,
                    required_feature_columns=non_numeric_features,
                    use_pretrain=False,
                    use_sharing=False,
                )
                if non_numeric_features
                else None
            )
            self.numeric_weight = (
                nn.Parameter(
                    torch.empty(
                        len(self.numeric_features),
                        self.numeric_embedding_dim,
                    )
                )
                if self.numeric_features
                else None
            )

    def init_weights(self):
        if self.numeric_weight is not None:
            nn.init.xavier_normal_(self.numeric_weight)

    def forward(self, X, feature_emb=None):
        """Compute the LR output.

        Args:
            X (dict): Raw feature inputs.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        if self.numeric_embedding_dim is None:
            embed_weights = self.embedding_layer(X)
            output = embed_weights.sum(dim=1)
        else:
            if feature_emb is None:
                raise ValueError(
                    "feature_emb is required when numeric_embedding_dim is enabled."
                )
            if feature_emb.ndim != 3:
                raise ValueError(
                    "feature_emb must have shape "
                    "(batch_size, num_fields, embedding_dim)."
                )
            expected_shape = (
                len(self.feature_names),
                self.numeric_embedding_dim,
            )
            if tuple(feature_emb.shape[1:]) != expected_shape:
                raise ValueError(
                    "feature_emb has incompatible field/dimension shape "
                    f"{tuple(feature_emb.shape[1:])}; expected {expected_shape}."
                )

            if self.embedding_layer is not None:
                embed_weights = self.embedding_layer(X)
                output = embed_weights.sum(dim=1)
            else:
                output = feature_emb.new_zeros((feature_emb.size(0), 1))

            if self.numeric_weight is not None:
                numeric_emb = feature_emb[:, self.numeric_feature_indices, :]
                numeric_output = torch.einsum(
                    "bfd,fd->b",
                    numeric_emb,
                    self.numeric_weight,
                ).unsqueeze(-1)
                output = output + numeric_output
        if self.bias is not None:
            output += self.bias
        return output

